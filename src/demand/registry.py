"""File-based model registry.

A run directory is self-describing: model bytes, the frozen feature spec, the
serving history and calendar slice it was trained with, and a metadata file with
the git SHA, cutoff and validation numbers. `latest.json` is the only mutable
pointer, so a bad deploy is a one-line revert.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from demand import __version__
from demand.config import PROJECT_ROOT, Config
from demand.features.lags import DAYS_SINCE_LAST_SALE_CAP
from demand.features.spec import FeatureSpec
from demand.models.base import Forecaster, ModelError
from demand.models.baseline import SeasonalNaiveForecaster
from demand.models.gbm import LgbmForecaster


class RegistryError(RuntimeError):
    """Raised when an artifact cannot be resolved or loaded."""


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    model: Forecaster
    spec: FeatureSpec
    history: pd.DataFrame
    calendar: pd.DataFrame
    metadata: dict[str, Any]

    @property
    def run_id(self) -> str:
        return str(self.metadata["run_id"])

    @property
    def horizon(self) -> int:
        return int(self.metadata["horizon"])

    @property
    def cutoff(self) -> pd.Timestamp:
        return pd.Timestamp(self.metadata["cutoff"])

    @property
    def series(self) -> list[str]:
        return sorted(self.history["series_id"].unique().tolist())


def git_revision(root: Path = PROJECT_ROOT) -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
        )
        return sha or "nogit", bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "nogit", False


def new_run_id(model: Forecaster, cutoff: pd.Timestamp) -> str:
    sha, _ = git_revision()
    return f"{model.name}-{pd.Timestamp(cutoff):%Y%m%d}-{sha}"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a file completely, then replace the destination atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_run(
    model: Forecaster,
    spec: FeatureSpec,
    history: pd.DataFrame,
    calendar: pd.DataFrame,
    cfg: Config,
    out_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> RunArtifact:
    """Write a complete run to staging, publish it atomically, then move latest.json."""
    cutoff = pd.Timestamp(history["date"].max()).normalize()
    artifacts_dir = out_dir or cfg.artifacts_dir
    runs_dir = artifacts_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    base_run_id = new_run_id(model, cutoff)
    sha, dirty = git_revision()
    base: Path | None = None
    run_id = base_run_id
    metadata: dict[str, Any] | None = None

    # Keep existing runs immutable. If this run id already exists, publish a
    # sibling with a short suffix instead of deleting the old artifact.
    for attempt in range(4):
        candidate_id = base_run_id if attempt == 0 else f"{base_run_id}-{uuid.uuid4().hex[:8]}"
        candidate = runs_dir / candidate_id
        if candidate.exists():
            candidate_id = f"{base_run_id}-{uuid.uuid4().hex[:8]}"
            candidate = runs_dir / candidate_id
        staging = Path(tempfile.mkdtemp(prefix=f".{candidate_id}.", dir=runs_dir))
        try:
            model.save(staging)
            spec.save(staging / "feature_spec.json")
            history.to_parquet(staging / "serving_history.parquet", index=False)
            calendar.to_parquet(staging / "calendar.parquet", index=False)
            metadata = {
                "run_id": candidate_id,
                "model": model.name,
                "horizon": spec.horizon,
                "cutoff": cutoff.strftime("%Y-%m-%d"),
                "created_at": datetime.now(UTC).isoformat(),
                "git_sha": sha,
                "git_dirty": dirty,
                "package_version": __version__,
                "dataset": {
                    "stores": cfg.data.stores,
                    "depts": cfg.data.depts,
                    "synthetic": cfg.data.synthetic,
                    "n_series": int(history["series_id"].nunique()),
                    "history_days": int(history["date"].nunique()),
                },
                "serving_history_days": int(history["date"].nunique()),
                "model_metadata": model.metadata(),
            }
            if extra:
                metadata.update(extra)
            (staging / "metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            try:
                os.replace(staging, candidate)
            except OSError:
                if candidate.exists() and attempt < 3:
                    continue
                raise
            base = candidate
            run_id = candidate_id
            break
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    if base is None or metadata is None:
        raise RegistryError("could not publish a complete run directory")
    latest = artifacts_dir / "latest.json"
    _atomic_write_text(
        latest,
        json.dumps({"run_id": run_id, "path": str(base)}, indent=2) + "\n",
    )
    return RunArtifact(
        path=base, model=model, spec=spec, history=history, calendar=calendar, metadata=metadata
    )


def _read_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metadata.json"
    if not path.exists():
        raise RegistryError(f"missing metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: Path) -> RunArtifact:
    metadata = _read_metadata(run_dir)
    spec = FeatureSpec.load(run_dir / "feature_spec.json")
    history = pd.read_parquet(run_dir / "serving_history.parquet")
    history["date"] = pd.to_datetime(history["date"]).dt.normalize()
    calendar = pd.read_parquet(run_dir / "calendar.parquet")
    kind = str(metadata.get("model", ""))
    if kind == "lightgbm" or kind.startswith("lightgbm_p"):
        model: Forecaster = LgbmForecaster.load(run_dir, spec)
    elif kind == "seasonal_naive":
        model = SeasonalNaiveForecaster(
            season_length=int(metadata.get("model_metadata", {}).get("season_length", 7))
        )
        model.fit(history, pd.Timestamp(metadata["cutoff"]))
    else:
        raise RegistryError(f"unsupported model kind in registry: {kind!r}")
    return RunArtifact(
        path=run_dir, model=model, spec=spec, history=history, calendar=calendar, metadata=metadata
    )


def load_latest(artifacts_dir: Path) -> RunArtifact:
    pointer = artifacts_dir / "latest.json"
    if not pointer.exists():
        raise RegistryError(f"no registered model under {artifacts_dir}; run `demand train` first")
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    run_dir = Path(payload["path"])
    if not run_dir.is_absolute():
        run_dir = artifacts_dir / "runs" / payload["run_id"]
    if not run_dir.exists():
        raise RegistryError(f"latest run points at a missing directory: {run_dir}")
    return load_run(run_dir)


def ensure_serving_history(panel: pd.DataFrame, days: int) -> pd.DataFrame:
    """Tail window that satisfies lags, horizon and the capped stale-sale feature."""
    if panel.empty:
        raise ModelError("cannot build serving history from an empty panel")
    days = max(days, DAYS_SINCE_LAST_SALE_CAP + 1)
    cutoff = pd.Timestamp(panel["date"].max())
    start = cutoff - pd.Timedelta(days=days)
    return panel[panel["date"] > start].sort_values(["series_id", "date"]).reset_index(drop=True)
