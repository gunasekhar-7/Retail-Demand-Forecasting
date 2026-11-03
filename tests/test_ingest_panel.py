from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from demand.config import Config
from demand.data.ingest import IngestError, _day_columns, validate_layout
from demand.data.panel import PanelError, read_panel, validate_panel


def test_validate_layout_reports_scope(demo_config: Config) -> None:
    report = validate_layout(demo_config)
    assert report.n_series == 2
    assert report.n_series_in_scope == 2
    assert report.n_days == 260
    assert report.start_date == "2011-01-29"
    assert report.end_date == "2011-10-15"
    assert report.calendar_end_date == "2011-11-12"
    assert report.calendar_extra_days == 28
    assert report.synthetic is True


def test_validate_layout_rejects_missing_files(demo_config: Config, tmp_path: Path) -> None:
    cfg = demo_config.model_copy(deep=True)
    cfg.data.raw_dir = tmp_path
    with pytest.raises(IngestError, match="missing required file"):
        validate_layout(cfg)


def test_validate_layout_rejects_calendar_gap(demo_config: Config, tmp_path: Path) -> None:
    cfg = demo_config.model_copy(deep=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv"):
        (raw / name).write_bytes((demo_config.data.raw_dir / name).read_bytes())
    calendar = pd.read_csv(raw / "calendar.csv")
    calendar.drop(index=10).to_csv(raw / "calendar.csv", index=False)
    cfg.data.raw_dir = raw
    with pytest.raises(IngestError, match="calendar has"):
        validate_layout(cfg)


def test_day_columns_must_be_contiguous() -> None:
    assert _day_columns(["id", "d_1", "d_2"]) == ["d_1", "d_2"]
    with pytest.raises(IngestError, match="contiguous"):
        _day_columns(["id", "d_1", "d_3"])


def test_panel_is_balanced_and_typed(demo_config: Config) -> None:
    frame = read_panel(demo_config.data.panel_path)
    stats = validate_panel(frame)
    assert stats["n_series"] == 2
    assert stats["n_days"] == 260
    assert {"series_id", "date", "sales", "sell_price", "snap_CA"}.issubset(frame.columns)
    assert frame["sales"].dtype.kind in {"i", "u"}
    assert not frame["date"].is_monotonic_increasing  # sorted by series, then date


def test_panel_duplicate_rows_are_rejected(demo_config: Config) -> None:
    frame = read_panel(demo_config.data.panel_path)
    duplicated = pd.concat([frame, frame.head(3)], ignore_index=True)
    with pytest.raises(PanelError, match="duplicate"):
        validate_panel(duplicated)
