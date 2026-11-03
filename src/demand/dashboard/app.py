"""Streamlit dashboard.

Run with `demand dashboard`. Every heavy object is cached: the Parquet read with
`st.cache_data`, aggregates with `st.cache_data`, and the filter panel is a
fragment so changing a selector does not rebuild the whole page.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from demand.dashboard.data_access import (
    DashboardData,
    error_histogram,
    fold_table,
    horizon_curve,
    kpi_table,
    load_data,
    series_frame,
    series_summary,
    worst_series,
)

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📦", layout="wide")


@st.cache_data(ttl=300, show_spinner="Loading backtest artifacts…")
def _load(artifacts_dir: str, _stamp: float) -> DashboardData:
    return load_data(Path(artifacts_dir))


@st.cache_data(ttl=300)
def _horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    return horizon_curve(predictions)


@st.cache_data(ttl=300)
def _summary(predictions: pd.DataFrame) -> pd.DataFrame:
    return series_summary(predictions)


def _pct(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return f"{number:.1%}" if pd.notna(number) else "—"


@st.fragment
def series_panel(data: DashboardData) -> None:
    left, right = st.columns([1, 3])
    with left:
        model = st.selectbox("Model", data.models, key="series_model")
        series_id = st.selectbox("SKU–store", data.series, key="series_id")
    with right:
        frame = series_frame(data.predictions, series_id, model)
        if frame.empty:
            st.info("No predictions for this selection.")
            return
        chart = frame.set_index("date")[["y", "yhat"]].rename(
            columns={"y": "actual", "yhat": "forecast"}
        )
        st.line_chart(chart, height=340)
        summary = _summary(data.predictions)
        row = summary[(summary["model"] == model) & (summary["series_id"] == series_id)]
        if not row.empty:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("WAPE", _pct(row["wape"].iloc[0]))
            col_b.metric("Bias", _pct(row["bias_pct"].iloc[0]))
            col_c.metric("Actual units", f"{row['actual'].iloc[0]:,.0f}")


def main() -> None:
    artifacts = Path(os.environ.get("DEMAND_ARTIFACTS", "artifacts"))
    predictions_path = artifacts / "backtest" / "predictions.parquet"
    if not predictions_path.exists():
        st.error(
            "No backtest artifacts found. Run `demand demo` (or `demand backtest`) first, "
            f"then reload. Looked in `{predictions_path}`."
        )
        st.stop()
        return
    data = _load(str(artifacts), predictions_path.stat().st_mtime)

    st.title("Retail Demand Forecasting")
    st.caption(
        f"28-day SKU–store forecasts · {len(data.metrics.get('folds', []))}-fold walk-forward "
        f"validation · {data.predictions['model'].nunique()} models compared"
    )
    if data.synthetic:
        st.warning(
            "Demo dataset: these numbers verify the pipeline end to end and are not "
            "estimates of Walmart demand. Run the M5 pipeline for real results.",
            icon="⚠️",
        )

    kpis = kpi_table(data.metrics)
    naive = kpis.loc[kpis["model"] == "seasonal_naive", "wape"]
    deployable = kpis[kpis["model"] != "prophet"]
    best = (
        deployable.iloc[0] if not deployable.empty else (kpis.iloc[0] if not kpis.empty else None)
    )

    top = st.columns(4)
    top[0].metric("Models compared", len(kpis))
    top[1].metric("Series", f"{data.predictions['series_id'].nunique():,}")
    top[2].metric("Best deployable WAPE", _pct(best["wape"]) if best is not None else "—")
    if not naive.empty and best is not None:
        delta = float(naive.iloc[0]) - float(best["wape"])
        top[3].metric("vs naive", f"{delta:+.1%}", delta=f"{-delta:+.1%} error")
    else:
        top[3].metric("vs naive", "—")

    overview, drilldown, validation = st.tabs(["Overview", "SKU drill-down", "Validation"])

    with overview:
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Error by forecast horizon")
            st.line_chart(_horizon(data.predictions), height=320)
            st.caption("WAPE per horizon; day 1–7 are usually the easiest to hit.")
        with right:
            st.subheader("Model comparison")
            display = kpis.copy()
            for column in ("wape", "mape", "bias", "mean_rmsse"):
                if column in display:
                    display[column] = display[column].map(
                        lambda value: f"{value:.3f}" if pd.notna(value) else "—"
                    )
            st.dataframe(display, hide_index=True, width="stretch")
            st.caption("Lower WAPE and |bias| are better. RMSSE scales by naive error.")
        st.subheader("Highest-error series")
        model_for_table = st.selectbox("Model for error ranking", data.models, key="worst_model")
        st.dataframe(
            worst_series(data.predictions, model_for_table, n=12)[
                ["series_id", "wape", "bias_pct", "actual"]
            ].assign(
                wape=lambda d: d["wape"].map(_pct), bias_pct=lambda d: d["bias_pct"].map(_pct)
            ),
            hide_index=True,
            width="stretch",
        )

    with drilldown:
        series_panel(data)

    with validation:
        st.subheader("Walk-forward folds")
        folds = fold_table(data.metrics)
        if not folds.empty:
            st.dataframe(folds, hide_index=True, width="stretch")
        else:
            st.dataframe(data.folds, hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Fold stability")
            if not data.folds.empty:
                st.bar_chart(data.folds.set_index("fold")[["horizon"]], height=260)
        with right:
            st.subheader("Error distribution")
            model = st.selectbox("Model", data.models, key="hist_model")
            histogram = error_histogram(data.predictions, model)
            st.bar_chart(histogram.set_index("error")["count"], height=260)
            st.caption("Actual − forecast. A long right tail means under-forecasting.")

    with st.sidebar:
        st.header("Run")
        meta = data.run_meta
        if meta:
            st.write(f"**Model** {meta.get('model', '—')}")
            st.write(f"**Cutoff** {meta.get('cutoff', '—')}")
            st.write(f"**Run** `{meta.get('run_id', '—')}`")
            st.write(f"**Git** `{meta.get('git_sha', '—')}`")
        st.divider()
        st.caption(
            "Artifacts: `artifacts/backtest` and `artifacts/runs`. "
            "Regenerate with `demand backtest` then `demand train`."
        )


main()
