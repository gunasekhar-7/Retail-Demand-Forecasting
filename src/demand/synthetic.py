"""Generate an M5-shaped synthetic dataset.

The layout mirrors the real competition files (`sales_train_evaluation.csv` wide
with `d_1..d_N`, `calendar.csv`, `sell_prices.csv`) so every downstream stage is
exercised against the same schema. Numbers produced from this data are demo
numbers and are labelled as such in reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from demand.config import SyntheticConfig

CATEGORY = "FOODS"
DEPT = "FOODS_3"
STATE_BY_STORE = {
    "CA_1": "CA",
    "CA_2": "CA",
    "TX_1": "TX",
    "TX_2": "TX",
    "WI_1": "WI",
    "WI_2": "WI",
    "WI_3": "WI",
}
SNAP_WINDOW = {"CA": 10, "TX": 15, "WI": 10}
WEEKLY_FACTOR = np.array([0.85, 0.80, 0.86, 0.92, 1.15, 1.40, 1.24])  # Mon..Sun


@dataclass(frozen=True)
class SyntheticDataset:
    raw_dir: Path
    sales_path: Path
    calendar_path: Path
    prices_path: Path
    n_series: int
    n_days: int


def _item_ids(n_items: int) -> list[str]:
    return [f"{DEPT}_{i:03d}" for i in range(1, n_items + 1)]


def build_calendar(cfg: SyntheticConfig) -> pd.DataFrame:
    total_days = cfg.days + cfg.calendar_extra_days
    dates = pd.date_range(cfg.start_date, periods=total_days, freq="D")
    wday = dates.day_of_week.to_numpy() + 1
    calendar = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + np.arange(total_days) // 7,
            "weekday": dates.day_name(),
            "wday": wday,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in range(1, total_days + 1)],
            "event_name_1": pd.Series([None] * total_days, dtype="object"),
            "event_type_1": pd.Series([None] * total_days, dtype="object"),
            "event_name_2": pd.Series([None] * total_days, dtype="object"),
            "event_type_2": pd.Series([None] * total_days, dtype="object"),
            "snap_CA": [int(day <= SNAP_WINDOW["CA"]) for day in dates.day],
            "snap_TX": [int(day <= SNAP_WINDOW["TX"]) for day in dates.day],
            "snap_WI": [int(day <= SNAP_WINDOW["WI"]) for day in dates.day],
        }
    )
    idx = np.arange(total_days)
    primary = idx % 91 == 45
    secondary = idx % 137 == 100
    calendar.loc[primary, "event_name_1"] = "SyntheticEvent"
    calendar.loc[primary, "event_type_1"] = "Event1"
    calendar.loc[secondary, "event_name_2"] = "CulturalEvent"
    calendar.loc[secondary, "event_type_2"] = "Cultural"
    return calendar


def build_sales_and_prices(
    cfg: SyntheticConfig, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calendar_full = build_calendar(cfg)
    calendar = calendar_full.iloc[: cfg.days].reset_index(drop=True)
    dates = pd.to_datetime(calendar["date"])
    n_days = cfg.days
    doy = dates.dt.day_of_year.to_numpy()
    wday = dates.dt.day_of_week.to_numpy() + 1
    yearly = 1.0 + 0.25 * np.sin(2 * np.pi * (doy - 80) / 365.25)
    trend = 1.0 + cfg.trend_per_day * np.arange(n_days)
    promo = np.where(calendar["event_name_1"].notna(), 1.0 + cfg.promo_lift, 1.0)

    items = _item_ids(cfg.items_per_store)
    price_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []

    for store in cfg.stores:
        state = STATE_BY_STORE.get(store, "CA")
        snap = calendar[f"snap_{state}"].to_numpy()
        for item in items:
            base = rng.uniform(1.5, 9.0)
            item_price = round(float(rng.uniform(2.0, 12.0)), 2)
            weekly = WEEKLY_FACTOR[wday - 1]
            seasonal = base * weekly * yearly * trend * promo
            seasonal = seasonal * np.where(snap == 1, 1.06, 1.0)
            noise = rng.lognormal(mean=0.0, sigma=cfg.noise, size=n_days)
            sales = np.clip(np.rint(seasonal * noise), 0, None).astype(int)

            row: dict[str, object] = {
                "id": f"{item}_{store}_evaluation",
                "item_id": item,
                "dept_id": DEPT,
                "cat_id": CATEGORY,
                "store_id": store,
                "state_id": state,
            }
            row.update({f"d_{i + 1}": int(value) for i, value in enumerate(sales)})
            wide_rows.append(row)

            weeks = calendar["wm_yr_wk"].to_numpy()
            for week in pd.unique(weeks):
                drift = 1.0 + rng.normal(0.0, 0.02)
                price_rows.append(
                    {
                        "store_id": store,
                        "item_id": item,
                        "wm_yr_wk": int(week),
                        "sell_price": round(item_price * drift, 2),
                    }
                )

    sales = pd.DataFrame(wide_rows)
    prices = pd.DataFrame(price_rows)
    return sales, calendar_full, prices


def generate_synthetic_dataset(
    cfg: SyntheticConfig, out_dir: Path, seed: int = 42
) -> SyntheticDataset:
    out_dir.mkdir(parents=True, exist_ok=True)
    sales, calendar, prices = build_sales_and_prices(cfg, seed=seed)
    sales_path = out_dir / "sales_train_evaluation.csv"
    calendar_path = out_dir / "calendar.csv"
    prices_path = out_dir / "sell_prices.csv"
    sales.to_csv(sales_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    prices.to_csv(prices_path, index=False)
    return SyntheticDataset(
        raw_dir=out_dir,
        sales_path=sales_path,
        calendar_path=calendar_path,
        prices_path=prices_path,
        n_series=len(sales),
        n_days=cfg.days,
    )
