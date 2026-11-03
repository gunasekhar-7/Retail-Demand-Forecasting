"""API contracts.

Pydantic v2 models are the schema, the validation and the OpenAPI document.
`horizon_days` carries a hard upper bound (the trained horizon) that the router
checks against the loaded artifact.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(min_length=1, max_length=120, examples=["FOODS_3_090_CA_1"])
    horizon_days: int = Field(default=28, ge=1, le=56)


class ForecastPoint(BaseModel):
    date: dt.date
    yhat: float = Field(ge=0, description="Predicted units for the day")


class ForecastResponse(BaseModel):
    series_id: str
    model_version: str
    generated_at: dt.datetime
    points: list[ForecastPoint]


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    series_id: str = Field(min_length=1, max_length=120, examples=["FOODS_3_090_CA_1"])
    horizon_days: int = Field(default=28, ge=1, le=56)
    price_multiplier: float = Field(
        default=1.0, gt=0.1, le=5.0, description="Multiplier on unit price (e.g. 0.85 for -15%)"
    )
    event_override: str | None = Field(default=None, max_length=50)


class WhatIfResponse(BaseModel):
    series_id: str
    model_version: str
    price_multiplier: float
    baseline_units: float
    scenario_units: float
    delta_units: float
    delta_pct: float
    points: list[ForecastPoint]


class ReloadResponse(BaseModel):
    status: str
    previous_version: str
    current_version: str
    n_series: int


class DriftReportResponse(BaseModel):
    status: Literal["ok", "warning", "alert", "unavailable"]
    generated_at: str | None = None
    reference_rows: int = 0
    current_rows: int = 0
    features: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None
    series: int
    horizon: int | None
    detail: str | None = None


class ModelInfo(BaseModel):
    model_version: str
    model: str
    cutoff: dt.date
    horizon: int
    created_at: dt.datetime
    git_sha: str
    n_series: int
    dataset: dict[str, object]
    validation: dict[str, float] | None = None
