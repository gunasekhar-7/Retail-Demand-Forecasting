"""Walk-forward folds with strict temporal boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SplitError(ValueError):
    """Raised when the requested folds cannot exist without overlap."""


@dataclass(frozen=True)
class Fold:
    index: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def horizon(self) -> int:
        return int((self.test_end - self.test_start).days) + 1

    def to_dict(self) -> dict[str, str | int]:
        return {
            "fold": self.index,
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "test_start": self.test_start.strftime("%Y-%m-%d"),
            "test_end": self.test_end.strftime("%Y-%m-%d"),
            "horizon": self.horizon,
        }


def walk_forward_folds(
    last_date: pd.Timestamp,
    horizon: int = 28,
    n_folds: int = 4,
    step_days: int = 28,
) -> list[Fold]:
    """Build expanding-window folds ending exactly on `last_date`.

    Fold k validates the `horizon` days immediately before fold k+1, so the test
    windows never overlap and every training window ends the day before its
    validation window starts.
    """
    if horizon < 1 or n_folds < 1 or step_days < 1:
        raise SplitError("horizon, n_folds and step_days must be positive")
    end = pd.Timestamp(last_date).normalize()
    folds: list[Fold] = []
    for offset in range(n_folds):
        test_end = end - pd.Timedelta(days=offset * step_days)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)
        folds.append(
            Fold(
                index=n_folds - offset,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    folds.reverse()
    for i in range(1, len(folds)):
        if folds[i].test_start <= folds[i - 1].test_end:
            raise SplitError("generated folds overlap; reduce step_days or horizon")
    return folds
