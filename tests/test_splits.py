from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from demand.evaluation.splits import SplitError, walk_forward_folds


def test_folds_end_on_the_last_date_and_do_not_overlap() -> None:
    folds = walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=7, n_folds=3, step_days=7)
    assert [fold.index for fold in folds] == [1, 2, 3]
    assert folds[-1].test_end == pd.Timestamp("2024-12-31")
    for fold in folds:
        assert fold.horizon == 7
        assert fold.train_end == fold.test_start - pd.Timedelta(days=1)
    for earlier, later in pairwise(folds):
        assert later.test_start == earlier.test_end + pd.Timedelta(days=1)


def test_expanding_windows_grow() -> None:
    folds = walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=7, n_folds=4, step_days=7)
    assert [fold.train_end for fold in folds] == sorted(fold.train_end for fold in folds)


def test_overlapping_folds_are_rejected() -> None:
    with pytest.raises(SplitError, match="overlap"):
        walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=7, n_folds=3, step_days=3)


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(SplitError):
        walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=0, n_folds=2)
    with pytest.raises(SplitError):
        walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=7, n_folds=0)


def test_fold_serializes_for_reports() -> None:
    fold = walk_forward_folds(pd.Timestamp("2024-12-31"), horizon=7, n_folds=1)[0]
    payload = fold.to_dict()
    assert payload["fold"] == 1
    assert payload["horizon"] == 7
    assert payload["test_end"] == "2024-12-31"
