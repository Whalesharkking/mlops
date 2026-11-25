"""Time-ordered splits over issue dates (MS1 review B2, MS2 review E2).

Splits are defined on dates, not on row positions, because some issue dates are missing
(eight weather runs) and every issue date has ten rows. All ten horizons of an issue date
always land in the same fold.
"""

from collections.abc import Iterator
from datetime import date, timedelta

import pandas as pd

from nectarflow.common.config import CV_BLOCK_STARTS, CV_LAST_ISSUE, GAP_DAYS, GATE_HOLDOUT_DAYS


def train_mask(frame: pd.DataFrame, first_held_out: date) -> pd.Series:
    """Rows usable for training when validation starts at ``first_held_out``.

    Issue dates end GAP_DAYS before, so the latest training target (t0 + 9) is the day
    before ``first_held_out`` and no target is shared with the validation rows.
    """
    return frame["issue_date"] <= first_held_out - timedelta(days=GAP_DAYS)


def expanding_folds(
    frame: pd.DataFrame,
    block_starts: tuple[date, ...] = CV_BLOCK_STARTS,
    last_issue: date = CV_LAST_ISSUE,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """(train index, validation index) per block. Each block runs until the next one starts."""
    ends = [b - timedelta(days=1) for b in block_starts[1:]] + [last_issue]
    for start, end in zip(block_starts, ends, strict=True):
        val = frame.index[(frame["issue_date"] >= start) & (frame["issue_date"] <= end)]
        train = frame.index[train_mask(frame, start)]
        if len(val) and len(train):
            yield train, val


def holdout_start(frame: pd.DataFrame, days: int = GATE_HOLDOUT_DAYS) -> date:
    """First issue date of the gate holdout: the last ``days`` issue dates with final labels."""
    return frame["issue_date"].max() - timedelta(days=days - 1)
