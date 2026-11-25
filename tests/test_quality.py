"""Label quality report from as-of snapshots (MS2 review A1/E3)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nectarflow.features import quality


def snapshot(poll: date, values: dict[date, float]) -> pd.DataFrame:
    days = [poll - timedelta(days=k) for k in range(1, 5)]
    return pd.DataFrame({"date": days, "label": [values.get(d, np.nan) for d in days], "poll_date": poll})


def test_availability_and_changes():
    d = date(2026, 11, 1)
    snaps = pd.concat(
        [
            # poll 3.11.: yesterday (2.11.) missing, 1.11. present
            snapshot(d + timedelta(days=2), {d: 0.1, d - timedelta(days=1): 0.2}),
            # poll 4.11.: everything present, but 1.11. changed by 0.05
            snapshot(
                d + timedelta(days=3), {d + timedelta(days=2): 0.0, d + timedelta(days=1): 0.3, d: 0.15}
            ),
        ]
    )
    avail = quality.availability(snaps)
    assert avail["t0_minus_1"].tolist() == [False, True]
    assert avail["t0_minus_2"].tolist() == [True, True]
    assert quality.weekly_missing_share(avail)["missing_t0_1"].sum() == pytest.approx(0.5)

    changes = quality.label_changes(snaps)
    assert changes["changed"] == 1
    assert changes["max_change_kg"] == pytest.approx(0.05)
