from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nectarflow.common.config import FEATURE_COLUMNS, FINAL_LAG_DAYS
from nectarflow.features.compute import build_climatology, build_feature_rows, momentum


def label_table(start: date, values: list[float]) -> pd.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame({"date": dates, "label": values, "label_clean": values})


def test_momentum_windows_end_at_t0_minus_two():
    # day index i has label i, so sums identify exactly which days were used
    labels = label_table(date(2025, 5, 1), [float(i) for i in range(30)])
    t0 = date(2025, 5, 21)  # index 20
    last = 20 - FINAL_LAG_DAYS  # index 18 is the newest day allowed
    mom = momentum(labels, t0)
    assert mom["mom_1d"] == last
    assert mom["mom_3d"] == last + (last - 1) + (last - 2)
    assert mom["mom_14d"] == sum(range(last - 13, last + 1))


def test_momentum_never_uses_non_final_days():
    values = [0.0] * 30
    values[19] = 99.0  # t0 - 1: not final yet
    values[20] = 99.0  # t0 itself
    mom = momentum(label_table(date(2025, 5, 1), values), date(2025, 5, 21))
    assert all(v == 0.0 for v in mom.values())


def test_momentum_window_with_a_gap_is_nan():
    values = [0.1] * 30
    values[15] = np.nan
    mom = momentum(label_table(date(2025, 5, 1), values), date(2025, 5, 21))
    assert mom["mom_1d"] == pytest.approx(0.1)
    assert np.isnan(mom["mom_7d"])


def test_climatology_uses_only_2019_to_2023_and_wraps_the_year():
    days = pd.date_range("2019-01-01", "2026-12-31", freq="D")
    value = np.where(days.year <= 2023, 1.0, 50.0)  # later years must not leak in
    labels = pd.DataFrame({"date": days.date, "label": value, "label_clean": value})
    clim = build_climatology(labels).set_index("doy")
    assert (clim["clim"] == 1.0).all()
    # ±7 days also reach across new year: 15 days x 5 years, give or take the leap day
    assert clim.loc[1, "n_values"] >= 70


def test_feature_rows_have_ten_horizons_and_the_right_target_dates():
    t0 = date(2025, 6, 10)
    targets = [t0 + timedelta(days=h - 1) for h in range(1, 11)]
    weather = pd.DataFrame(
        {
            "target_date": targets,
            "tmax": 20.0,
            "tmin": 10.0,
            "precip_sum": 0.0,
            "shortwave_sum": 25.0,
            "wind_max": 8.0,
            "n_hours": 24,
        }
    )
    labels = label_table(date(2025, 5, 1), [0.2] * 60)
    clim = pd.DataFrame({"doy": range(1, 367), "clim": 0.5, "n_values": 75})
    rows = build_feature_rows(t0, weather, labels, clim)

    assert len(rows) == 10
    assert list(rows["h"]) == list(range(1, 11))
    assert list(rows["target_date"]) == targets
    assert (rows["issue_date"] == t0).all()
    assert set(FEATURE_COLUMNS) <= set(rows.columns)
    assert "label" not in rows.columns  # labels are joined at training time
    assert rows["mom_7d"].iloc[0] == pytest.approx(1.4)
