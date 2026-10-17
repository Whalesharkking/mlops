from datetime import date, timedelta

import numpy as np
import pandas as pd

from nectarflow.features.labels import daily_labels, midnight_weights


def test_label_is_midnight_to_midnight_and_ignores_the_noon_dip(series_factory):
    series = series_factory(date(2025, 5, 1), 5, lambda d: 0.4)
    labels = daily_labels(series, asof=date(2025, 5, 6))
    assert np.allclose(labels["label"], 0.4)
    assert list(labels["date"]) == [date(2025, 5, 1) + timedelta(days=i) for i in range(5)]


def test_half_hour_timestamps_of_old_data_still_count_as_midnight(series_factory):
    series = series_factory(date(2019, 6, 1), 3, lambda d: -0.2, minute=30)
    labels = daily_labels(series, asof=date(2019, 6, 4))
    assert np.allclose(labels["label"], -0.2)


def test_missing_midnight_gives_nan_not_an_interpolated_value(series_factory):
    series = series_factory(date(2025, 5, 1), 4, lambda d: 0.3)
    local = series["ts"].dt.tz_convert("Europe/Zurich")
    gap = (local.dt.date == date(2025, 5, 3)) & (local.dt.hour == 0)
    labels = daily_labels(series[~gap], asof=date(2025, 5, 5))
    by_day = labels.set_index("date")["label"]
    assert np.isnan(by_day[date(2025, 5, 2)])  # ends at the missing midnight
    assert np.isnan(by_day[date(2025, 5, 3)])  # starts at it
    assert np.isclose(by_day[date(2025, 5, 1)], 0.3)


def test_midnight_weights_survive_daylight_saving_change(series_factory):
    series = series_factory(date(2025, 3, 28), 4, lambda d: 0.1)  # DST starts 30 March
    mw = midnight_weights(series)
    assert date(2025, 3, 31) in mw.index
    assert np.allclose(np.diff(mw.to_numpy()), 0.1)


def test_label_is_final_only_two_days_later(series_factory):
    series = series_factory(date(2025, 5, 1), 10, lambda d: 0.1)
    labels = daily_labels(series, asof=date(2025, 5, 11)).set_index("date")
    assert labels.loc[date(2025, 5, 9), "is_final"]
    assert not labels.loc[date(2025, 5, 10), "is_final"]


def test_jump_flag_depends_on_season():
    from nectarflow.features.labels import is_jump

    dates = pd.Series([date(2025, 6, 1), date(2025, 6, 2), date(2025, 1, 10), date(2025, 1, 11)])
    label = pd.Series([1.8, 3.0, 0.3, 1.5])
    assert list(is_jump(label, dates)) == [False, True, False, True]


def test_flagged_days_are_removed_from_label_clean(series_factory):
    series = series_factory(date(2025, 1, 1), 5, lambda d: 4.0 if d == date(2025, 1, 3) else -0.1)
    labels = daily_labels(series, asof=date(2025, 1, 6)).set_index("date")
    assert labels.loc[date(2025, 1, 3), "jump_flag"]
    assert np.isnan(labels.loc[date(2025, 1, 3), "label_clean"])
    assert np.isclose(labels.loc[date(2025, 1, 2), "label_clean"], -0.1)
