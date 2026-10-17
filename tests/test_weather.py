from datetime import date

import pytest

from nectarflow.features.weather import IncompleteRunError, check_complete, daily_weather
from tests.conftest import hourly_run


def test_complete_run_passes():
    check_complete(hourly_run(date(2025, 6, 1)), date(2025, 6, 1))


def test_partial_run_is_rejected():
    with pytest.raises(IncompleteRunError, match="72 of 240"):
        check_complete(hourly_run(date(2025, 6, 1), hours=72), date(2025, 6, 1))


def test_daily_weather_uses_local_days_and_averages_locations():
    t0 = date(2025, 6, 1)  # summer time: 00 UTC is 02:00 local
    daily = daily_weather(hourly_run(t0), t0).set_index("target_date")

    assert len(daily) == 10
    assert daily["tmax"].iloc[0] == pytest.approx(11.5)
    assert daily.loc[t0, "n_hours"] == 22  # 02:00..23:00 local
    assert daily["n_hours"].iloc[-1] == 24
    # 24 h at 100 W/m² = 8.64 MJ/m²; precipitation 1 mm/h
    assert daily["shortwave_sum"].iloc[5] == pytest.approx(8.64)
    assert daily["precip_sum"].iloc[5] == pytest.approx(24.0)
    # the first hour carries no accumulated value and is skipped, not counted as zero
    assert daily.loc[t0, "precip_sum"] == pytest.approx(21.0)


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError("should have been handled before raise_for_status")


class _FakeSession:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_run_missing_from_archive_counts_as_incomplete():
    from nectarflow.features.weather import fetch_run

    body = {"error": True, "reason": "The requested model run is not available. Model: ecmwf_ifs"}
    with pytest.raises(IncompleteRunError, match="not available"):
        fetch_run(date(2025, 8, 5), session=_FakeSession(_FakeResponse(400, body)))
