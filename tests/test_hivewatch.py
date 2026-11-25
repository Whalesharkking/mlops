"""HiveWatch client: retry on outages, immediate failure on client errors (MS2 review B1)."""

import pytest
import requests

from nectarflow.features import hivewatch


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(hivewatch.systime, "sleep", lambda s: None)


def test_scale_count_survives_a_short_outage():
    session = FakeSession(
        [requests.ConnectionError(), FakeResponse(503), FakeResponse(200, [{"nofScales": 372}])]
    )
    assert hivewatch.fetch_n_scales(session) == 372
    assert session.calls == 3


def test_client_error_fails_at_once():
    session = FakeSession([FakeResponse(404)])
    with pytest.raises(requests.HTTPError):
        hivewatch.fetch_n_scales(session)
    assert session.calls == 1


def test_gives_up_after_the_last_retry():
    session = FakeSession([FakeResponse(503)] * hivewatch.RETRIES)
    with pytest.raises(requests.HTTPError):
        hivewatch.fetch_n_scales(session)
    assert session.calls == hivewatch.RETRIES
