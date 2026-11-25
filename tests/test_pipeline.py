"""End-to-end run of the pipeline with the two APIs replaced by fakes (no network)."""

from datetime import date, timedelta

import pandas as pd
import pytest

from nectarflow.features import pipeline
from nectarflow.features import store as fs
from tests.conftest import hourly_run


@pytest.fixture
def fake_apis(monkeypatch, series_factory):
    series = series_factory(date(2024, 1, 1), 200, lambda d: 0.3 if d.month in (5, 6) else -0.05)
    monkeypatch.setattr(pipeline, "fetch_full_history", lambda today, session=None: series)
    monkeypatch.setattr(pipeline, "fetch_n_scales", lambda session=None: 364)

    def fake_run(issue_date, session=None):
        if issue_date == date(2024, 6, 2):
            raise pipeline.IncompleteRunError("run 2024-06-02: only 72 of 240 hours available")
        return hourly_run(issue_date)

    monkeypatch.setattr(pipeline, "fetch_run", fake_run)


def test_backfill_and_daily_run_write_the_same_layout(tmp_path, fake_apis):
    today = date(2024, 7, 1)
    store = fs.FeatureStore(tmp_path)
    report = pipeline.run(date(2024, 5, 30), date(2024, 6, 3), store, today=today, request_interval_s=0)

    assert report.written == [date(2024, 5, 30), date(2024, 5, 31), date(2024, 6, 1), date(2024, 6, 3)]
    assert report.incomplete == [date(2024, 6, 2)]
    assert not (tmp_path / fs.features_path(date(2024, 6, 2))).exists()  # never a partial run

    rows = pd.read_parquet(tmp_path / fs.features_path(date(2024, 6, 1)))
    assert len(rows) == 10
    assert rows["mom_1d"].notna().all()
    for rel in (fs.LABELS_PATH, fs.CLIMATOLOGY_PATH, fs.N_SCALES_PATH, fs.asof_labels_path(today)):
        assert (tmp_path / rel).exists()

    # the as-of snapshot holds daily labels only, never the hourly series (HiveWatch terms)
    asof = pd.read_parquet(tmp_path / fs.asof_labels_path(today))
    assert "ts" not in asof.columns and "weight" not in asof.columns
    assert len(asof) == 30 and (asof["poll_date"] == today).all()

    # a second run skips what is already stored (idempotent daily job)
    again = pipeline.run(
        date(2024, 5, 30), date(2024, 5, 31), fs.FeatureStore(tmp_path), today=today, request_interval_s=0
    )
    assert again.written == [] and len(again.skipped) == 2


def test_future_end_date_is_rejected(tmp_path, fake_apis):
    today = date(2024, 7, 1)
    with pytest.raises(ValueError):
        pipeline.run(
            today, today + timedelta(days=1), fs.FeatureStore(tmp_path), today=today, request_interval_s=0
        )


def test_missing_scale_count_does_not_stop_the_run(tmp_path, fake_apis, monkeypatch):
    def broken(session=None):
        raise pipeline.requests.ConnectionError("HiveWatch down")

    monkeypatch.setattr(pipeline, "fetch_n_scales", broken)
    today = date(2024, 7, 1)
    report = pipeline.run(today, today, fs.FeatureStore(tmp_path), today=today, request_interval_s=0)
    assert report.written == [today]
    n_scales = pd.read_parquet(tmp_path / fs.N_SCALES_PATH)
    assert n_scales["n_scales"].isna().all()


def test_missing_final_label_is_flagged_but_features_are_written(
    tmp_path, fake_apis, monkeypatch, series_factory
):
    # the series ends before t0 - 2, like a scale network that stopped uploading
    series = series_factory(date(2024, 1, 1), 180, lambda d: 0.1)
    monkeypatch.setattr(pipeline, "fetch_full_history", lambda today, session=None: series)
    today = date(2024, 7, 1)
    report = pipeline.run(today, today, fs.FeatureStore(tmp_path), today=today, request_interval_s=0)
    assert not report.final_label_present
    assert report.written == [today]  # the run still writes, then the job fails
    rows = pd.read_parquet(tmp_path / fs.features_path(today))
    assert rows["mom_1d"].isna().all()


def test_pipeline_pushes_even_if_hivewatch_fails(tmp_path, monkeypatch):
    pushed = []
    store = fs.FeatureStore(tmp_path)
    monkeypatch.setattr(store, "push", lambda message: pushed.append(message))

    def down(today, session=None):
        raise pipeline.requests.ConnectionError("HiveWatch down")

    monkeypatch.setattr(pipeline, "fetch_full_history", down)
    with pytest.raises(pipeline.requests.ConnectionError):
        pipeline.run(date(2024, 7, 1), date(2024, 7, 1), store, today=date(2024, 7, 1))
    assert len(pushed) == 1  # finally block ran


def test_daily_job_fails_when_the_final_label_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pipeline.RunReport(final_label_present=False))
    assert pipeline.main(["--local"]) == 1
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pipeline.RunReport())
    assert pipeline.main(["--local"]) == 0
