"""Training pipeline on synthetic data: reader, CV, evaluation, gate and both protocols."""

import json
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import wandb

from nectarflow.common.config import FEATURE_COLUMNS, GAP_DAYS, GATE_MIN_GAIN_KG, GATE_MIN_ROWS, HORIZONS
from nectarflow.features import store as fs
from nectarflow.training import cv, data, gate, train
from nectarflow.training.evaluate import gate_scores, mae_table, scoring_mask, success_criterion
from nectarflow.training.registry import LocalRegistry

FIRST, LAST = date(2024, 3, 14), date(2026, 9, 30)


def synthetic_store(root, first=FIRST, last=LAST, missing=(date(2025, 8, 5),)):
    """Feature files and a label table like the real store. The label depends on tmax."""
    rng = np.random.default_rng(0)
    days = pd.date_range(first, last + timedelta(days=12)).date
    season = np.array([np.sin(np.pi * (d.timetuple().tm_yday - 60) / 180) for d in days])
    tmax = 10 + 10 * season + rng.normal(0, 3, len(days))
    label = 0.05 * (tmax - 10) + rng.normal(0, 0.05, len(days))
    labels = pd.DataFrame({"date": days, "label": label, "tmax": tmax})
    labels["is_final"] = labels["date"] <= last + timedelta(days=10)
    labels["jump_flag"] = False
    labels["label_clean"] = labels["label"]
    by_day = labels.set_index("date")

    for issue in pd.date_range(first, last).date:
        if issue in missing:
            continue
        targets = [issue + timedelta(days=h - 1) for h in HORIZONS]
        rows = pd.DataFrame({"issue_date": issue, "target_date": targets, "h": list(HORIZONS)})
        rows["tmax"] = by_day.loc[targets, "tmax"].to_numpy() + rng.normal(0, 0.5, 10)
        for col in ("tmin", "precip_sum", "shortwave_sum", "wind_max"):
            rows[col] = rng.normal(0, 1, 10)
        rows["n_hours"] = [23] + [24] * 9
        prev = issue - timedelta(days=2)
        for k in (1, 3, 7, 14):
            rows[f"mom_{k}d"] = by_day["label"].get(prev, np.nan) * k
        rows["clim_target"] = 0.0
        doy = pd.to_datetime(rows["target_date"]).dt.dayofyear
        rows["doy_sin"], rows["doy_cos"] = np.sin(2 * np.pi * doy / 366), np.cos(2 * np.pi * doy / 366)
        path = root / fs.features_path(issue)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(path, index=False)
    (root / "labels").mkdir(exist_ok=True)
    labels.drop(columns="tmax").to_parquet(root / fs.LABELS_PATH, index=False)
    return root


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return synthetic_store(tmp_path_factory.mktemp("store"))


@pytest.fixture(scope="module")
def frame(store_root):
    return data.load_frame(store_root)


@pytest.fixture
def run():
    r = wandb.init(mode="disabled")
    yield r
    r.finish()


# --- reader (E1) ---------------------------------------------------------------------


def test_reader_joins_on_target_date_and_keeps_only_final_labels(store_root, frame):
    labels = pd.read_parquet(store_root / fs.LABELS_PATH).set_index("date")
    assert set(frame.columns) >= set(FEATURE_COLUMNS) | {"issue_date", "target_date", "label", "label_clean"}
    row = frame[(frame["issue_date"] == date(2025, 6, 1)) & (frame["h"] == 5)].iloc[0]
    assert row["target_date"] == date(2025, 6, 5)
    assert row["label"] == labels.loc[date(2025, 6, 5), "label"]  # label of the target, not the issue day
    assert labels.loc[frame["target_date"], "is_final"].all()
    assert date(2025, 8, 5) not in set(frame["issue_date"])  # a missing run stays missing


def test_reader_downloads_exactly_the_resolved_commit(monkeypatch, tmp_path):
    calls = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def dataset_info(self, repo_id, revision):
            calls["asked"] = revision
            return SimpleNamespace(sha="abc123")

    def fake_snapshot(repo_id, **kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(fs, "HfApi", FakeApi)
    monkeypatch.setattr(fs, "snapshot_download", fake_snapshot)
    root, revision = data.download("user/repo", "main", tmp_path)
    assert revision == "abc123" and root == tmp_path
    assert calls["asked"] == "main" and calls["revision"] == "abc123"  # branch resolved, then pinned
    assert not any("raw" in p for p in calls["allow_patterns"])  # no raw weather runs


def test_scale_weights_ignore_days_without_a_count(tmp_path):
    (tmp_path / "quality").mkdir()
    counts = pd.DataFrame(
        {
            "poll_date": [date(2026, 10, 18), date(2026, 10, 19)],
            "n_scales": pd.array([372, None], dtype="Int64"),
        }
    )
    counts.to_parquet(tmp_path / fs.N_SCALES_PATH, index=False)
    w = data.scale_weights(pd.Series([date(2026, 10, 18), date(2026, 10, 19)]), tmp_path)
    assert np.isfinite(w).all()
    assert w[0] / w[1] == pytest.approx(372 / 364)  # recorded count, else the estimate


def test_scale_weights_follow_the_linear_estimate_and_average_one():
    w = data.scale_weights(pd.Series([date(2019, 1, 1), date(2026, 10, 17), date(2030, 1, 1)]))
    assert w.mean() == pytest.approx(1.0)
    assert w[0] / w[1] == pytest.approx(50 / 364)
    assert w[1] == w[2]  # no extrapolation beyond the estimate


# --- CV (E2) -------------------------------------------------------------------------


def test_folds_never_share_a_target_and_keep_issue_dates_whole(frame):
    folds = list(cv.expanding_folds(frame))
    assert len(folds) == 6
    for train_idx, val_idx in folds:
        tr, va = frame.loc[train_idx], frame.loc[val_idx]
        assert tr["target_date"].max() < va["target_date"].min()
        assert (va["issue_date"].min() - tr["issue_date"].max()).days == GAP_DAYS
        assert not set(tr["issue_date"]) & set(va["issue_date"])
    sizes = [len(t) for t, _ in folds]
    assert sizes == sorted(sizes)  # expanding window


def test_holdout_is_the_last_eight_weeks(frame):
    start = cv.holdout_start(frame)
    assert (frame["issue_date"].max() - start).days == 55
    assert frame[cv.train_mask(frame, start)]["target_date"].max() < start


# --- evaluation (B3) -----------------------------------------------------------------


def test_mae_table_scores_all_predictors_on_the_same_rows():
    df = pd.DataFrame(
        {
            "h": [1, 1, 2, 2],
            "target_date": [date(2026, 5, 1)] * 4,
            "label": [1.0, np.nan, 0.0, 0.5],
            "mom_1d": [0.0, 0.0, np.nan, 0.5],
            "clim_target": [0.5, 0.5, 0.5, 0.5],
        }
    )
    table = mae_table(df, {"persistence": df["mom_1d"], "climatology": df["clim_target"]})
    assert scoring_mask(df).tolist() == [True, False, False, True]
    assert table.loc["all", "n"] == 2
    assert table.loc[1, "persistence"] == 1.0 and table.loc[2, "climatology"] == 0.0


def test_success_criterion_needs_ten_percent_for_early_horizons():
    table = pd.DataFrame(
        {"persistence": [1.0, 1.0], "climatology": [0.5, 0.5], "model": [0.46, 0.5]}, index=[1, 6]
    )
    table.loc["all"] = [1.0, 0.5, 0.48]
    ok = success_criterion(table, "model")
    assert ok.tolist() == [False, True]  # h = 1 needs <= 0.45, h = 6 needs <= 0.5


def test_gate_scores_count_only_season_targets():
    df = pd.DataFrame(
        {
            "h": [1, 1],
            "target_date": [date(2026, 1, 10), date(2026, 5, 10)],
            "label": [0.0, 1.0],
            "mom_1d": [0.0, 0.0],
            "clim_target": [0.0, 0.0],
        }
    )
    scores, n = gate_scores(df, {"challenger": np.array([5.0, 1.0])})
    assert n == 1 and scores["challenger"] == 0.0


# --- gate (E2, MS1 A3) ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("challenger", "champion", "rows", "promote"),
    [
        (0.20, None, GATE_MIN_ROWS, True),  # first champion: baselines suffice
        (0.20, 0.22, GATE_MIN_ROWS, True),
        (0.20, 0.20 + GATE_MIN_GAIN_KG / 2, GATE_MIN_ROWS, False),  # gain too small
        (0.20, 0.30, GATE_MIN_ROWS - 1, False),  # winter: too few season rows
        (0.26, 0.40, GATE_MIN_ROWS, False),  # worse than climatology
    ],
)
def test_gate_decision(challenger, champion, rows, promote):
    scores = {"challenger": challenger, "persistence": 0.30, "climatology": 0.25}
    assert gate.decide(scores, rows, champion).promote is promote


def test_gate_never_promotes_when_the_holdout_ends_off_season():
    scores = {"challenger": 0.10, "persistence": 0.30, "climatology": 0.25}
    assert gate.decide(scores, GATE_MIN_ROWS, 0.30, holdout_end=date(2026, 9, 30)).promote
    october = gate.decide(scores, 10 * GATE_MIN_ROWS, 0.30, holdout_end=date(2026, 10, 1))
    assert not october.promote and "off season" in october.reason


def test_ablation_reports_every_feature_group(frame, store_root):
    config = train.ModelConfig({"num_leaves": 7}, 30, False)
    table = train.run_ablation(frame, store_root, config)
    assert list(table.index) == ["all features", "without weather", "without momentum"]
    # the synthetic label depends on tmax, so dropping the weather must hurt
    assert table.loc["without weather", "test_mae"] > table.loc["all features", "test_mae"]


def test_gate_needs_both_baselines():
    scores = {"challenger": 0.20, "persistence": 0.19, "climatology": 0.25}
    decision = gate.decide(scores, GATE_MIN_ROWS, champion=0.30)
    assert not decision.promote and "persistence" in decision.reason


def test_gate_refuses_a_champion_that_saw_the_holdout():
    gate.check_unseen(date(2026, 9, 1), date(2026, 9, 2))
    with pytest.raises(gate.GateError):
        gate.check_unseen(date(2026, 9, 2), date(2026, 9, 2))


# --- both protocols end to end -------------------------------------------------------


def test_eval_then_retrain(frame, store_root, tmp_path, run):
    registry = LocalRegistry(tmp_path / "registry")
    params_path = tmp_path / "model.json"

    decision = train.run_eval(
        frame,
        store_root,
        "rev1",
        registry,
        run,
        trials=2,
        timeout_s=60,
        params_path=params_path,
        workdir=tmp_path / "eval",
    )
    assert decision.promote, decision.reason  # learnable data: beats both baselines
    champion = registry.champion()
    assert champion.version == "v0"
    assert champion.meta["feature_columns"] == FEATURE_COLUMNS
    assert champion.meta["train_end"] == "2025-12-31" and champion.meta["data_revision"] == "rev1"
    config = train.ModelConfig.load(params_path)
    assert json.loads(params_path.read_text())["data_revision"] == "rev1"

    # Retrain at the end of September: the holdout is August and September (season rows).
    # Each protocol is its own W&B run.
    retrain_run = wandb.init(mode="disabled")
    decision = train.run_retrain(
        frame, store_root, "rev2", registry, retrain_run, config=config, workdir=tmp_path / "retrain"
    )
    retrain_run.finish()
    assert (tmp_path / "registry" / "v1" / "model.json").exists()  # every challenger is registered
    assert "season rows" not in decision.reason
    assert registry.champion().version == ("v1" if decision.promote else "v0")  # alias follows the gate

    # A second evaluation run never replaces an existing champion.
    before = registry.champion().version
    eval_run = wandb.init(mode="disabled")
    again = train.run_eval(
        frame,
        store_root,
        "rev3",
        registry,
        eval_run,
        trials=1,
        timeout_s=60,
        params_path=tmp_path / "model2.json",
        workdir=tmp_path / "eval2",
    )
    eval_run.finish()
    assert not again.promote and registry.champion().version == before

    # Reproducible: same data, same config, same predictions.
    a = train.fit(frame, config.params, config.num_boost_round, store_root, config.use_scale_weight)
    b = train.fit(frame, config.params, config.num_boost_round, store_root, config.use_scale_weight)
    assert np.array_equal(train.predict(a, frame), train.predict(b, frame))


def test_retrain_refuses_to_run_without_a_champion(frame, store_root, tmp_path, run):
    config = train.ModelConfig({"num_leaves": 7}, 20, False)
    with pytest.raises(gate.GateError, match="no champion"):
        train.run_retrain(
            frame, store_root, "rev", LocalRegistry(tmp_path / "r"), run, config=config, workdir=tmp_path
        )


def test_retrain_refuses_a_champion_trained_on_the_holdout(frame, store_root, tmp_path, run):
    registry = LocalRegistry(tmp_path / "registry")
    config = train.ModelConfig({"num_leaves": 7}, 20, False)
    booster = train.fit(frame, config.params, 20, None, False)  # trained on everything
    registry.register(train.save_model(booster, frame, tmp_path / "m"), promote=True)
    with pytest.raises(gate.GateError):
        train.run_retrain(frame, store_root, "rev", registry, run, config=config, workdir=tmp_path / "w")


# --- W&B registry (mocked, no network) -----------------------------------------------


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.metadata, self.version, self.files = metadata, "v7", []

    def add_dir(self, path):
        self.files.append(path)

    def wait(self):
        return self


class FakeRun:
    def __init__(self):
        self.links = []

    def log_artifact(self, artifact):
        self.logged = artifact

    def link_artifact(self, artifact, target, aliases):
        self.links.append((target, aliases))
        return artifact


def wandb_registry(monkeypatch, tmp_path, error=None):
    from nectarflow.training import registry as reg

    class FakeApi:
        def artifact(self, name):
            raise error

    monkeypatch.setattr(wandb, "Api", FakeApi)
    monkeypatch.setattr(wandb, "Artifact", FakeArtifact)
    return reg.WandbRegistry(FakeRun(), tmp_path)


def test_wandb_registry_without_champion(monkeypatch, tmp_path):
    error = wandb.errors.CommError("artifact membership 'champion' not found in 'org/wandb-registry-model'")
    assert wandb_registry(monkeypatch, tmp_path, error).champion() is None


def test_wandb_registry_does_not_hide_other_errors(monkeypatch, tmp_path):
    registry = wandb_registry(monkeypatch, tmp_path, wandb.errors.CommError("HTTP 503 service unavailable"))
    with pytest.raises(wandb.errors.CommError):
        registry.champion()


def test_wandb_registry_moves_the_alias_only_on_promotion(monkeypatch, tmp_path):
    registry = wandb_registry(monkeypatch, tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.json").write_text("{}")
    registry.register(model, promote=False)
    registry.register(model, promote=True)
    target = "wandb-registry-model/nectarflow-forecast"
    assert registry.run.links == [(target, []), (target, ["champion"])]
