"""Training pipeline: read features at a pinned revision, train LightGBM, evaluate, register.

    python -m nectarflow.training.train --mode eval      # search, final test on 2026, first champion
    python -m nectarflow.training.train --mode retrain   # weekly: frozen parameters, gate decides
    python -m nectarflow.training.train --mode ablation  # test MAE without weather or momentum
    ... --offline                                        # W&B offline, local registry in DATA_DIR

Evaluation protocol. Hyperparameters are searched with the expanding-window CV on issue
dates up to 22.12.2025. The final model is trained on all of them with a fixed number of
rounds and tested once on 2026. The parameters are written to ``configs/model.json``
(uploaded as workflow artifact in Actions) and committed by hand, so the weekly retrain
uses versioned, frozen parameters. If there is no champion yet, the gate decides on the
2026 season rows whether this model becomes the first one.

Retrain protocol. The last eight weeks of issue dates with final labels are the holdout.
The challenger is trained on the issue dates up to ten days before it, then the gate
compares it with the champion and both baselines on the season rows of the holdout.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from nectarflow.common.config import (
    EVAL_TARGET,
    FEATURE_COLUMNS,
    MOMENTUM_FEATURES,
    REGISTRY_COLLECTION,
    TEST_FIRST_ISSUE,
    TRAIN_LAST_ISSUE,
    TRAIN_TARGET,
    WANDB_PROJECT,
    WEATHER_FEATURES,
    Settings,
)
from nectarflow.training import cv, gate
from nectarflow.training.data import download, load_frame, scale_weights, split_by_issue
from nectarflow.training.evaluate import baseline_predictions, gate_scores, mae_table, success_criterion
from nectarflow.training.registry import (
    META_FILE,
    MODEL_FILE,
    LocalRegistry,
    RegisteredModel,
    Registry,
    WandbRegistry,
)

log = logging.getLogger("nectarflow.training")

SEED = 42
BASE_PARAMS = {
    "objective": "l1",  # the project metric is MAE
    "metric": "l1",
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 2,
    "verbosity": -1,
}
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 100
DEFAULT_PARAMS_PATH = Path("configs/model.json")


def to_json(obj: Any) -> str:
    """Strict JSON (no NaN), so MS4 and other parsers can read it. NaN becomes null."""

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [clean(v) for v in x]
        if isinstance(x, float) and np.isnan(x):
            return None
        return x

    return json.dumps(clean(obj), indent=2, default=str, allow_nan=False) + "\n"


@dataclass(frozen=True)
class ModelConfig:
    params: dict[str, Any]
    num_boost_round: int
    use_scale_weight: bool

    def to_json(self, **extra) -> str:
        return to_json({**asdict(self), **extra})

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        raw = json.loads(path.read_text())
        return cls(raw["params"], int(raw["num_boost_round"]), bool(raw["use_scale_weight"]))


# --- model ---------------------------------------------------------------------------


def _train_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[TRAIN_TARGET].notna()]


def fit(
    df: pd.DataFrame,
    params: dict,
    rounds: int,
    root: Path | None,
    use_weight: bool,
    valid: pd.DataFrame | None = None,
    columns: list[str] = FEATURE_COLUMNS,
) -> lgb.Booster:
    """Train on ``TRAIN_TARGET``. With ``valid``, stop early on the raw label of that block."""
    rows = _train_rows(df)
    weight = scale_weights(rows["target_date"], root) if use_weight else None
    dtrain = lgb.Dataset(rows[columns], rows[TRAIN_TARGET], weight=weight, free_raw_data=False)
    kwargs: dict[str, Any] = {}
    if valid is not None:
        v = valid[valid[EVAL_TARGET].notna()]
        kwargs["valid_sets"] = [lgb.Dataset(v[columns], v[EVAL_TARGET], reference=dtrain)]
        kwargs["callbacks"] = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
    return lgb.train({**BASE_PARAMS, **params}, dtrain, num_boost_round=rounds, **kwargs)


def predict(booster: lgb.Booster, df: pd.DataFrame, columns: list[str] = FEATURE_COLUMNS) -> np.ndarray:
    return booster.predict(df[columns], num_threads=BASE_PARAMS["num_threads"])


def cv_score(
    frame: pd.DataFrame, params: dict, root: Path | None, use_weight: bool
) -> tuple[float, int, list[float]]:
    """Mean validation MAE over the expanding-window folds, the median best iteration and the fold MAEs.

    Early stopping uses the validation block itself, so the CV MAE is slightly optimistic.
    It only ranks parameter sets. The final test on 2026 never stops early.
    """
    maes, iters = [], []
    for train_idx, val_idx in cv.expanding_folds(frame):
        val = frame.loc[val_idx]
        booster = fit(frame.loc[train_idx], params, MAX_ROUNDS, root, use_weight, valid=val)
        scored = val[EVAL_TARGET].notna()
        pred = predict(booster, val[scored])
        maes.append(float(np.mean(np.abs(pred - val.loc[scored, EVAL_TARGET]))))
        iters.append(booster.best_iteration or MAX_ROUNDS)
    if not maes:
        raise ValueError("no CV fold has training and validation rows")
    return float(np.mean(maes)), int(np.median(iters)), maes


def search(
    frame: pd.DataFrame, root: Path | None, trials: int, timeout_s: int
) -> tuple[dict, int, float, pd.DataFrame]:
    """Small Optuna search over the CV. Seeded sampler and fixed trial count, timeout as budget.

    Returns the best parameters, their number of rounds, their CV MAE and one row per trial.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        mae, rounds, folds = cv_score(frame, params, root, use_weight=False)
        trial.set_user_attr("rounds", rounds)
        for i, fold_mae in enumerate(folds):
            trial.set_user_attr(f"fold_{i}_mae", round(fold_mae, 4))
        return mae

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, timeout=timeout_s)
    best = study.best_trial
    log.info("search: %d trials, best CV MAE %.4f", len(study.trials), best.value)
    trials_table = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    return {**best.params, "bagging_freq": 1}, int(best.user_attrs["rounds"]), float(best.value), trials_table


# --- artifacts -----------------------------------------------------------------------


def git_commit() -> str:
    if sha := os.environ.get("GITHUB_SHA"):
        return sha
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    # a model trained from uncommitted code is marked, its code is not in that commit
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def save_model(booster: lgb.Booster, train: pd.DataFrame, out: Path, **meta) -> Path:
    """Booster plus the metadata inference needs to check its input (no training/serving skew)."""
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(out / MODEL_FILE)
    rows = _train_rows(train)
    info = {
        "feature_columns": FEATURE_COLUMNS,
        "train_target": TRAIN_TARGET,
        "eval_target": EVAL_TARGET,
        "train_first_issue": rows["issue_date"].min(),
        "train_last_issue": rows["issue_date"].max(),
        "train_end": rows["target_date"].max(),  # gate check: champion never saw the holdout
        "n_train_rows": len(rows),
        "git_commit": git_commit(),
        **meta,
    }
    (out / META_FILE).write_text(to_json(info))
    return out


def champion_predictions(champion: RegisteredModel, df: pd.DataFrame) -> np.ndarray:
    booster = lgb.Booster(model_file=str(champion.path / MODEL_FILE))
    return predict(booster, df, champion.meta["feature_columns"])


def _log_table(run, name: str, table: pd.DataFrame, index: bool = True) -> None:
    import wandb

    if not index:
        run.log({name: wandb.Table(dataframe=table.round(4))})
        return
    df = table.reset_index().round(4).astype({"h": str})  # rows 1..10 and "all"
    if "meets_criterion" in df:
        df["meets_criterion"] = df["meets_criterion"].map({True: "yes", False: "no"}).fillna("")
    run.log({name: wandb.Table(dataframe=df)})


# --- protocols -----------------------------------------------------------------------


def run_eval(
    frame: pd.DataFrame,
    root: Path | None,
    revision: str,
    registry: Registry,
    run,
    *,
    trials: int,
    timeout_s: int,
    params_path: Path,
    workdir: Path,
) -> gate.GateDecision:
    dev = split_by_issue(frame, last=TRAIN_LAST_ISSUE)
    test = split_by_issue(frame, first=TEST_FIRST_ISSUE)

    params, rounds, cv_mae, trials_table = search(dev, root, trials, timeout_s)
    _log_table(run, "search_trials", trials_table, index=False)
    # E4: the same parameters with the scale-count weight. They were searched without it,
    # so the comparison slightly favours "no weight".
    weighted_mae, weighted_rounds, _ = cv_score(dev, params, root, use_weight=True)
    use_weight = weighted_mae < cv_mae
    if use_weight:
        rounds = weighted_rounds
    log.info("E4 scale weight: CV MAE %.4f with, %.4f without -> %s", weighted_mae, cv_mae, use_weight)
    config = ModelConfig(params, rounds, use_weight)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(
        config.to_json(
            cv_mae=round(min(cv_mae, weighted_mae), 4), data_revision=revision, trials=len(trials_table)
        )
    )
    run.config.update({"mode": "eval", "data_revision": revision, **asdict(config)})
    run.summary.update(
        {"cv_mae": cv_mae, "cv_mae_weighted": weighted_mae, "search_trials": len(trials_table)}
    )

    booster = fit(dev, params, rounds, root, use_weight)
    preds = {**baseline_predictions(test), "model": predict(booster, test)}
    table = mae_table(test, preds)
    table["meets_criterion"] = success_criterion(table, "model").reindex(table.index)
    print(f"Test 2026 (issue dates from {TEST_FIRST_ISSUE}), MAE in kg/day:")
    print(table.round(3).to_string())
    _log_table(run, "test_2026_mae", table)

    scores, n_rows = gate_scores(test, {**preds, "challenger": preds["model"]})
    champion = registry.champion()
    if champion is None:
        decision = gate.decide(scores, n_rows, champion=None)  # first champion (bootstrap)
    else:
        decision = gate.GateDecision(
            False, f"champion {champion.version} exists, eval mode never replaces it"
        )
    model_dir = save_model(
        booster,
        dev,
        workdir / "model",
        mode="eval",
        data_revision=revision,
        params=params,
        num_boost_round=rounds,
        use_scale_weight=use_weight,
        test_mae=table["model"].to_dict(),
        gate_scores=scores,
        gate_rows=n_rows,
    )
    version = registry.register(model_dir, promote=decision.promote)
    _finish(run, decision, version, scores, n_rows)
    return decision


def run_retrain(
    frame: pd.DataFrame,
    root: Path | None,
    revision: str,
    registry: Registry,
    run,
    *,
    config: ModelConfig,
    workdir: Path,
) -> gate.GateDecision:
    first_holdout = cv.holdout_start(frame)
    train = frame[cv.train_mask(frame, first_holdout)]
    holdout = frame[frame["issue_date"] >= first_holdout]
    run.config.update({"mode": "retrain", "data_revision": revision, "holdout_start": str(first_holdout)})

    # The first champion comes only from the evaluation protocol. A missing champion in the
    # weekly run is an error to look at, not a reason to promote on the baselines alone.
    champion = registry.champion()
    if champion is None:
        raise gate.GateError("no champion in the registry, run --mode eval first")
    gate.check_unseen(date.fromisoformat(champion.meta["train_end"]), first_holdout)

    booster = fit(train, config.params, config.num_boost_round, root, config.use_scale_weight)
    preds = {
        **baseline_predictions(holdout),
        "challenger": predict(booster, holdout),
        "champion": champion_predictions(champion, holdout),
    }
    table = mae_table(holdout, preds)
    print(f"Holdout (issue dates from {first_holdout}), MAE in kg/day:")
    print(table.round(3).to_string())
    _log_table(run, "holdout_mae", table)

    scores, n_rows = gate_scores(holdout, preds)
    decision = gate.decide(
        scores, n_rows, champion=scores["champion"], holdout_end=holdout["issue_date"].max()
    )
    model_dir = save_model(
        booster,
        train,
        workdir / "model",
        mode="retrain",
        data_revision=revision,
        params=config.params,
        num_boost_round=config.num_boost_round,
        use_scale_weight=config.use_scale_weight,
        holdout_start=first_holdout,
        gate_scores=scores,
        gate_rows=n_rows,
        champion_version=champion.version,
    )
    version = registry.register(model_dir, promote=decision.promote)
    _finish(run, decision, version, scores, n_rows)
    return decision


def run_ablation(frame: pd.DataFrame, root: Path | None, config: ModelConfig) -> pd.DataFrame:
    """Test MAE on 2026 with the frozen config, with all features and without each group."""
    dev = split_by_issue(frame, last=TRAIN_LAST_ISSUE)
    test = split_by_issue(frame, first=TEST_FIRST_ISSUE)
    variants = {
        "all features": [],
        "without weather": WEATHER_FEATURES,
        "without momentum": MOMENTUM_FEATURES,
    }
    rows = {}
    for name, dropped in variants.items():
        columns = [c for c in FEATURE_COLUMNS if c not in dropped]
        booster = fit(
            dev, config.params, config.num_boost_round, root, config.use_scale_weight, columns=columns
        )
        pred = predict(booster, test, columns)
        season, _ = gate_scores(test, {"model": pred})
        rows[name] = {
            "test_mae": mae_table(test, {"model": pred}).loc["all", "model"],
            "season_mae": season["model"],
        }
    return pd.DataFrame(rows).T


def _finish(run, decision: gate.GateDecision, version: str, scores: dict, n_rows: int) -> None:
    verdict = "promoted to champion" if decision.promote else "not promoted"
    log.info("registered %s, %s: %s", version, verdict, decision.reason)
    run.summary.update(
        {
            "model_version": version,
            "promoted": decision.promote,
            "gate_reason": decision.reason,
            "gate_rows": n_rows,
            **{f"gate_mae_{k}": v for k, v in scores.items()},
        }
    )


# --- CLI -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["eval", "retrain", "ablation"], required=True)
    p.add_argument("--revision", default="main", help="dataset revision (commit id or branch)")
    p.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH, help="frozen model config")
    p.add_argument("--trials", type=int, default=50, help="eval: number of search trials")
    p.add_argument("--timeout", type=int, default=1200, help="eval: search time budget in seconds")
    p.add_argument("--offline", action="store_true", help="W&B offline and a local registry")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("nectarflow").setLevel(logging.INFO)
    args = parse_args(argv)
    settings = Settings.from_env()
    if not settings.hf_repo:
        log.error("HF_DATASET_REPO is not set")
        return 1
    tracked = args.mode != "ablation"
    if tracked and not args.offline and not os.environ.get("WANDB_API_KEY"):
        log.error("WANDB_API_KEY is not set (use --offline for a local run)")
        return 1

    import wandb

    root, revision = download(
        settings.hf_repo, args.revision, settings.data_dir / "hf-cache", settings.hf_token
    )
    frame = load_frame(root)
    log.info("dataset %s@%s", settings.hf_repo, revision)
    if not tracked:
        print(f"Ablation on 2026, dataset revision {revision}, MAE in kg/day:")
        print(run_ablation(frame, root, ModelConfig.load(args.params)).round(3).to_string())
        return 0

    with (
        wandb.init(
            project=WANDB_PROJECT,
            job_type=f"train-{args.mode}",
            mode="offline" if args.offline else "online",
            dir=str(settings.data_dir),
            config={"dataset": settings.hf_repo, "git_commit": git_commit(), "features": FEATURE_COLUMNS},
        ) as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        workdir = Path(tmp)
        if args.offline:
            registry: Registry = LocalRegistry(settings.data_dir / "registry" / REGISTRY_COLLECTION)
        else:
            registry = WandbRegistry(run, workdir)
        try:
            if args.mode == "eval":
                run_eval(
                    frame,
                    root,
                    revision,
                    registry,
                    run,
                    trials=args.trials,
                    timeout_s=args.timeout,
                    params_path=args.params,
                    workdir=workdir,
                )
            else:
                config = ModelConfig.load(args.params)
                run_retrain(frame, root, revision, registry, run, config=config, workdir=workdir)
        except gate.GateError as err:
            log.error("gate refused: %s", err)
            run.finish(exit_code=1)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
