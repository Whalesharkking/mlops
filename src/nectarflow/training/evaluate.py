"""MAE per horizon for the baselines and a model (MS2 review B3, MS1 review B1).

    python -m nectarflow.training.evaluate --until 2026-11-20              # baselines on 2026
    python -m nectarflow.training.evaluate --from 2025-01-01 --until 2025-12-22 --revision <sha>

All predictors are scored on the same rows: rows with a final raw label and a persistence
value (persistence is empty when a day of its window is missing or flagged).
"""

import argparse
import logging
import sys
from datetime import date

import numpy as np
import pandas as pd

from nectarflow.common.config import EVAL_TARGET, GATE_MONTHS, TEST_FIRST_ISSUE, Settings
from nectarflow.training.data import download, load_frame, split_by_issue

log = logging.getLogger(__name__)

PERSISTENCE = "persistence"
CLIMATOLOGY = "climatology"
BASELINES = (PERSISTENCE, CLIMATOLOGY)


def baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Persistence repeats the last final day (t0 - 2), climatology is the 2019-2023 median."""
    return {
        PERSISTENCE: frame["mom_1d"].to_numpy(dtype=float),
        CLIMATOLOGY: frame["clim_target"].to_numpy(dtype=float),
    }


def scoring_mask(frame: pd.DataFrame) -> np.ndarray:
    return (frame[EVAL_TARGET].notna() & frame["mom_1d"].notna() & frame["clim_target"].notna()).to_numpy()


def in_season(frame: pd.DataFrame) -> np.ndarray:
    return pd.to_datetime(frame["target_date"]).dt.month.isin(GATE_MONTHS).to_numpy()


def mae_table(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per horizon plus an ``all`` row: row count and the MAE of every predictor."""
    mask = scoring_mask(frame)
    df = frame.loc[mask, ["h"]].copy()
    y = frame.loc[mask, EVAL_TARGET].to_numpy(dtype=float)
    for name, pred in predictions.items():
        df[name] = np.abs(np.asarray(pred, dtype=float)[mask] - y)
    per_h = df.groupby("h").agg(n=("h", "size"), **{n: (n, "mean") for n in predictions})
    total = pd.DataFrame({"n": [len(df)], **{n: [df[n].mean()] for n in predictions}}, index=["all"])
    table = pd.concat([per_h, total])
    table.index.name = "h"
    return table


def success_criterion(table: pd.DataFrame, model: str) -> pd.Series:
    """Proposal criterion per horizon: beat the stronger baseline by 10 % for h = 1..5, match it for 6..10."""
    rows = table.drop(index="all")
    best = rows[list(BASELINES)].min(axis=1)
    factor = np.where(rows.index.astype(int) <= 5, 0.9, 1.0)
    return rows[model] <= best * factor


def gate_scores(frame: pd.DataFrame, predictions: dict[str, np.ndarray]) -> tuple[dict[str, float], int]:
    """Season MAE of every predictor (target month March to September) and the row count."""
    mask = scoring_mask(frame) & in_season(frame)
    if not mask.any():
        return {n: float("nan") for n in predictions}, 0
    y = frame.loc[mask, EVAL_TARGET].to_numpy(dtype=float)
    scores = {n: float(np.mean(np.abs(np.asarray(p, dtype=float)[mask] - y))) for n, p in predictions.items()}
    return scores, int(mask.sum())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="first", type=date.fromisoformat, default=TEST_FIRST_ISSUE)
    p.add_argument("--until", type=date.fromisoformat, required=True, help="last issue date")
    p.add_argument("--revision", default="main", help="dataset revision (commit id or branch)")
    p.add_argument("--season", action="store_true", help="only targets in March to September")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    settings = Settings.from_env()
    if not settings.hf_repo:
        log.error("HF_DATASET_REPO is not set")
        return 1
    root, revision = download(
        settings.hf_repo, args.revision, settings.data_dir / "hf-cache", settings.hf_token
    )
    frame = split_by_issue(load_frame(root), last=args.until, first=args.first)
    if args.season:
        frame = frame[in_season(frame)]
    print(f"dataset {settings.hf_repo}@{revision}, issue dates {args.first}..{args.until}")
    print(mae_table(frame, baseline_predictions(frame)).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
