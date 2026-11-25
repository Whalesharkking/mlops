"""Read the training data from the feature store at one pinned dataset revision (MS2 review E1).

A model is reproducible from the revision alone: the revision fixes every feature file and
the label table, including which labels were final at that time.
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from nectarflow.common.config import N_SCALES_ESTIMATE
from nectarflow.features import store as fs

log = logging.getLogger(__name__)

# Only what training needs. The raw weather runs (about 1000 files) are not downloaded.
TRAINING_PATTERNS = ["features/*.parquet", fs.LABELS_PATH, fs.N_SCALES_PATH]


def download(repo_id: str, revision: str, cache_dir: Path, token: str | None = None) -> tuple[Path, str]:
    """The training files of one revision (see ``store.download_revision``)."""
    return fs.download_revision(repo_id, revision, cache_dir, TRAINING_PATTERNS, token)


def load_frame(root: Path) -> pd.DataFrame:
    """All feature rows joined with their label on ``target_date``, final labels only.

    The join is on the target date, never on the issue date: the row for (t0, h) gets the
    label of day t0 + h - 1. Rows whose target day is not final yet are dropped.
    """
    files = sorted((root / "features").glob("issue_date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no feature files under {root / 'features'}")
    features = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    labels = pd.read_parquet(root / fs.LABELS_PATH)
    labels = labels.loc[labels["is_final"], ["date", "label", "label_clean"]]

    frame = features.merge(labels, left_on="target_date", right_on="date", how="inner").drop(columns="date")
    for col in ("issue_date", "target_date"):
        frame[col] = pd.to_datetime(frame[col]).dt.date
    frame = frame.sort_values(["issue_date", "h"]).reset_index(drop=True)
    log.info(
        "%d rows with a final label, issue dates %s..%s",
        len(frame),
        frame["issue_date"].min(),
        frame["issue_date"].max(),
    )
    return frame


def scale_weights(target_dates: pd.Series, root: Path | None = None) -> np.ndarray:
    """Sample weight proportional to the number of scales on the target day (MS2 review E4).

    The mean of more scales is less noisy. Before the first recorded count the HiveWatch
    linear estimate is used. Weights are scaled to mean 1.
    """
    (d0, n0), (d1, n1) = N_SCALES_ESTIMATE
    days = pd.to_datetime(target_dates)
    t = (days - pd.Timestamp(d0)).dt.days / (pd.Timestamp(d1) - pd.Timestamp(d0)).days
    n = n0 + t.clip(0, 1) * (n1 - n0)
    if root is not None and (root / fs.N_SCALES_PATH).exists():
        recorded = pd.read_parquet(root / fs.N_SCALES_PATH).dropna(subset=["n_scales"])
        recorded = recorded.set_index(pd.to_datetime(recorded["poll_date"]))["n_scales"]
        n = n.where(~days.isin(recorded.index), days.map(recorded))
    weights = n.to_numpy(dtype=float)
    return weights / weights.mean()


def split_by_issue(frame: pd.DataFrame, last: date | None = None, first: date | None = None) -> pd.DataFrame:
    """Rows with ``first <= issue_date <= last`` (either bound optional)."""
    mask = pd.Series(True, index=frame.index)
    if first is not None:
        mask &= frame["issue_date"] >= first
    if last is not None:
        mask &= frame["issue_date"] <= last
    return frame[mask]
