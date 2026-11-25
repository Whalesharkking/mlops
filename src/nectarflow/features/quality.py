"""Label quality from the as-of archive (MS1 review A1, MS2 review A1/E3).

    python -m nectarflow.features.quality              # report on the newest dataset revision

Every daily run stores the labels of the last 30 days as known that morning
(``asof/labels/poll_date=*.parquet``). From these snapshots this report measures how often
yesterday's label (t0 - 1) and the label of t0 - 2 were still missing at poll time, per
week, and how many labels changed after they first appeared, and by how much.
"""

import argparse
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from nectarflow.common.config import FINAL_LAG_DAYS, Settings
from nectarflow.features.store import download_revision

log = logging.getLogger(__name__)

CHANGE_TOLERANCE_KG = 1e-6


def load_snapshots(root: Path) -> pd.DataFrame:
    files = sorted((root / "asof" / "labels").glob("poll_date=*.parquet"))
    if not files:
        raise FileNotFoundError(f"no as-of snapshots under {root}")
    snaps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    for col in ("date", "poll_date"):
        snaps[col] = pd.to_datetime(snaps[col]).dt.date
    return snaps


def availability(snaps: pd.DataFrame) -> pd.DataFrame:
    """Per poll date: was the label of t0 - 1 and of t0 - FINAL_LAG_DAYS known that morning?"""
    present = snaps[snaps["label"].notna()]
    known = set(zip(present["poll_date"], present["date"], strict=True))
    polls = sorted(snaps["poll_date"].unique())
    return pd.DataFrame(
        {
            "poll_date": polls,
            "t0_minus_1": [(p, p - timedelta(days=1)) in known for p in polls],
            "t0_minus_2": [(p, p - timedelta(days=FINAL_LAG_DAYS)) in known for p in polls],
        }
    )


def weekly_missing_share(avail: pd.DataFrame) -> pd.DataFrame:
    """Share of polls per ISO week whose t0 - 1 (and t0 - 2) label was missing."""
    week = pd.to_datetime(avail["poll_date"]).dt.strftime("%G-W%V")
    return (
        avail.assign(week=week)
        .groupby("week")
        .agg(
            polls=("poll_date", "size"),
            missing_t0_1=("t0_minus_1", lambda s: 1 - s.mean()),
            missing_t0_2=("t0_minus_2", lambda s: 1 - s.mean()),
        )
    )


def label_changes(snaps: pd.DataFrame) -> dict[str, float]:
    """Compare each label with its first appearance in the archive."""
    seen = snaps[snaps["label"].notna()].sort_values("poll_date")
    first = seen.groupby("date")["label"].transform("first")
    diff = (seen["label"] - first).abs()
    changed = diff.groupby(seen["date"]).max()
    return {
        "labels": int(changed.size),
        "changed": int((changed > CHANGE_TOLERANCE_KG).sum()),
        "max_change_kg": float(changed.max()) if changed.size else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--revision", default="main")
    args = p.parse_args(argv)
    settings = Settings.from_env()
    if not settings.hf_repo:
        log.error("HF_DATASET_REPO is not set")
        return 1
    root, revision = download_revision(
        settings.hf_repo, args.revision, settings.data_dir / "hf-cache", ["asof/labels/*"], settings.hf_token
    )
    snaps = load_snapshots(root)
    avail = availability(snaps)
    first, last = avail["poll_date"].min(), avail["poll_date"].max()
    print(f"dataset {settings.hf_repo}@{revision}, {len(avail)} polls {first}..{last}")
    missing_1, missing_2 = int((~avail["t0_minus_1"]).sum()), int((~avail["t0_minus_2"]).sum())
    print(f"t0-1 missing on {missing_1} of {len(avail)} polls, t0-2 missing on {missing_2}")
    print(weekly_missing_share(avail).round(2).to_string())
    c = label_changes(snaps)
    print(f"{c['changed']} of {c['labels']} labels changed after first appearing", end=", ")
    print(f"max change {c['max_change_kg']:.4f} kg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
