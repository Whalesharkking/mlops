"""Feature pipeline: daily run and backfill share this code path.

    python -m nectarflow.features.pipeline                       # today (daily job)
    python -m nectarflow.features.pipeline --start 2024-03-14    # backfill up to today
    python -m nectarflow.features.pipeline --local               # write to DATA_DIR only

Each run polls the full HiveWatch history (one request), recomputes labels and the
climatology, archives the as-of state of the recent series and the scale count, and then
writes the feature rows of every requested issue date that is not in the store yet.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from nectarflow.common.config import TZ, WEATHER_ARCHIVE_START, Settings
from nectarflow.features import store as fs
from nectarflow.features.compute import build_climatology, build_feature_rows
from nectarflow.features.hivewatch import fetch_full_history, fetch_n_scales
from nectarflow.features.labels import daily_labels
from nectarflow.features.weather import IncompleteRunError, daily_weather, fetch_run

log = logging.getLogger("nectarflow.features")

AS_OF_SNAPSHOT_DAYS = 30
CHECKPOINT_EVERY = 100  # backfill: upload every 100 issue dates, so a crash loses little


@dataclass
class RunReport:
    written: list[date] = field(default_factory=list)
    skipped: list[date] = field(default_factory=list)
    incomplete: list[date] = field(default_factory=list)


def fetch_run_waiting(issue_date: date, session: requests.Session, wait_minutes: int) -> pd.DataFrame:
    """Fetch a run; if it is still incomplete, poll again every 10 minutes up to the limit."""
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        try:
            return fetch_run(issue_date, session)
        except IncompleteRunError as err:
            if time.monotonic() >= deadline:
                raise
            log.info("%s, retrying in 10 min", err)
            time.sleep(600)


def ingest_hive(store: fs.FeatureStore, today: date, session: requests.Session) -> pd.DataFrame:
    """Poll HiveWatch, write labels, as-of snapshot and scale count. Returns the labels."""
    series = fetch_full_history(today, session)
    labels = daily_labels(series, asof=today)
    store.write(fs.LABELS_PATH, labels)

    # As-of snapshot (review A1): the recent series exactly as polled today, so MS3 can
    # measure how much the last days still change before they are final.
    since = pd.Timestamp(today - timedelta(days=AS_OF_SNAPSHOT_DAYS), tz="UTC")
    snapshot = series[series["ts"] >= since].assign(poll_date=today)
    store.write(fs.poll_path(today), snapshot)

    # Scale count (review A2). HiveWatch keeps no history, so we build it from today on.
    n_scales = fetch_n_scales(session)
    history = store.read(fs.N_SCALES_PATH)
    row = pd.DataFrame({"poll_date": [today], "n_scales": [n_scales]})
    history = row if history is None else pd.concat([history[history["poll_date"] != today], row])
    store.write(fs.N_SCALES_PATH, history.sort_values("poll_date"))

    labelled = labels[labels["label"].notna()]
    log.info(
        "hive: %d labelled days up to %s, %d of them final, %d jump-flagged, %d scales",
        len(labelled),
        labels["date"].max(),
        int(labelled["is_final"].sum()),
        int(labels["jump_flag"].sum()),
        n_scales,
    )
    return labels


def run(
    start: date,
    end: date,
    store: fs.FeatureStore,
    *,
    today: date,
    force: bool = False,
    wait_minutes: int = 0,
    session: requests.Session | None = None,
    request_interval_s: float = 1.5,
) -> RunReport:
    http = session or requests.Session()
    report = RunReport()
    if end > today:
        raise ValueError(f"end {end} lies in the future (today is {today})")
    if start < WEATHER_ARCHIVE_START:
        log.warning("no archived runs before %s, starting there", WEATHER_ARCHIVE_START)
        start = WEATHER_ARCHIVE_START

    labels = ingest_hive(store, today, http)
    climatology = build_climatology(labels)
    store.write(fs.CLIMATOLOGY_PATH, climatology)

    span = f"{start}" if start == end else f"{start}..{end}"
    issue_date = start
    try:
        while issue_date <= end:
            if not force and store.exists(fs.features_path(issue_date)):
                report.skipped.append(issue_date)
            else:
                try:
                    wait = wait_minutes if issue_date == today else 0
                    hourly = fetch_run_waiting(issue_date, http, wait)
                except IncompleteRunError as err:
                    log.warning("%s", err)
                    report.incomplete.append(issue_date)
                else:
                    store.write(fs.run_path(issue_date), hourly)
                    weather = daily_weather(hourly, issue_date)
                    rows = build_feature_rows(issue_date, weather, labels, climatology)
                    store.write(fs.features_path(issue_date), rows)
                    report.written.append(issue_date)
                    if len(report.written) % CHECKPOINT_EVERY == 0:
                        store.push(f"features {span}: checkpoint at {issue_date}")
                if start != end:
                    time.sleep(request_interval_s)  # stay below the Open-Meteo rate limit
            issue_date += timedelta(days=1)
    finally:
        # Also on an error: keep what was computed, a rerun skips it.
        store.push(f"features {span}: {len(report.written)} issue dates (poll {today})")
    log.info(
        "done: %d written, %d already in store, %d incomplete runs %s",
        len(report.written),
        len(report.skipped),
        len(report.incomplete),
        [d.isoformat() for d in report.incomplete],
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=date.fromisoformat, help="first issue date (default: today)")
    p.add_argument("--end", type=date.fromisoformat, help="last issue date (default: today)")
    p.add_argument("--local", action="store_true", help="do not upload to the Hugging Face Hub")
    p.add_argument("--force", action="store_true", help="recompute issue dates already in the store")
    p.add_argument(
        "--wait-minutes",
        type=int,
        default=0,
        help="for today's run: keep polling this long while the run is incomplete",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Our own messages at INFO, third-party HTTP request logs only from WARNING on.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("nectarflow").setLevel(logging.INFO)
    args = parse_args(argv)
    settings = Settings.from_env()
    today = datetime.now(TZ).date()
    start = args.start or today
    end = args.end or today

    if args.local or not settings.hf_repo:
        if not args.local:
            log.warning("HF_DATASET_REPO not set, running in local mode")
        store = fs.FeatureStore(settings.data_dir)
    else:
        if not settings.hf_token:
            log.error("HF_TOKEN is required to write to %s", settings.hf_repo)
            return 1
        store = fs.FeatureStore(settings.data_dir, settings.hf_repo, settings.hf_token)

    report = run(start, end, store, today=today, force=args.force, wait_minutes=args.wait_minutes)
    # The daily job must fail loudly if today's run never became complete.
    if today in report.incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
