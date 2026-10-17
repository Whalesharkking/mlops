"""Label: net weight change of one local calendar day, midnight to midnight."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nectarflow.common.config import (
    FINAL_LAG_DAYS,
    FLOW_SEASON_MONTHS,
    JUMP_THRESHOLD_FLOW_SEASON_KG,
    JUMP_THRESHOLD_OFF_SEASON_KG,
    MIDNIGHT_TOLERANCE_MINUTES,
    TZ,
)


def midnight_weights(series: pd.DataFrame) -> pd.Series:
    """Weight at each local midnight, indexed by the date that starts at that midnight.

    Uses the point closest to midnight within the tolerance. Older data sits on :30, newer
    data on :00, so a strict 00:00 lookup would lose 2019-2020. Missing midnights stay
    missing; nothing is interpolated.
    """
    local = series["ts"].dt.tz_convert(TZ)
    nearest = (local + pd.Timedelta(hours=12)).dt.normalize()
    distance = (local - nearest).abs()
    candidates = pd.DataFrame(
        {
            "date": nearest.dt.date,
            "distance": distance,
            # :30 data has two points 30 min from midnight; take the one after it, consistently
            "before": (local < nearest).to_numpy(),
            "weight": series["weight"].to_numpy(),
        }
    )
    candidates = candidates[candidates["distance"] <= pd.Timedelta(minutes=MIDNIGHT_TOLERANCE_MINUTES)]
    best = candidates.sort_values(["distance", "before"]).drop_duplicates("date").sort_values("date")
    return best.set_index("date")["weight"]


def is_jump(label: pd.Series, dates: pd.Series) -> pd.Series:
    """Flag implausible daily changes (snow load, faulty or joining scales)."""
    months = pd.to_datetime(dates).dt.month
    threshold = np.where(
        months.isin(FLOW_SEASON_MONTHS), JUMP_THRESHOLD_FLOW_SEASON_KG, JUMP_THRESHOLD_OFF_SEASON_KG
    )
    return label.abs() > threshold


def daily_labels(series: pd.DataFrame, asof: date) -> pd.DataFrame:
    """One row per day that has ended by ``asof``.

    Columns: ``date``, ``label`` (kg/day, NaN if a midnight is missing), ``is_final``,
    ``jump_flag``. ``label_clean`` is the label with jump days removed; features and the
    climatology use it.
    """
    mw = midnight_weights(series)
    first, last = min(mw.index), asof - timedelta(days=1)
    days = pd.date_range(first, last, freq="D").date
    start = mw.reindex(days).to_numpy()
    end = mw.reindex([d + timedelta(days=1) for d in days]).to_numpy()
    out = pd.DataFrame({"date": days, "label": end - start})
    out["is_final"] = out["date"] <= asof - timedelta(days=FINAL_LAG_DAYS)
    out["jump_flag"] = is_jump(out["label"], out["date"]) & out["label"].notna()
    out["label_clean"] = out["label"].where(~out["jump_flag"])
    return out
