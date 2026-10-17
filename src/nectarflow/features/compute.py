"""Feature engineering. One row per issue date t0 and horizon h.

Point-in-time rule: a row for t0 only uses information that was known and final on the
morning of t0. Weather comes from the run issued at t0, momentum from final labels up to
t0 - FINAL_LAG_DAYS, and the climatology from years before training and test.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nectarflow.common.config import (
    CLIMATOLOGY_HALF_WINDOW_DAYS,
    CLIMATOLOGY_YEARS,
    FINAL_LAG_DAYS,
    HORIZONS,
    MOMENTUM_WINDOWS,
)


def _doy(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d).dt.dayofyear


def build_climatology(labels: pd.DataFrame) -> pd.DataFrame:
    """Median daily change per day of year over CLIMATOLOGY_YEARS, ±7 days, jumps excluded.

    The median is robust against single faulty scales (review A2). Day-of-year distance is
    circular, so 31 Dec and 1 Jan are neighbours.
    """
    first, last = CLIMATOLOGY_YEARS
    years = pd.to_datetime(labels["date"]).dt.year
    base = labels[(years >= first) & (years <= last)].dropna(subset=["label_clean"])
    base_doy = _doy(base["date"]).to_numpy()
    values = base["label_clean"].to_numpy()

    rows = []
    for doy in range(1, 367):
        dist = np.abs(base_doy - doy)
        dist = np.minimum(dist, 366 - dist)
        window = values[dist <= CLIMATOLOGY_HALF_WINDOW_DAYS]
        rows.append((doy, float(np.median(window)) if len(window) else np.nan, len(window)))
    return pd.DataFrame(rows, columns=["doy", "clim", "n_values"])


def momentum(labels: pd.DataFrame, issue_date: date) -> dict[str, float]:
    """Net change over the last k final days, ending at t0 - FINAL_LAG_DAYS (review A1).

    A window with a missing or flagged day is NaN, so a gap never looks like zero change.
    """
    last_final = issue_date - timedelta(days=FINAL_LAG_DAYS)
    series = labels.set_index("date")["label_clean"]
    out = {}
    for k in MOMENTUM_WINDOWS:
        days = [last_final - timedelta(days=i) for i in range(k)]
        window = series.reindex(days)
        out[f"mom_{k}d"] = float(window.sum()) if window.notna().all() else np.nan
    return out


def build_feature_rows(
    issue_date: date,
    weather_daily: pd.DataFrame,
    labels: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    """The ten feature rows of one issue date. Labels are not part of these rows.

    Training joins the label table on ``target_date`` later; this keeps the feature rows
    immutable once written, while labels can still become final.
    """
    rows = pd.DataFrame({"h": list(HORIZONS)})
    rows.insert(0, "issue_date", issue_date)
    rows.insert(1, "target_date", [issue_date + timedelta(days=h - 1) for h in HORIZONS])
    rows = rows.merge(weather_daily, on="target_date", how="left")

    for name, value in momentum(labels, issue_date).items():
        rows[name] = value

    doy = _doy(rows["target_date"])
    rows["clim_target"] = doy.map(climatology.set_index("doy")["clim"])
    angle = 2 * np.pi * (doy - 1) / 366
    rows["doy_sin"] = np.sin(angle)
    rows["doy_cos"] = np.cos(angle)
    return rows
