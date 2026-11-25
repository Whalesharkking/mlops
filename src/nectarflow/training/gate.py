"""Champion/challenger gate (MS1 review A3, MS2 review E2). The gate alone decides on promotion.

A challenger becomes champion only if, on the season rows of the holdout, all three hold:
it beats the champion by at least GATE_MIN_GAIN_KG, it beats persistence and it beats
climatology. Two season rules keep winter out (MS1 review A3): the holdout must end in
March to September, and it must hold at least GATE_MIN_ROWS rows with a target in that
window. So from October to February nothing is ever promoted.

Without a champion the champion condition is skipped and the baselines still apply. This
is only used by the evaluation protocol for the first champion, the weekly retrain refuses
to run without one.
"""

from dataclasses import dataclass
from datetime import date

from nectarflow.common.config import GATE_MIN_GAIN_KG, GATE_MIN_ROWS, GATE_MONTHS
from nectarflow.training.evaluate import BASELINES


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str


class GateError(RuntimeError):
    """The comparison would not be fair, so no decision is made."""


def check_unseen(champion_train_end: date | None, first_holdout_target: date) -> None:
    """The champion must not have been trained on any holdout target (fair comparison)."""
    if champion_train_end is not None and champion_train_end >= first_holdout_target:
        raise GateError(
            f"champion was trained on targets up to {champion_train_end}, "
            f"the holdout starts at {first_holdout_target}"
        )


def decide(
    scores: dict[str, float], n_rows: int, champion: float | None, holdout_end: date | None = None
) -> GateDecision:
    """``scores`` holds the season MAE of ``challenger`` and of both baselines.

    ``holdout_end`` is the last holdout issue date of the weekly retrain. The first champion
    (evaluation protocol, test on 2026) passes None, its test set is not a recent holdout.
    """
    if holdout_end is not None and holdout_end.month not in GATE_MONTHS:
        return GateDecision(False, f"off season, the holdout ends on {holdout_end}")
    if n_rows < GATE_MIN_ROWS:
        return GateDecision(False, f"only {n_rows} season rows in the holdout (need {GATE_MIN_ROWS})")
    mae = scores["challenger"]
    for name in BASELINES:
        if not mae < scores[name]:
            return GateDecision(False, f"MAE {mae:.4f} does not beat {name} ({scores[name]:.4f})")
    if champion is None:
        return GateDecision(True, f"no champion yet, MAE {mae:.4f} beats both baselines")
    if mae > champion - GATE_MIN_GAIN_KG:
        return GateDecision(
            False, f"MAE {mae:.4f} is not {GATE_MIN_GAIN_KG} below the champion ({champion:.4f})"
        )
    return GateDecision(True, f"MAE {mae:.4f} beats the champion ({champion:.4f}) and both baselines")
