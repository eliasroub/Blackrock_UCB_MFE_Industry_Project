"""Grade the PM against the analyst it is meant to improve on.

The benchmark is deliberately within-driver: for each driver, compare the PM's
conviction about it against **that driver's own analyst's** conviction, both scored by
the same ``ICEvaluator`` on the same outcome. It answers the question the layer exists
to raise — does an agent that read all seven reports call inflation better than the
inflation specialist did? — and it is robust to the one confound in the corpus (two
legs re-run at a later model vintage), because every comparison is between two signals
about the same driver drawn from the same file.

**The level series must be rebuilt on the PM's clock.** This is the trap the whole
module is arranged around. ``ICEvaluator`` aligns signal to outcome by index *label*,
so grading a month-end PM series against the inflation analyst's own CPI-15th level
series produces an empty join, an ``n`` near zero, and no warning whatsoever. So the
outcome is recomputed here at the meeting dates with ``FeaturePanel`` — free, offline,
and through the same ``AsOf`` gate the analyst used.

That correction propagates: the analyst baseline must be re-scored on the same clock
too. An analyst's *published* IC is against its own release clock and is a different
question ("did it call the next CPI print?" versus "did it call the level at month
end?"). The two numbers are not comparable and the table says so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import yaml

# Prefix-dispatching loader (EQ_* → data/equity, else FRED) so the equities pod's
# drivers are priceable here; pure-FRED inputs load exactly as before.
from src.data.equity_local import load_any_bundle as load_bundle
from src.layered.evaluation.ic import ICEvaluator, required_ic
from src.layered.evaluation.panel import FeaturePanel
from src.layered.features import FeatureEngine, from_persona
from src.layered.pm.board import ViewBoard

PERSONA_DIR = Path(__file__).resolve().parents[1] / "analysts" / "personas"


def _engine(driver: str, persona_dir: Optional[Path] = None) -> FeatureEngine:
    path = (persona_dir or PERSONA_DIR) / f"{driver}.yaml"
    persona = yaml.safe_load(path.read_text()) or {}
    return FeatureEngine(from_persona(driver, persona))


def driver_levels(drivers: list[str], dates: pd.DatetimeIndex,
                  macro: Optional[dict] = None,
                  persona_dir: Optional[Path] = None) -> pd.DataFrame:
    """Each driver's headline measurement at every meeting date.

    One column per driver, computed from its persona's own ``level_feature`` so the
    outcome the PM is graded on is the same quantity the analyst was graded on — only
    observed on the PM's clock instead of the analyst's.
    """
    engines = {d: _engine(d, persona_dir) for d in drivers}
    if macro is None:
        inputs = sorted({s for e in engines.values() for s in e.inputs})
        macro = load_bundle(inputs)
    out = {}
    for d, eng in engines.items():
        panel = FeaturePanel(eng)
        out[d] = panel.level(panel.build(macro, dates))
    return pd.DataFrame(out, index=dates)


def analyst_snap(board: ViewBoard, dates: pd.DatetimeIndex,
                 drivers: Optional[list[str]] = None) -> pd.DataFrame:
    """The baseline: each analyst's latest signed conviction known at each meeting.

    This is what the PM had in front of it, so it is the honest thing to beat — not
    the analyst's own-clock published number, which was scored against a different
    outcome.
    """
    cols = drivers or board.drivers
    rows = {}
    for ts in dates:
        m = board.at(ts)
        rows[ts] = {d: (m.entries[d].view.signed_conviction
                        if d in m.entries and m.entries[d].present else np.nan)
                    for d in cols}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def consensus_blend(snap: pd.DataFrame, polarity: Mapping[str, float],
                    weight: float = 0.5) -> pd.DataFrame:
    """A mechanical control: half the driver's own analyst, half the oriented panel.

    The point of a PM is to let the other six inform the seventh. This does that with
    arithmetic instead of a model, so "the PM helped" can be separated from "looking
    at the panel at all helped". One constant, declared in advance, nothing fitted.
    """
    p = pd.Series({d: float(polarity.get(d, 1.0)) for d in snap.columns})
    oriented = snap.mul(p, axis=1)
    panel_mean = oriented.mean(axis=1, skipna=True)
    out = {}
    for d in snap.columns:
        out[d] = weight * snap[d] + (1.0 - weight) * float(p[d]) * panel_mean
    return pd.DataFrame(out, index=snap.index)


def benchmark(pm_frame: pd.DataFrame, board: ViewBoard, dates: pd.DatetimeIndex,
              polarity: Mapping[str, float], *, steps: int = 1,
              macro: Optional[dict] = None,
              answer_space: str = "driver") -> pd.DataFrame:
    """One row per driver: PM against its analyst against the mechanical control.

    ``answer_space`` is the pod's declared coordinate system for its convictions and
    MUST match what the pod told the model, because this function grades against each
    driver's own level. A pod that answers on the rate axis is saying "this driver is
    pushing yields up", which for a -1-polarity driver is the opposite sign from "this
    driver's measurement rises" — so its numbers are re-oriented through ``polarity``
    before being scored, and only then is ``ic_pm`` comparable with ``ic_analyst``.

    Grading a rate-axis run as though it were driver-space is exactly the error that
    produced a balance_sheet ``ic_pm`` of -0.167 against an analyst's +0.714 on the
    first duration run. Passing the pod's declaration through rather than assuming a
    default is what stops that recurring silently.
    """
    if answer_space not in ("driver", "rate"):
        raise ValueError(f"answer_space must be 'driver' or 'rate', got {answer_space!r}")
    drivers = list(pm_frame.columns)
    if answer_space == "rate":
        # Back onto the driver axis, so every column below grades the same quantity.
        # A driver with no declared polarity keeps its sign rather than being assumed
        # +1, matching `disagreement.oriented`, which skips rather than defaults.
        pm_frame = pm_frame.mul(pd.Series({d: float(polarity.get(d, 1.0))
                                           for d in drivers}))
    levels = driver_levels(drivers, dates, macro=macro)
    snap = analyst_snap(board, dates, drivers)
    mech = consensus_blend(snap, polarity)

    rows = []
    for d in drivers:
        lv = levels[d].dropna()
        if lv.empty:
            rows.append({"driver": d, "n": 0})
            continue
        ev = ICEvaluator(lv, steps=steps)

        pm_sig = pm_frame[d].dropna()
        # The silent-collapse guard. A label mismatch here yields n≈0 with no error,
        # so it is checked rather than assumed.
        shared = pm_sig.index.intersection(lv.index)
        if len(shared) < 0.5 * len(pm_sig):
            raise ValueError(
                f"{d}: PM signal and level series share only {len(shared)} of "
                f"{len(pm_sig)} dates — the clocks disagree, so the IC would be a "
                f"fiction. Check that the run grid and `dates` match."
            )

        r_pm = ev.evaluate(pm_sig, f"pm:{d}")
        r_an = ev.evaluate(snap[d].dropna(), f"analyst:{d}")
        r_me = ev.evaluate(mech[d].dropna(), f"mech:{d}")
        rows.append({
            "driver": d,
            "n": r_pm.n,
            "ic_analyst": r_an.ic,
            "ic_pm": r_pm.ic,
            "d_ic": r_pm.ic - r_an.ic,
            "ic_mech": r_me.ic,
            "t_analyst": r_an.t_stat,
            "t_pm": r_pm.t_stat,
            "hit_analyst": r_an.hit_rate,
            "hit_pm": r_pm.hit_rate,
            "breadth": ev.breadth,
            "ic_for_ir_1": required_ic(1.0, ev.breadth),
        })
    return pd.DataFrame(rows).set_index("driver")


def summarize(table: pd.DataFrame) -> str:
    """The honest reading of the table.

    Seven drivers at ~12 observations a year is a small sample seven times over. A
    single driver's improvement of 0.05 is noise; what carries information is whether
    the sign is consistent across drivers. Stated in the output rather than left for
    the reader to remember.
    """
    d = table["d_ic"].dropna()
    if d.empty:
        return "no comparable drivers"
    wins = int((d > 0).sum())
    lines = [
        f"PM beat its analyst on {wins}/{len(d)} drivers; mean Δic {d.mean():+.3f}.",
        f"Median n per driver: {table['n'].median():.0f} "
        f"(breadth ≈ {table['breadth'].median():.1f} bets/yr).",
        "A single driver's Δic of ±0.05 is noise at this breadth — read the sign "
        "consistency across drivers and the t-statistics, not any one row.",
        "Analyst ICs here are re-scored on the PM's month-end clock and so differ "
        "from the published per-driver numbers, which grade a different outcome.",
    ]
    return "\n".join(lines)
