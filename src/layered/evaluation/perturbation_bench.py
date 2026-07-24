"""Score a perturbation run against its clean baseline (Tier-1 A / B / C).

The perturbation *arms* live in ``src.layered.perturb`` and produce ordinary run files.
This module reads a baseline run and one or more perturbed runs and answers the question
each arm was built to raise. It imports only ``runs``/``pm_runs``/``ic`` — no
``layered.pm`` — so it is safe to re-export from the evaluation package.

  * **A — leak / "unlearning" (Canayaz).** A content perturbation that should flip a
    reasoning model's call (``signflip_momentum``, ``counterfactual_path``). A high
    ``flip_rate`` means the model followed the altered evidence; a low one means it
    ignored the perturbation — the fingerprint of recall. ``direction_response``.
  * **B — scrambled prior (Han).** The PM read reports rotated under the wrong driver
    labels. If its calls move a lot versus the clean run, it read the evidence; if they
    barely move, it answered from the label — a recited prior. ``scramble_response``.
  * **C — robustness battery (Homo Silicus).** Meaning-preserving variants
    (``whitespace``, ``reword_scaffold``, ``reorder_features``) that should not change a
    reasoning model's ordering. ``ic_stability`` reports the IC per variant and its
    dispersion; a wide spread is fragility, not skill.

Two arms report a *rate of change*, and the direction of "good" is opposite between
them: A and B want a large response to the perturbation (the model is reading the
evidence), C wants a small one (the model is invariant to meaning-preserving noise).
Stated on each function so the sign is never read backwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.layered.evaluation.ic import ICEvaluator
from src.layered.evaluation.pm_runs import load_pm_run
from src.layered.evaluation.runs import load_run


def _aligned_signed(baseline_path: str, perturbed_path: str) -> pd.DataFrame:
    """The two analyst signed-conviction series on their shared meeting dates."""
    b, p = load_run(baseline_path), load_run(perturbed_path)
    return pd.concat([b.signed.rename("base"), p.signed.rename("pert")],
                     axis=1).dropna()


# ── A: leak / unlearning ─────────────────────────────────────────────────────────
def direction_response(baseline_path: str, perturbed_path: str) -> dict:
    """How far an analyst's call moved under a content perturbation (arm A).

    ``flip_rate`` is the fraction of non-flat baseline meetings whose sign *reversed* —
    the response a ``signflip_momentum``/``counterfactual_path`` arm should produce if
    the model reads the evidence. **High is following-the-evidence; low is the
    recall fingerprint.** A reversal is a strictly opposite sign: a move to exactly flat
    is a withdrawn call, not a reversal, so it is counted in ``n_to_flat`` rather than
    inflating ``flip_rate`` (the earlier ``sign(base) != sign(pert)`` test folded it in).
    Reported with the raw mean absolute change in signed conviction, which does not need
    a sign to be informative.
    """
    a = _aligned_signed(baseline_path, perturbed_path)
    nz = a[a["base"] != 0]
    reversed_ = np.sign(nz["pert"]) == -np.sign(nz["base"])
    return {
        "n": int(len(a)),
        "n_nonflat": int(len(nz)),
        "flip_rate": float(reversed_.mean()) if len(nz) else float("nan"),
        "n_to_flat": int((nz["pert"] == 0).sum()),
        "mean_abs_change": float((a["pert"] - a["base"]).abs().mean()) if len(a) else float("nan"),
        "corr": float(a["base"].corr(a["pert"])) if len(a) > 2 else float("nan"),
    }


# ── C: robustness battery ────────────────────────────────────────────────────────
def ic_stability(variants: dict[str, str], steps: int = 1) -> pd.DataFrame:
    """IC of each meaning-preserving variant, for reading the *spread* (arm C).

    ``variants`` maps a label ("baseline", "whitespace", …) to a run path. Each is scored
    by its own ``ICEvaluator`` on its own level (unchanged by a meaning-preserving
    perturbation). A tight cluster of ICs is robustness; a wide spread is fragility —
    the signal only survived one exact phrasing. **Small dispersion is good here.**
    """
    rows = []
    for label, path in variants.items():
        run = load_run(path)
        ev = ICEvaluator(run.level.dropna(), steps=steps)
        res = ev.evaluate(run.signed, label)
        rows.append({"variant": label, "n": res.n, "ic": res.ic, "t_stat": res.t_stat,
                     "hit_rate": res.hit_rate})
    out = pd.DataFrame(rows).set_index("variant")
    return out


def ic_dispersion(table: pd.DataFrame) -> dict:
    """The one-line read of an ``ic_stability`` table: how far the ICs spread."""
    ics = table["ic"].dropna()
    if ics.empty:
        return {"n_variants": 0}
    return {"n_variants": int(len(ics)), "ic_mean": float(ics.mean()),
            "ic_std": float(ics.std(ddof=0)), "ic_min": float(ics.min()),
            "ic_max": float(ics.max()), "ic_range": float(ics.max() - ics.min())}


# ── B: scrambled prior ───────────────────────────────────────────────────────────
def scramble_response(baseline_pm_path: str, scrambled_pm_path: str) -> dict:
    """How far the PM's calls moved when reports were mislabeled (arm B).

    Per (driver, meeting), the fraction whose sign changed between the clean and the
    scrambled run. **High means the PM read the (mislabeled) evidence; low means it
    answered from the driver label — a recited prior.** Pooled across drivers, plus the
    per-driver breakdown, because a PM might read some drivers' evidence and recite
    others'.
    """
    b, s = load_pm_run(baseline_pm_path), load_pm_run(scrambled_pm_path)
    drivers = [d for d in b.drivers if d in s.drivers]
    per_driver, pooled_flips, pooled_n = {}, 0, 0
    for d in drivers:
        pair = pd.concat([b.frame[d].rename("base"), s.frame[d].rename("scr")],
                         axis=1).dropna()
        nz = pair[pair["base"] != 0]
        if len(nz):
            flips = int((np.sign(nz["base"]) != np.sign(nz["scr"])).sum())
            per_driver[d] = {"n": int(len(nz)), "flip_rate": flips / len(nz)}
            pooled_flips += flips
            pooled_n += len(nz)
    return {
        "n": pooled_n,
        "flip_rate": (pooled_flips / pooled_n) if pooled_n else float("nan"),
        "per_driver": per_driver,
    }


# ── any two PM arms ──────────────────────────────────────────────────────────────
def pm_response(baseline_pm_path: str, other_pm_path: str) -> dict:
    """How far a PM's calls moved between two arms, on the strict-reversal rule.

    Generic on purpose — it inspects neither run's config — so it serves the scramble,
    the date-disclosure leak probe, and any two PM runs sharing a meeting grid.

    Two deliberate differences from ``scramble_response``, which is left alone because
    its definition is already committed to a recorded verdict:

      * **A move to exactly flat is a withdrawn call, not a reversal**, so it lands in
        ``n_to_flat`` instead of inflating ``flip_rate`` — the rule
        ``direction_response`` already uses. The two functions therefore do *not*
        report interchangeable numbers, and a disclosure arm's most plausible response
        (withdrawing or sharpening a call rather than reversing it) is exactly the case
        where they diverge.
      * **``per_meeting`` is returned**, because drivers within a meeting share one
        brief, one panel and one call. The meeting is the unit of observation; a
        t-statistic taken over cells would be inflated by roughly ``sqrt(k)`` for a
        k-driver pod. Feed this series to ``paired_arm_test``, never the cells.
    """
    b, s = load_pm_run(baseline_pm_path), load_pm_run(other_pm_path)
    drivers = [d for d in b.drivers if d in s.drivers]
    per_driver, per_meeting = {}, {}
    pooled_flips = pooled_flat = pooled_n = 0
    deltas: list[pd.Series] = []
    for d in drivers:
        pair = pd.concat([b.frame[d].rename("base"), s.frame[d].rename("other")],
                         axis=1).dropna()
        if pair.empty:
            continue
        deltas.append((pair["other"] - pair["base"]).abs())
        nz = pair[pair["base"] != 0]
        if not len(nz):
            continue
        rev = np.sign(nz["other"]) == -np.sign(nz["base"])
        per_driver[d] = {"n": int(len(nz)), "flip_rate": float(rev.mean())}
        pooled_flips += int(rev.sum())
        pooled_flat += int((nz["other"] == 0).sum())
        pooled_n += len(nz)
        for ts, flipped in rev.items():
            hit, tot = per_meeting.get(ts, (0, 0))
            per_meeting[ts] = (hit + int(flipped), tot + 1)

    meetings = pd.Series({ts: h / t for ts, (h, t) in per_meeting.items()
                          if t}).sort_index()
    alld = pd.concat(deltas) if deltas else pd.Series(dtype=float)
    return {
        "n": pooled_n,
        "n_meetings": int(len(meetings)),
        "flip_rate": (pooled_flips / pooled_n) if pooled_n else float("nan"),
        "n_to_flat": pooled_flat,
        "mean_abs_change": float(alld.mean()) if len(alld) else float("nan"),
        "per_driver": per_driver,
        "per_meeting": meetings,
    }


def paired_arm_test(a: pd.Series, b: pd.Series) -> dict:
    """Paired mean / se / t on two arms' per-meeting series, on their shared meetings.

    The clustered test the decision rules are written against: one observation per
    meeting, differenced within meeting so the panel's own month-to-month variation
    cancels. ``n`` is meetings, and it is the ``n`` any t-statistic here must quote.
    """
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = pair["a"] - pair["b"]
    n = int(len(d))
    if n < 2:
        return {"n": n, "mean": float("nan"), "sd": float("nan"),
                "se": float("nan"), "t": float("nan")}
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else float("nan")
    return {"n": n, "mean": float(d.mean()), "sd": sd, "se": se,
            "t": float(d.mean() / se) if se and np.isfinite(se) else float("nan")}
