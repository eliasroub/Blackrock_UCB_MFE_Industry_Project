"""What an analyst actually leaned on, and when it changed its mind about that.

``input_ranking`` gives, per meeting, every measurement the analyst was handed with a
``pull`` (the direction it pushed THE VIEW, not the direction the input moved) and a
``weight`` 0-1 — including the ones it ignored, at weight 0. That last part is what
makes this an attribution rather than a citation list: an input missing from the field
is indistinguishable from one never read, so the contract requires all of them.

Two different questions live in that data and a single mean weight answers only one:

  WHICH   which measurements carry the view          attention_summary, weight_panel
  WHEN    does that shift, and with what             signed_panel, top_input,
                                                     concentration, pull_flips

The distinction is diagnostic, not cosmetic. An analyst that always leans on the same
input is *rigid* — it will miss a regime change by construction. One that rotates is
*responsive*, or is chasing noise, and which of those it is depends on whether the
rotation lines up with anything real. Neither shows up in an IC.

None of this may inform a prompt — the honesty rule from ``docs/analyst-layer.md`` §6.
It is offline over saved runs and free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PULL_SIGN", "weight_panel", "signed_panel", "top_input", "concentration",
    "pull_flips", "attention_summary", "regime_attention",
]

PULL_SIGN = {"up": 1.0, "down": -1.0, "neutral": 0.0}


def _rankings(run) -> pd.Series:
    """(date -> list of (input, pull, weight)), degraded rows already excluded."""
    v = run.views
    live = v.loc[~v["degraded"], "input_ranking"]
    return live[live.map(len) > 0]


def weight_panel(run) -> pd.DataFrame:
    """``(date x input)`` of raw weight. NaN where the input was not ranked at all.

    NaN and 0 mean different things and are kept apart: 0 is "I read this and it did
    not move me", NaN is "this was not in the list". Collapsing them would turn a
    contract violation into an opinion.
    """
    rows = {}
    for dt, ranking in _rankings(run).items():
        rows[dt] = {name: w for name, _, w in ranking}
    return pd.DataFrame(rows).T.sort_index()


def signed_panel(run) -> pd.DataFrame:
    """``(date x input)`` of ``weight * sign(pull)`` — the contribution to the view.

    This is the panel to plot. Sign carries which way the input pushed the analyst,
    magnitude carries how far, so a diverging colour scale reads directly as "this
    measurement argued up / argued down / was set aside".
    """
    rows = {}
    for dt, ranking in _rankings(run).items():
        rows[dt] = {name: w * PULL_SIGN.get(pull, 0.0) for name, pull, w in ranking}
    return pd.DataFrame(rows).T.sort_index()


def top_input(run) -> pd.DataFrame:
    """Per meeting: the heaviest input, its weight, its pull, and the stated call.

    ``agrees`` is the faithfulness check the field exists to enable — does the input
    the analyst says mattered most push the way it actually called? A run where this
    is often False is one where the prose and the attribution disagree, which is worse
    than either being wrong on its own.
    """
    v = run.views
    out = []
    for dt, ranking in _rankings(run).items():
        name, pull, w = max(ranking, key=lambda x: x[2])
        direction = v.at[dt, "direction"]
        out.append({"asof": dt, "top_input": name, "weight": w, "pull": pull,
                    "direction": direction,
                    "agrees": (pull == direction) if direction in ("up", "down")
                              and pull in ("up", "down") else np.nan})
    return pd.DataFrame(out).set_index("asof")


def concentration(run) -> pd.Series:
    """Herfindahl index of the weights per meeting, in [0, 1].

    Weights are normalised to sum 1 first, so this measures *spread of attention*, not
    total conviction. 1/k means "all k inputs mattered equally"; near 1 means one input
    carried the view. Rising concentration is an analyst narrowing onto a single
    signal, which is worth knowing whether or not its IC moved.
    """
    out = {}
    for dt, ranking in _rankings(run).items():
        w = np.array([x[2] for x in ranking], dtype=float)
        s = w.sum()
        out[dt] = float(((w / s) ** 2).sum()) if s > 0 else np.nan
    return pd.Series(out).sort_index()


def pull_flips(run) -> pd.DataFrame:
    """Per input: how often its pull reversed sign between consecutive meetings.

    A measurement that flips constantly is either genuinely two-sided or the analyst
    has no stable read of it. Compared against ``mean_weight`` this separates "a minor
    input it is inconsistent about" from "the input carrying the view is unstable",
    which are very different problems.
    """
    sp = signed_panel(run)
    rows = []
    for col in sp.columns:
        s = sp[col].dropna()
        nz = s[s != 0]
        flips = int((np.sign(nz).diff() != 0).sum() - 1) if len(nz) > 1 else 0
        rows.append({"input": col, "n_ranked": int(s.notna().sum()),
                     "n_directional": len(nz),
                     "flips": max(0, flips),
                     "flip_rate": round(max(0, flips) / max(1, len(nz) - 1), 3),
                     "mean_weight": round(float(weight_panel(run)[col].mean()), 3)})
    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False)


def attention_summary(run) -> pd.DataFrame:
    """Per input, the WHICH answer: how much, how often top, how often set aside."""
    wp, sp = weight_panel(run), signed_panel(run)
    ti = top_input(run)["top_input"].value_counts()
    n = len(wp)
    rows = []
    for col in wp.columns:
        w = wp[col]
        rows.append({
            "input": col,
            "mean_weight": float(w.mean()),
            "share_top": float(ti.get(col, 0) / n),
            "share_ignored": float((w == 0).sum() / max(1, w.notna().sum())),
            "share_ranked": float(w.notna().sum() / n),
            # net direction of argument over the whole run
            "mean_signed": float(sp[col].mean()),
            "share_up": float((sp[col] > 0).sum() / max(1, sp[col].notna().sum())),
        })
    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False).reset_index(drop=True)


def regime_attention(run, breaks=("2020-01-01", "2022-07-01")) -> pd.DataFrame:
    """Mean weight per input per subperiod — the WHEN answer, as a table.

    Same breaks as ``ic_diagnostics.regime_ic`` so attention shifts can be read
    against skill shifts on one timeline. A driver whose IC collapsed in a regime and
    whose attention also rotated is a different story from one that rotated and held.
    """
    wp = weight_panel(run)
    if wp.empty:
        return pd.DataFrame()
    edges = [wp.index.min()] + [pd.Timestamp(b) for b in breaks] + [wp.index.max()]
    cols = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = wp.loc[(wp.index >= lo) & (wp.index < hi)] if hi != edges[-1] \
            else wp.loc[wp.index >= lo]
        cols[f"{pd.Timestamp(lo).year}-{pd.Timestamp(hi).year}"] = w.mean()
    out = pd.DataFrame(cols)
    return out.reindex(out.mean(axis=1).sort_values(ascending=False).index)
