"""IC diagnostics beyond the single number.

``ICEvaluator.evaluate`` answers "did signed conviction order the next release
correctly, over the whole sample". That is the right headline and it is one number
over ~125 observations, which hides most of what you want to know about an analyst.
Everything here decomposes that number along an axis the point estimate collapses:

  time        rolling_ic          is the skill stable, or one regime?
  horizon     horizon_decay       is it a one-release blip or persistent?
  size        conviction_buckets  does the conviction ladder order anything?
              tercile_spread      a rank-IC-free read, robust to one outlier
  magnitude   magnitude_skill     does it know when a BIG move is coming? (separate skill)
  regime      regime_ic           pre-COVID / COVID / hiking-and-after
  precision   bootstrap_ic_ci     an honest interval instead of a normal approximation
  cases       ic_outliers         which dates broke the fit, so they can be read
  arms        arm_disagreement    where two arms disagree, which one is right?

All of it is offline over saved runs, all $0, and none of it may inform a prompt —
the same honesty rule the feature ICs live under (``docs/analyst-layer.md`` §6).

Two conventions inherited from ``ic.py`` and not restated in every function:
signals align to outcomes **by index label**, and the outcome is the *next-release*
change ``level.shift(-steps) - level``, which is non-overlapping on a release clock
so an ordinary t-statistic is honest.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

__all__ = [
    "align", "rank_ic", "rolling_ic", "bootstrap_ic_ci", "horizon_decay",
    "conviction_buckets", "tercile_spread", "magnitude_skill", "regime_ic",
    "ic_outliers", "arm_disagreement",
    # the t-statistic itself
    "signal_autocorr", "effective_n", "permutation_ic_pvalue", "jackknife_t",
    "rolling_t", "ic_for_t", "IR1_BAR",
]

# The project's own bar: IR = IC x sqrt(breadth), breadth ~ 12 releases/yr, so an
# information ratio of 1.0 needs this much IC. Quoted in plots as a reference line
# because 0.05 is a fine cross-sectional IC and a useless single-driver one.
IR1_BAR = 0.289


def align(signed: pd.Series, level: pd.Series, steps: int = 1) -> pd.DataFrame:
    """``(s, y)`` on shared dates: signal, and the next-``steps``-release change."""
    lvl = pd.Series(level).dropna().sort_index()
    y = (lvl.shift(-steps) - lvl).dropna()
    return pd.concat([pd.Series(signed).rename("s"), y.rename("y")],
                     axis=1).dropna()


def rank_ic(s: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    """``(n, Spearman IC, t)``. Pearson-on-ranks, as ``ICEvaluator`` does it."""
    a = pd.concat([s.rename("s"), y.rename("y")], axis=1).dropna()
    n = len(a)
    if n < 3 or a["s"].nunique() < 2 or a["y"].nunique() < 2:
        return n, float("nan"), float("nan")
    ic = float(a["s"].rank().corr(a["y"].rank()))
    t = ic * math.sqrt((n - 2) / (1 - ic * ic)) if abs(ic) < 1 else float("nan")
    return n, ic, t


# ── time ────────────────────────────────────────────────────────────────────

def rolling_ic(signed, level, window: int = 24, steps: int = 1) -> pd.Series:
    """Rank IC over a trailing ``window`` of observations, indexed by window end.

    24 is the project's stated default — two years on a monthly clock. Short enough
    to show a regime, long enough that the estimate is not noise: at n=24 the IC
    needed for t=2 is ~0.41, so read the *shape*, not whether a point clears a bar.
    """
    a = align(signed, level, steps)
    if len(a) < window:
        return pd.Series(dtype=float)
    out = {}
    for i in range(window, len(a) + 1):
        w = a.iloc[i - window:i]
        _, ic, _ = rank_ic(w["s"], w["y"])
        out[a.index[i - 1]] = ic
    return pd.Series(out).sort_index()


# ── precision ───────────────────────────────────────────────────────────────

def bootstrap_ic_ci(signed, level, steps: int = 1, n_boot: int = 10_000,
                    alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile-bootstrap CI for the IC, plus the share of resamples below zero.

    Why bother: ``ic.py`` reports ``p_approx`` from ``math.erfc``, a normal
    approximation, and says so in the column name — scipy is not a dependency. At
    n~125 that is fine for a rough read but it assumes a symmetric sampling
    distribution the rank IC does not have near +/-1. Resampling observation pairs
    makes no such assumption and gives an interval you can put on a slide.

    ``p_sign`` is the fraction of resamples on the opposite side of zero from the
    point estimate — a bootstrap analogue of a one-sided p-value, and the number to
    quote when the normal-approx t sits near 2.
    """
    a = align(signed, level, steps)
    n, ic, t = rank_ic(a["s"], a["y"])
    if n < 8 or ic != ic:
        return {"n": n, "ic": ic, "t": t, "lo": float("nan"),
                "hi": float("nan"), "p_sign": float("nan")}
    rng = np.random.default_rng(seed)
    sv, yv = a["s"].to_numpy(), a["y"].to_numpy()
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = pd.Series(sv[idx]).rank().corr(pd.Series(yv[idx]).rank())
    boots = boots[~np.isnan(boots)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_sign = float((boots <= 0).mean() if ic > 0 else (boots >= 0).mean())
    return {"n": n, "ic": ic, "t": t, "lo": float(lo), "hi": float(hi),
            "p_sign": p_sign}


# ── horizon ─────────────────────────────────────────────────────────────────

def horizon_decay(signed, level, steps_list=(1, 2, 3)) -> pd.DataFrame:
    """IC at 1, 2, 3 releases ahead — is the call early, persistent, or a blip?

    The analyst is *asked* for one release ahead, so steps>1 is not grading it on
    its mandate. It is diagnosis: a signal that also works at 2-3 releases is
    picking up a slow-moving state, while one that only works at 1 and dies is
    either genuinely timely or fitting the next print's noise. Overlapping windows
    at steps>1 inflate the t, so read the IC and ignore the t there.
    """
    rows = []
    for k in steps_list:
        a = align(signed, level, k)
        n, ic, t = rank_ic(a["s"], a["y"])
        rows.append({"steps": k, "n": n, "ic": ic,
                     # only steps=1 is non-overlapping, so only its t is honest
                     "t": t if k == 1 else float("nan")})
    return pd.DataFrame(rows)


# ── size / calibration ──────────────────────────────────────────────────────

def conviction_buckets(signed, level, q: int = 4, steps: int = 1) -> pd.DataFrame:
    """Mean signed outcome per |conviction| quantile — does the ladder order?

    ``ICEvaluator.calibration_split`` asks whether conviction adds *anything* over
    direction alone, as one number. This asks the sharper question: is the
    relationship **monotone**? An analyst whose top-conviction calls do no better
    than its middling ones is not calibrated even if its overall IC is fine, and
    that is a different repair from "the direction is wrong".

    ``mean_aligned`` is ``sign(signed) * y``: positive means the call was right,
    scaled by how far the driver moved. Deliberately not a rank measure, because
    the question is whether SIZE is informative.
    """
    a = align(signed, level, steps)
    a = a[a["s"] != 0]
    if len(a) < q * 3:
        return pd.DataFrame()
    a = a.assign(mag=a["s"].abs(),
                 aligned=np.sign(a["s"]) * a["y"])
    try:
        a["bucket"] = pd.qcut(a["mag"], q, labels=[f"Q{i+1}" for i in range(q)],
                              duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = a.groupby("bucket", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "mean_abs_conviction": g["mag"].mean().round(3),
        "mean_aligned": g["aligned"].mean(),
        "hit_rate": g["aligned"].apply(lambda x: float((x > 0).mean())).round(3),
    }).reset_index()


def tercile_spread(signed, level, steps: int = 1) -> dict:
    """Top-tercile minus bottom-tercile mean outcome, with a t on the difference.

    A rank IC uses every pair and is therefore sensitive to the middle of the
    distribution, where an analyst has no view anyway. This reads only the ends —
    what a long/short book would actually hold — and is robust to one wild
    observation in a way a correlation is not. When it disagrees with the IC, the
    IC is being carried by the middle.
    """
    a = align(signed, level, steps)
    if len(a) < 15:
        return {"n": len(a), "spread": float("nan"), "t": float("nan")}
    lo_c, hi_c = a["s"].quantile([1 / 3, 2 / 3])
    lo, hi = a.loc[a["s"] <= lo_c, "y"], a.loc[a["s"] >= hi_c, "y"]
    if len(lo) < 3 or len(hi) < 3:
        return {"n": len(a), "spread": float("nan"), "t": float("nan")}
    spread = float(hi.mean() - lo.mean())
    se = math.sqrt(hi.var(ddof=1) / len(hi) + lo.var(ddof=1) / len(lo))
    return {"n": len(a), "n_hi": len(hi), "n_lo": len(lo), "spread": spread,
            "t": spread / se if se > 0 else float("nan")}


# ── magnitude, a separate skill ──────────────────────────────────────────────

def magnitude_skill(signed, level, steps: int = 1) -> dict:
    """Rank IC of |conviction| against |realized move| — does it know when it matters?

    Nothing in the repo tests this, and it is orthogonal to direction: an analyst
    can be a coin flip on sign while still knowing which months are eventful, and
    that is a genuinely useful signal for sizing (and for a PM's trust discount).
    Reported alongside directional IC so the two are never conflated.
    """
    a = align(signed, level, steps)
    n, ic, t = rank_ic(a["s"].abs(), a["y"].abs())
    return {"n": n, "ic_magnitude": ic, "t": t}


# ── regime ──────────────────────────────────────────────────────────────────

def regime_ic(signed, level, breaks=("2020-01-01", "2022-07-01"),
              steps: int = 1) -> pd.DataFrame:
    """IC per subperiod. Defaults split pre-COVID / COVID / hiking-and-after.

    The whole-sample IC of an analyst that worked only in 2020-22 looks identical to
    one that works everywhere. 2022-07 is chosen as the second break because the
    hiking cycle was well underway by then, not fitted to any outcome.
    """
    a = align(signed, level, steps)
    edges = [a.index.min()] + [pd.Timestamp(b) for b in breaks] + [a.index.max()]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = a.loc[(a.index >= lo) & (a.index < hi)] if hi != edges[-1] \
            else a.loc[a.index >= lo]
        n, ic, t = rank_ic(w["s"], w["y"])
        rows.append({"from": str(pd.Timestamp(lo).date()),
                     "to": str(pd.Timestamp(hi).date()),
                     "n": n, "ic": ic, "t": t})
    return pd.DataFrame(rows)


# ── the special cases ───────────────────────────────────────────────────────

def ic_outliers(signed, level, steps: int = 1, z: float = 2.5) -> pd.DataFrame:
    """Dates whose |studentized residual| exceeds ``z`` on the OLS fit of y on s.

    The standard read, and deliberately so: fit y = a + b*s, studentize the
    residuals by the leave-one-out sigma, and flag |t_i| > 2.5. These are the
    meetings where the analyst's conviction and the driver's actual move disagreed
    most violently, which is exactly the list you want to read the reports for.

    Returns the residual, the leverage, and Cook's distance — leverage separates
    "an extreme call that was wrong" (high leverage) from "an ordinary call that a
    freak move blindsided" (high residual, low leverage). They need different
    explanations.
    """
    a = align(signed, level, steps)
    if len(a) < 10:
        return pd.DataFrame()
    x, y = a["s"].to_numpy(float), a["y"].to_numpy(float)
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    resid = y - fit
    h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)   # leverage
    dof = n - 2
    sse = float(resid @ resid)
    # leave-one-out sigma, so a point cannot mask itself
    s2_i = (sse - resid ** 2 / np.maximum(1e-12, 1 - h)) / max(1, dof - 1)
    denom = np.sqrt(np.maximum(1e-12, s2_i * (1 - h)))
    tstud = resid / denom
    cooks = (resid ** 2 / (2 * max(1e-12, sse / dof))) * (h / np.maximum(1e-12, (1 - h) ** 2))
    out = pd.DataFrame({
        "signed": x, "outcome": y, "fitted": fit, "resid": resid,
        "studentized": tstud, "leverage": h, "cooks_d": cooks,
    }, index=a.index)
    out["is_outlier"] = out["studentized"].abs() > z
    return out.sort_values("studentized", key=abs, ascending=False)


# ── arms ────────────────────────────────────────────────────────────────────

def arm_disagreement(signed_by_arm: dict, level, a: str, b: str,
                     steps: int = 1) -> dict:
    """On the dates two arms called opposite directions, which one was right?

    This is a sharper test of "does the text add information" than comparing two
    ICs. Two arms can have near-identical ICs while disagreeing on a third of the
    sample; the IC difference averages that away, and this does not. Restricted to
    strict sign disagreements with a non-zero realized move, so there is always a
    right answer.
    """
    ya = align(signed_by_arm[a], level, steps)
    yb = align(signed_by_arm[b], level, steps)
    j = ya.join(yb["s"].rename("s_b"), how="inner").rename(columns={"s": "s_a"})
    j = j[(np.sign(j["s_a"]) != np.sign(j["s_b"])) & (j["s_a"] != 0)
          & (j["s_b"] != 0) & (j["y"] != 0)]
    if j.empty:
        return {"n_disagree": 0, f"{a}_right": float("nan"),
                f"{b}_right": float("nan"), "share_of_sample": 0.0}
    a_right = float((np.sign(j["s_a"]) == np.sign(j["y"])).mean())
    return {
        "n_disagree": len(j),
        "share_of_sample": round(len(j) / max(1, len(ya)), 3),
        f"{a}_right": round(a_right, 3),
        f"{b}_right": round(1 - a_right, 3),
        "dates": list(j.index),
    }

# ── the t-statistic itself ──────────────────────────────────────────────────
# `ic.py` computes  t = ic * sqrt((n-2)/(1-ic^2))  and a p from math.erfc.
# That is the classical test for a Pearson correlation, applied to ranks. It
# carries five assumptions, and this section measures each rather than trusting it:
#
#   1  independence of observations.   The release clock is the defence, and it
#      works on the OUTCOME side (non-overlapping windows). It does nothing about
#      the SIGNAL side: a memory-armed analyst repeats itself, so consecutive
#      convictions are correlated and the effective sample is smaller than n.
#      -> effective_n, and the adjusted t beside it.
#   2  bivariate normality, for the transform to be an exact t.  Ranks are
#      uniform, not normal, so this is an approximation at any n.
#      -> permutation_ic_pvalue makes no distributional assumption at all.
#   3  the normal approximation in the p-value.  Fine at n~125, wrong at n~17,
#      which is exactly the size of the post-2024 slice.
#      -> quote the bootstrap interval or the permutation p there, never p_approx.
#   4  no single dominant observation.  A correlation and its t inherit any one
#      pair's leverage.
#      -> jackknife_t reports how far the t moves when the worst offenders go.
#   5  a stable relationship.  One number over 125 months assumes the thing being
#      measured did not change.
#      -> rolling_ic, and rolling_t only with the caveat documented there.


def signal_autocorr(signed, lag: int = 1) -> float:
    """Lag-``lag`` autocorrelation of the signal. The input to ``effective_n``."""
    s = pd.Series(signed).dropna().sort_index()
    return float(s.autocorr(lag)) if len(s) > lag + 2 else float("nan")


def effective_n(signed, level, steps: int = 1) -> dict:
    """Sample size adjusted for serial correlation, and the t that follows from it.

    The release clock makes the *outcomes* non-overlapping, which is what licenses
    an ordinary t. It does not make the *signal* independent — an analyst holding a
    view across several releases (and the memory arm encourages exactly that)
    produces autocorrelated convictions, and two correlated observations carry less
    than two observations' worth of information.

    Uses the standard first-order correction, n_eff = n * (1-r1*r2)/(1+r1*r2) with
    r1, r2 the lag-1 autocorrelations of signal and outcome (Bartlett / Quenouille).
    It is a first-order fix, not a Newey-West panacea: if |r1*r2| is small the t
    barely moves and the original is fine, which is itself the useful finding.
    """
    a = align(signed, level, steps)
    n, ic, t = rank_ic(a["s"], a["y"])
    r1 = float(a["s"].autocorr(1)) if len(a) > 4 else float("nan")
    r2 = float(a["y"].autocorr(1)) if len(a) > 4 else float("nan")
    if not (np.isfinite(r1) and np.isfinite(r2)) or not np.isfinite(ic):
        return {"n": n, "r1_signal": r1, "r1_outcome": r2,
                "n_eff": float("nan"), "t_adj": float("nan")}
    rho = r1 * r2
    n_eff = n * (1 - rho) / (1 + rho) if abs(rho) < 1 else float("nan")
    n_eff = max(3.0, min(float(n), n_eff)) if np.isfinite(n_eff) else float("nan")
    t_adj = (ic * math.sqrt((n_eff - 2) / (1 - ic * ic))
             if np.isfinite(n_eff) and abs(ic) < 1 else float("nan"))
    return {"n": n, "r1_signal": round(r1, 3), "r1_outcome": round(r2, 3),
            "n_eff": round(n_eff, 1) if np.isfinite(n_eff) else n_eff,
            "t": t, "t_adj": t_adj}


def permutation_ic_pvalue(signed, level, steps: int = 1, n_perm: int = 10_000,
                          seed: int = 0) -> dict:
    """Two-sided p by shuffling the signal. No distributional assumption at all.

    This is the honest answer to "can I trust that t". The null is "this signal has
    no relationship to the outcome", and shuffling the signal against fixed outcomes
    realises that null exactly, preserving both marginal distributions. If the
    permutation p and the normal-approx p agree, the parametric t was fine; when
    they diverge, the permutation one is right.
    """
    a = align(signed, level, steps)
    n, ic, t = rank_ic(a["s"], a["y"])
    if n < 8 or not np.isfinite(ic):
        return {"n": n, "ic": ic, "t": t, "p_perm": float("nan"),
                "p_approx": float("nan")}
    rng = np.random.default_rng(seed)
    yr = a["y"].rank().to_numpy()
    sr = a["s"].rank().to_numpy()
    yc = yr - yr.mean()
    denom = math.sqrt(float(yc @ yc))
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(sr)
        pc = p - p.mean()
        null[i] = float(pc @ yc) / (math.sqrt(float(pc @ pc)) * denom)
    p_perm = float((np.abs(null) >= abs(ic)).mean())
    return {"n": n, "ic": ic, "t": t,
            "p_perm": p_perm,
            "p_approx": math.erfc(abs(t) / math.sqrt(2.0)) if np.isfinite(t) else float("nan")}


def jackknife_t(signed, level, steps: int = 1, drop: int = 3) -> dict:
    """How far do IC and t move when the most influential observations are dropped?

    Not a significance test — a fragility read. If dropping three of 125 months
    halves the t, the result is one episode wearing a sample's clothes, and that is
    worth knowing before it goes on a slide. Influence is Cook's distance from the
    same OLS fit ``ic_outliers`` uses, so the dropped dates are reportable.
    """
    a = align(signed, level, steps)
    n0, ic0, t0 = rank_ic(a["s"], a["y"])
    o = ic_outliers(signed, level, steps)
    if o.empty or len(a) <= drop + 5:
        return {"ic": ic0, "t": t0, "ic_ex": float("nan"), "t_ex": float("nan"),
                "dropped": []}
    worst = list(o.sort_values("cooks_d", ascending=False).index[:drop])
    keep = a.drop(index=worst, errors="ignore")
    n1, ic1, t1 = rank_ic(keep["s"], keep["y"])
    return {"n": n0, "ic": ic0, "t": t0, "n_ex": n1, "ic_ex": ic1, "t_ex": t1,
            "t_retained": round(t1 / t0, 2) if t0 not in (0, float("nan")) else float("nan"),
            "dropped": [str(pd.Timestamp(d).date()) for d in worst]}


def rolling_t(signed, level, window: int = 24, steps: int = 1) -> pd.Series:
    """Rolling t. Provided, but read the warning.

    A t is a function of both the effect and the window length, so a rolling t over
    a fixed window is just the rolling IC rescaled by a constant — it adds no
    information over ``rolling_ic`` and invites reading a threshold crossing as an
    event. At window=24 the IC needed for t=2 is ~0.41, so most of a real analyst's
    history sits below the line even when the full-sample result is strong.

    Prefer ``rolling_ic`` with a horizontal reference at the window's own t=2 IC —
    same information, no false precision. This exists so that choice is explicit.
    """
    ric = rolling_ic(signed, level, window, steps)
    return ric.apply(lambda ic: ic * math.sqrt((window - 2) / (1 - ic * ic))
                     if np.isfinite(ic) and abs(ic) < 1 else float("nan"))


def ic_for_t(t: float, n: int) -> float:
    """The IC that would produce ``t`` at sample size ``n`` — the reference line."""
    if n <= 2:
        return float("nan")
    return float(t / math.sqrt(n - 2 + t * t))
