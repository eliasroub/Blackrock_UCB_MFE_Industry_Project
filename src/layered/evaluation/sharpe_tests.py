"""Paired Sharpe-ratio inference — the test the arm comparison stands on.

With ~120 monthly observations the standard error of an annualized Sharpe is ~0.3,
so comparing two arms by eyeballing their Sharpes answers nothing. But the arms are
run on IDENTICAL dates and are highly correlated (they often hold the same
position), and the variance of a *difference* is var1 + var2 - 2·cov — when the
covariance is large the standard error of the difference collapses. The test only
has to price the months where the arms disagree.

Two p-values, deliberately both reported:

  * ``p_asymptotic`` — the Jobson-Korkie (1981) z with Memmel's (2003) variance
    correction: theta = (1/n) * [2(1 - rho) + 0.5*(s1^2 + s2^2) - s1*s2*rho^2],
    z = (s1 - s2) / sqrt(theta), with s_i the per-period Sharpe and rho the return
    correlation. Exact under iid normality; a reference point.
  * ``p_boot`` — a circular block bootstrap of the CENTERED Sharpe difference,
    which keeps its validity when returns are fat-tailed and autocorrelated (they
    are). This is the simple percentile variant of Ledoit & Wolf (2008); the full
    LW test additionally studentizes each bootstrap draw with a HAC estimator. The
    percentile variant is slightly conservative and dramatically simpler — the
    preregistered decision rule reads ``p_boot``.

Block length and draw count are preregistered constants passed by the caller, not
tuned here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else np.nan


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf — scipy is not a dependency of this repo."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sharpe_diff_test(r1: pd.Series, r2: pd.Series, *, block: int = 6,
                     n_boot: int = 2000, seed: int = 0) -> dict:
    """Test H0: Sharpe(r1) == Sharpe(r2) on paired per-period return series.

    Alignment drops any date where either series is NaN — the comparison is only
    defined on months both arms actually traded. Sharpes in the output are
    annualized (√12) for readability; the statistics are computed per-period.
    """
    df = pd.concat({"a": r1, "b": r2}, axis=1).dropna()
    a, b = df["a"].values, df["b"].values
    n = len(df)
    if n < 12:
        return {"sharpe1": np.nan, "sharpe2": np.nan, "diff": np.nan, "z": np.nan,
                "p_asymptotic": np.nan, "p_boot": np.nan, "n": n}

    s1, s2 = _sharpe(a), _sharpe(b)
    rho = float(np.corrcoef(a, b)[0, 1])
    theta = (2.0 * (1.0 - rho) + 0.5 * (s1 ** 2 + s2 ** 2)
             - s1 * s2 * rho ** 2) / n
    z = (s1 - s2) / np.sqrt(theta) if theta > 0 else np.nan
    p_asym = 2.0 * (1.0 - _norm_cdf(abs(z))) if z == z else np.nan

    # Circular block bootstrap, PAIRED: both series are resampled with the same
    # block indices, so the cross-correlation that gives the test its power is
    # preserved in every draw.
    rng = np.random.default_rng(seed)
    d_hat = s1 - s2
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        diffs[k] = _sharpe(a[idx]) - _sharpe(b[idx])
    centered = diffs - d_hat
    p_boot = float((np.abs(centered) >= abs(d_hat)).mean())

    ann = np.sqrt(12.0)
    return {"sharpe1": s1 * ann, "sharpe2": s2 * ann, "diff": (s1 - s2) * ann,
            "z": float(z), "p_asymptotic": float(p_asym), "p_boot": p_boot, "n": n}
