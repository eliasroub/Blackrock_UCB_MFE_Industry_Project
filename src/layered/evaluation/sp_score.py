"""Returns-space scoring for the equities pod — S&P positions from driver views.

The rates pods are graded in yield space (``trade_pnl.yield_pnl``); the equities pod
has no comparable grader because the repo carries no equity price series. This module
closes that gap for the PM experiment: it maps any per-driver signed-conviction frame
to an S&P position through ONE deterministic, preregistered rule, and scores the
position series against monthly SPTR excess returns.

The map is deliberately mechanical — the polarity-oriented mean of whatever drivers
are present, clipped to [-1, 1] — and is applied identically to every arm (LLM PMs,
the mechanical PM, the raw-data PM) and to the analyst-layer baselines. That is what
makes the arm comparison a comparison of *views* rather than of sizing conventions:
an arm can only win here by orienting its convictions better, not by discovering a
cleverer position rule. The PM's own sized SPY trade is scored separately (see
``run_sp_score``) as the secondary "does PM sizing add anything?" question.

SPTR is the team's declared source of truth for any joint S&P number
(``../berkeley-mfe-blackrock-2026/total_assets_weekly.csv``, read-only). Weekly
Friday prints are resampled to month-end with ``.last()``; the ME label can sit 1-4
days after the final Friday print, which is accepted and stated rather than
interpolated away.

Timing is the whole game and lives in exactly one line: ``pos.shift(1) * exret`` in
:func:`score_positions`. A position formed at month-end t earns month t+1's return —
the shift IS the no-look-ahead guarantee, and the test suite constructs a series
where forgetting it flips the sign of the Sharpe.
"""
from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

TRADING_MONTHS = 12


def load_sptr(path: str, column: str = "SPTR Index") -> pd.Series:
    """The SPTR level series from the shared weekly file. Read-only, sibling repo."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["Dates"], format="%m/%d/%Y")
    out = pd.Series(df[column].astype(float).values, index=idx, name=column)
    return out.sort_index()


def load_riskfree(path: str = "data/fred/DGS1MO.csv") -> pd.Series:
    """DGS1MO as an annualized percent series, FRED CSV layout."""
    df = pd.read_csv(path)
    idx = pd.to_datetime(df["observation_date"])
    return pd.Series(pd.to_numeric(df["DGS1MO"], errors="coerce").values,
                     index=idx, name="DGS1MO").sort_index().dropna()


def monthly_excess_returns(sptr: pd.Series, rf: Optional[pd.Series] = None) -> pd.Series:
    """Month-end simple SPTR returns, in excess of the 1-month bill.

    The risk-free leg for the month labelled t is DGS1MO observed at the PREVIOUS
    month-end (known when the position is struck, earned over the month) divided by
    100 and by 12. A long position is financed at that rate and a short earns it —
    the standard symmetric-financing convention, so excess return is position-sign
    agnostic.
    """
    level = sptr.resample("ME").last()
    ret = level.pct_change()
    if rf is None:
        return ret.dropna()
    rf_me = (rf.resample("ME").last() / 100.0 / TRADING_MONTHS).shift(1)
    out = (ret - rf_me.reindex(ret.index)).dropna()
    out.name = "sptr_excess"
    return out


def positions_from_convictions(frame: pd.DataFrame,
                               polarity: Mapping[str, float]) -> pd.Series:
    """The preregistered map: polarity-oriented mean over available drivers.

    NaN drivers are simply absent from that meeting's mean — a panel with one view
    still yields a position, sized by that one view. A meeting where NO driver has a
    view maps to NaN (no position), never to 0.0: an all-absent panel is silence,
    and scoring silence as a deliberate flat would fabricate a call, the same rule
    the PM layer applies to degraded meetings.
    """
    oriented = pd.DataFrame({d: frame[d] * float(polarity.get(d, 1.0))
                             for d in frame.columns}, index=frame.index)
    return oriented.mean(axis=1, skipna=True).clip(-1.0, 1.0)


def score_positions(pos: pd.Series, exret: pd.Series) -> dict:
    """Grade a month-end position series against next-month excess returns.

    ``strat_t = pos_{t-1} * exret_t``. Months where the position is NaN are excluded
    from every statistic (no position, no bet — not a zero return that flatters the
    volatility denominator). Hit rate counts only months with a nonzero position.
    """
    pos = pos.dropna()
    strat = (pos.shift(1) * exret).dropna()
    if strat.empty:
        return {"n": 0, "mean_monthly": np.nan, "ann_sharpe": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "max_dd": np.nan}
    mu, sd = strat.mean(), strat.std(ddof=1)
    n = len(strat)
    sharpe_m = mu / sd if sd > 0 else np.nan
    active = strat[pos.shift(1).reindex(strat.index).fillna(0.0) != 0.0]
    cum = (1.0 + strat).cumprod()
    dd = (cum / cum.cummax() - 1.0).min()
    return {
        "n": n,
        "mean_monthly": mu,
        "ann_sharpe": sharpe_m * np.sqrt(TRADING_MONTHS) if sd > 0 else np.nan,
        "t_stat": sharpe_m * np.sqrt(n) if sd > 0 else np.nan,
        "hit_rate": float((active > 0).mean()) if len(active) else np.nan,
        "max_dd": float(dd),
    }


def ridge_baseline(X: pd.DataFrame, y: pd.Series, alpha: float = 1.0,
                   warmup: int = 36) -> pd.Series:
    """The learned-weights null: expanding walk-forward ridge, no LLM.

    At each month t the model is refit on every fully REALIZED pair — features at s,
    excess return at s+1, with s+1 <= t — then reads today's features to size the
    position carried into t+1. Nothing at t ever sees y_{t+1}: the target vector the
    fit uses stops one month short of the prediction date by construction.

    Closed-form numpy ridge (no sklearn dependency): features z-scored on the
    training window, intercept = training-mean of y, position = fitted forecast
    scaled by the training std of y and clipped to [-1, 1] — so a forecast one
    target-sigma high is a full-size position, mirroring the conviction scale of
    the other arms. Rows with any missing feature are skipped in both fit and
    predict. Before ``warmup`` realized pairs exist the position is NaN — an
    unwarmed model abstains rather than guessing.

    ``alpha`` and ``warmup`` are preregistered constants; changing them after seeing
    results is re-tuning, which the ledger forbids.
    """
    X = X.sort_index()
    y = y.sort_index()
    out = pd.Series(np.nan, index=X.index)
    dates = list(X.index)
    for i, t in enumerate(dates):
        x_t = X.loc[t]
        if x_t.isna().any():
            continue
        # Realized pairs: feature date s, target y at the NEXT feature date. Using
        # positional pairing over X's own index keeps the pairing exact when a month
        # is missing rather than silently pairing across a gap.
        feats, targs = [], []
        for j in range(i):
            s, s1 = dates[j], dates[j + 1]
            if s1 > t or s1 not in y.index:
                continue
            x_s = X.loc[s]
            if x_s.isna().any() or pd.isna(y.loc[s1]):
                continue
            feats.append(x_s.values.astype(float))
            targs.append(float(y.loc[s1]))
        if len(feats) < warmup:
            continue
        F = np.asarray(feats)
        tv = np.asarray(targs)
        mu, sd = F.mean(axis=0), F.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Z = (F - mu) / sd
        ty = tv - tv.mean()
        beta = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ ty)
        z_t = (x_t.values.astype(float) - mu) / sd
        yhat = tv.mean() + float(z_t @ beta)
        sigma = tv.std(ddof=1)
        out.loc[t] = float(np.clip(yhat / sigma, -1.0, 1.0)) if sigma > 0 else 0.0
    return out


def trade_positions(trades: pd.Series, instrument: str = "SPY") -> pd.Series:
    """The PM's own sized position, from its trade block (the SECONDARY metric).

    A meeting with no trade dict is NaN (the PM said nothing about position); a trade
    with empty legs and ``flat`` semantics arrives as legs={} and scores as an actual
    0.0 — the deliberate-flat/silence distinction the trade contract exists to keep.
    """
    out = pd.Series(np.nan, index=trades.index, dtype=float)
    for ts, tr in trades.items():
        if not isinstance(tr, dict):
            continue
        legs = tr.get("legs") or {}
        out.loc[ts] = float(legs.get(instrument, 0.0))
    return out
