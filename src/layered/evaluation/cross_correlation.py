"""How independent is the panel? — cross-analyst correlation over a board.

The analyst layer's central structural claim is **input isolation**: each analyst
reads only its own driver's evidence, so seven reports are seven observations
rather than one observation restated seven times. That claim is not self-evidently
true after the fact — a shared text channel, a common prompt scaffold, or a model
that leans on generic macro priors all pull the panel together — and the layer
above it only earns its breadth if the views underneath are actually distinct.

This module measures it: load an arm's board, align every analyst's signed
conviction on one clock, and report the mean absolute pairwise correlation. Lower
is more independent. It is pure post-processing over saved run files — no model
calls — so an arm already on disk can be scored for free.

**Why this exists now.** ``text/whole.py`` and ``text/cue.py`` both cite a
correlation rising "0.221 → 0.339" when the text channel stops being partitioned
by driver, and ``docs/analyst-layer.md`` repeats it. No module in this package
computes that quantity, so the figure cannot currently be reproduced from the
repo, and the A1 arm (cue / whole / none) has no way to re-establish it on a fresh
board. The numbers here are a reproducible replacement, not a claim to match the
historical ones: the method behind those is unrecorded, so treat them as
provisional and re-measure rather than comparing across the gap.

**Two exclusions, both load-bearing.**

* *Degraded* views never enter — they are emitted after a failure and carry no
  judgment. ``runs.load_run`` already drops them from ``signed``.
* *Carried* views are replays, not independent observations (see ``DriverView``:
  counting them as fresh "is what makes a monthly driver look like it produced 52
  opinions a year"). Two analysts both carrying produce a flat stretch that
  correlates perfectly for reasons that have nothing to do with reasoning, so
  carried rows are blanked by default. ``drop_carried=False`` restores them for a
  sensitivity check; it should not be the number that gets reported.

Correlation is rank-based (Spearman) to match ``ICEvaluator``, and because signed
conviction is a coarse bounded scale where a handful of extreme calls would
otherwise dominate a Pearson estimate. Pass ``method="pearson"`` to compare.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from src.layered.evaluation.runs import discover_runs, load_run


def board_matrix(paths: Iterable[str], *, drop_carried: bool = True) -> pd.DataFrame:
    """A ``(date × driver)`` frame of signed convictions, one column per run.

    Outer-joined on date rather than intersected: analysts run on their own release
    clocks, and forcing a common index here would silently discard every meeting
    where one driver had nothing to say. Pair overlap is handled per pair in
    :func:`cross_correlation`, where it can be reported instead of hidden.

    Raises on a duplicate driver — two runs of the same driver in one board is a
    mixed-arm board, and averaging a driver against itself would drag the mean
    toward 1.0 for a reason that is not about the panel.
    """
    cols: dict[str, pd.Series] = {}
    for path in paths:
        run = load_run(path)
        s = run.signed                                  # degraded already dropped
        if drop_carried:
            carried = run.views["carried"].reindex(s.index).fillna(False)
            s = s.mask(carried.astype(bool))
        driver = run.driver or os.path.basename(path)
        if driver in cols:
            raise ValueError(
                f"duplicate driver {driver!r} in board (second file: {path}). "
                f"A board is one run per driver; two arms mixed together would "
                f"correlate a driver against itself.")
        cols[driver] = s.rename(driver)
    if not cols:
        raise ValueError("no runs given")
    return pd.DataFrame(cols).sort_index()


@dataclass(frozen=True)
class CrossCorrelation:
    """The panel's pairwise correlation structure for one arm."""

    arm: str
    matrix: pd.DataFrame        # driver × driver, NaN where under-observed
    counts: pd.DataFrame        # driver × driver pairwise overlap n
    pairs: pd.DataFrame         # long form: a, b, r, abs_r, n — scored pairs only
    mean_abs: float             # THE number: mean |r| over scored pairs
    n_drivers: int
    n_pairs_scored: int
    n_pairs_dropped: int        # too little overlap, or an analyst with no variance
    dropped: tuple[str, ...]    # human-readable reasons, for the run log

    def summary(self) -> dict:
        return {"arm": self.arm, "mean_abs_corr": self.mean_abs,
                "n_drivers": self.n_drivers, "pairs_scored": self.n_pairs_scored,
                "pairs_dropped": self.n_pairs_dropped}


def cross_correlation(matrix: pd.DataFrame, *, arm: str = "", min_obs: int = 12,
                      method: str = "spearman") -> CrossCorrelation:
    """Pairwise correlation across the board's columns, plus the headline mean |r|.

    ``min_obs`` is the minimum number of dates two analysts must both have a
    non-blank view on before their correlation is scored. Under-observed pairs
    become NaN and are *counted*, never quietly treated as zero — a dropped pair
    and an uncorrelated pair say opposite things about independence, and averaging
    the first as if it were the second is how a thin board comes to look like a
    well-separated one.

    An analyst with no variance over the overlap (every call the same) also drops:
    its correlation is undefined, not zero.
    """
    corr = matrix.corr(method=method, min_periods=min_obs)
    counts = matrix.notna().astype(float).T.dot(matrix.notna().astype(float))
    counts = counts.astype(int)

    drivers = list(matrix.columns)
    rows, dropped = [], []
    for i, a in enumerate(drivers):
        for b in drivers[i + 1:]:
            n = int(counts.loc[a, b])
            r = corr.loc[a, b]
            if n < min_obs:
                dropped.append(f"{a}~{b}: only {n} shared meetings (min_obs={min_obs})")
                continue
            if not np.isfinite(r):
                both = matrix[[a, b]].dropna()
                flat = [c for c in (a, b) if both[c].nunique() < 2]
                dropped.append(f"{a}~{b}: undefined correlation"
                               + (f" — no variance in {', '.join(flat)}" if flat else ""))
                continue
            rows.append({"a": a, "b": b, "r": float(r), "abs_r": abs(float(r)), "n": n})

    pairs = pd.DataFrame(rows, columns=["a", "b", "r", "abs_r", "n"])
    if not pairs.empty:
        pairs = pairs.sort_values("abs_r", ascending=False).reset_index(drop=True)
    total_pairs = len(drivers) * (len(drivers) - 1) // 2
    mean_abs = float(pairs["abs_r"].mean()) if not pairs.empty else float("nan")

    return CrossCorrelation(
        arm=arm, matrix=corr, counts=counts, pairs=pairs, mean_abs=mean_abs,
        n_drivers=len(drivers), n_pairs_scored=len(pairs),
        n_pairs_dropped=total_pairs - len(pairs), dropped=tuple(dropped),
    )


def score_board(paths: Iterable[str], *, arm: str = "", drop_carried: bool = True,
                min_obs: int = 12, method: str = "spearman") -> CrossCorrelation:
    """Load a board and score it in one call — the common entry point."""
    return cross_correlation(board_matrix(paths, drop_carried=drop_carried),
                             arm=arm, min_obs=min_obs, method=method)


def discover_arm(reports_dir: str = "reports/ab", suffix: str = "_on") -> list[str]:
    """The run files of one arm, by the board's ``{driver}{suffix}.jsonl`` convention
    (``_on`` = cue, ``_whole``, ``_none``)."""
    return discover_runs(reports_dir, f"*{suffix}.jsonl")


def compare_arms(arms: Mapping[str, Iterable[str]], *, drop_carried: bool = True,
                 min_obs: int = 12, method: str = "spearman",
                 common_drivers: bool = True) -> tuple[pd.DataFrame, dict[str, CrossCorrelation]]:
    """Score several arms head to head — the A1 table (cue vs whole vs none).

    With ``common_drivers`` (the default) every arm is restricted to the drivers
    present in *all* of them before scoring. An arm that ran one extra analyst
    would otherwise be compared on a different panel, and panel composition moves
    mean |r| on its own — the comparison has to vary the text channel and nothing
    else.

    Returns the summary table and the per-arm results, so a caller can drill into
    which pair moved.
    """
    matrices = {name: board_matrix(paths, drop_carried=drop_carried)
                for name, paths in arms.items()}
    if common_drivers and matrices:
        shared = set.intersection(*(set(m.columns) for m in matrices.values()))
        if not shared:
            raise ValueError("arms share no drivers; nothing comparable to score")
        order = [d for d in next(iter(matrices.values())).columns if d in shared]
        matrices = {name: m[order] for name, m in matrices.items()}

    results = {name: cross_correlation(m, arm=name, min_obs=min_obs, method=method)
               for name, m in matrices.items()}
    table = pd.DataFrame([r.summary() for r in results.values()]).set_index("arm")
    return table, results
