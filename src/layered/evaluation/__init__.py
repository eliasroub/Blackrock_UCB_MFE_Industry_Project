"""Evaluation — scoring any signal against what the driver actually did.

Deliberately separate from the analyst and from the legacy ``diagnostics.py``,
because it has to grade three different kinds of thing across this project's life —
raw features now, analyst views next, benchmark rules later — and none of them
should have to know about the others. Everything here consumes a signal indexed by
release date, so a feature, a rule, and an LLM's signed conviction are scored by
identical code.

    panel.py     replay a feature spec across history into a (date × feature) matrix
    ic.py        rank IC against the driver's move over the next N releases
    runs.py      load a saved analyst run into scoreable signed/level series
    pm_runs.py   the same for a PM run — a separate loader, because ArbitratedView
                 carries N convictions and none of DriverView's scalar header
    pm_bench.py  the PM against its own analysts, on one shared clock
    trade_pnl.py the PM's *trade* — instrument weights against yield moves, which is a
                 different space from everything above and so a separate module

The PM loaders are named distinctly rather than overloading ``load_run``/``Run``: the
analyst notebooks import those by name, and a shadowed loader that silently accepted
the wrong file shape is the kind of failure this layer is meant to make impossible.

``pm_bench`` is deliberately NOT re-exported here. It depends on ``layered.pm``, which
in turn imports ``runs.view_from`` from this package — importing it at package level
would close that loop into a circular import. It also points the wrong way: evaluation
sits *below* the PM layer, so its package init should not drag the PM layer in. Import
it by module path::

    from src.layered.evaluation.pm_bench import benchmark
"""
from src.layered.evaluation.ic import ICEvaluator, ICResult, required_ic
from src.layered.evaluation.panel import FeaturePanel, release_dates
from src.layered.evaluation.pm_runs import PMRun, discover_pm_runs, load_pm_run
from src.layered.evaluation.runs import Run, discover_runs, load_run, view_from
# Safe to re-export where ``pm_bench`` is not: this module takes the pod's ``trade:``
# block as a plain dict and never imports ``layered.pm``, so it opens no import cycle.
from src.layered.evaluation.trade_pnl import (forward_yield_change, load_trades,
                                              score_trades, trade_validity, yield_pnl)
# Also PM-free: it reads run files through ``runs``/``pm_runs`` and scores with ``ic``,
# never touching ``layered.pm``. (``disagreement_signal`` is the opposite — it reuses
# ``pm_bench``'s clock rebuild, so it stays import-by-path only.)
from src.layered.evaluation.perturbation_bench import (direction_response, ic_dispersion,
                                                       ic_stability, scramble_response)
# PM-free as well: reads boards through ``runs`` and correlates with pandas, never
# touching ``layered.pm``. Measures how independent the panel actually is (the
# input-isolation claim), the quantity the cue/whole/none arm turns on.
from src.layered.evaluation.cross_correlation import (CrossCorrelation, board_matrix,
                                                      compare_arms, cross_correlation,
                                                      discover_arm, score_board)

__all__ = ["ICEvaluator", "ICResult", "FeaturePanel", "release_dates", "required_ic",
           "Run", "load_run", "discover_runs", "view_from",
           "PMRun", "load_pm_run", "discover_pm_runs",
           "load_trades", "yield_pnl", "forward_yield_change", "score_trades",
           "trade_validity",
           "direction_response", "ic_stability", "ic_dispersion", "scramble_response",
           "CrossCorrelation", "board_matrix", "cross_correlation", "score_board",
           "discover_arm", "compare_arms"]
