"""Shared wiring for the PM run scripts — build a board, build a PM.

Deliberately separate from ``analysts/build.py``. The analyst layer must not import
the PM layer: the dependency runs data → analyst → PM, and a build helper that
reached back down would be the first thing to blur it. ``preflight_llm`` is imported
*from* the analyst layer because that direction is fine and the check is identical.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.layered.analysts.build import preflight_llm, print_run_audit  # noqa: F401
from src.layered.pm.board import ViewBoard
from src.layered.pm.llm_pm import LLMPM


def build_pm(pod: str, llm=None, *, pod_dir: Optional[Path] = None,
             max_report_words: Optional[int] = None,
             blind: Optional[str] = None, use_memory: bool = False,
             perturbation=None, disclose: Optional[str] = None) -> LLMPM:
    """An ``LLMPM`` wired from its pod spec.

    ``perturbation`` is an evaluation-only arm (the scramble in
    ``src.layered.perturb.brief``, or a shared string perturbation); ``None`` is the
    shipped path. The run script resolves the ``--perturb`` name and passes it here.

    ``disclose`` is the date-disclosure leak arm (``src.layered.pm.brief``'s
    ``DISCLOSURE_ARMS``); ``None`` is likewise the shipped, date-blind path.
    """
    return LLMPM.from_pod(pod, llm=llm, pod_dir=pod_dir,
                          max_report_words=max_report_words, blind=blind,
                          use_memory=use_memory, perturbation=perturbation,
                          disclose=disclose)


def build_board(pm: LLMPM, directory: str = "reports/ab", suffix: str = "_on",
                *, check_identity: bool = True) -> ViewBoard:
    """The board for a pod — only the drivers it reads, on its own thresholds.

    Restricting to the pod spec rather than taking whatever is on disk means the spec,
    not the contents of a directory, decides who is in the room.

    The board is built from ``pm.reads``, not ``pm.listens_to``: it has to contain every
    driver that will be *rendered* into the brief, which for a pod that reads more of
    the panel than it opines on is the wider set. ``reads is None`` means "the whole
    panel", which is exactly ``from_dir``'s own default of discovering the driver set
    from disk.
    """
    return ViewBoard.from_dir(directory, suffix, drivers=pm.reads,
                              check_identity=check_identity, **pm.board_kwargs)
