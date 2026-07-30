"""The analyst layer — a population of isolated single-driver experts.

An analyst is fully described by its persona YAML (``personas/<driver>.yaml``):
the ``features`` block fixes what it may measure, ``text_cues`` fix which policy
language reaches it, and ``mandate`` fixes what it is asked to judge. There is one
implementation, ``LLMAnalyst``; adding an analyst is writing a config file, not a
subclass.
"""
from __future__ import annotations

from src.layered.analysts.build import (
    TEXT_ARMS,
    arm_spec,
    build_analyst,
    build_selector,
    preflight_llm,
    print_run_audit,
)
from src.layered.analysts.carry_forward import CarryForward
from src.layered.analysts.llm_analyst import LLMAnalyst

__all__ = [
    "LLMAnalyst",
    "TEXT_ARMS",
    "arm_spec",
    "CarryForward",
    "build_analyst",
    "build_selector",
    "preflight_llm",
    "print_run_audit",
]
