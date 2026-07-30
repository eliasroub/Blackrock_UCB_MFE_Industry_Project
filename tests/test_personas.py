"""Every persona is a valid analyst — the "add one = a YAML file" promise, checked."""
from __future__ import annotations

import yaml

from src.layered.analysts.llm_analyst import PERSONA_DIR, LLMAnalyst
from src.layered.features import from_persona

DRIVERS = [
    "inflation", "labor_tightness", "curve_slope", "term_premium",
    "balance_sheet", "financial_conditions", "inflation_expectations",
    # Equity drivers ported from macro-llm r7 (data/equity/ EQ_* series).
    "sector_breadth", "vol_regime", "positioning", "risk_appetite",
    # The six international drivers (ea/uk/jp x rates/equity) were removed in
    # the US-only refocus; their vendored data (data/intl/, per-bank corpora)
    # stays for a possible revival.
]


def test_all_personas_build_a_valid_spec():
    for driver in DRIVERS:
        persona = yaml.safe_load((PERSONA_DIR / f"{driver}.yaml").read_text()) or {}
        spec = from_persona(driver, persona)          # raises on dup names / bad level_feature
        names = {d.name for d in spec.definitions}
        assert names, f"{driver}: spec has no features"
        assert spec.level_feature in names, f"{driver}: level_feature not among features"
        assert spec.declared_inputs, f"{driver}: spec declares no raw inputs"


def test_all_personas_construct_an_analyst():
    """from_persona wires the engine + horizon for every driver without data or a key."""
    for driver in DRIVERS:
        analyst = LLMAnalyst.from_persona(driver, llm=None, text_selector=None)
        assert analyst.clock, f"{driver}: no clock resolved"
        assert analyst.horizon_days > 0
        assert analyst.cues, f"{driver}: no text cues"


def test_template_is_not_a_driver():
    """The template must not be mistaken for a persona (its name starts with _)."""
    assert not (PERSONA_DIR / "_TEMPLATE.yaml").stem in DRIVERS
