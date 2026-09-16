"""
Simulation Scenario Presets
==========================

Named presets that reshape the synthetic data generator to model situations
real-world data can't hand you on demand — a flu surge, a mid-year outbreak
spike, a mild season, or a higher-uninsured population. Each is a small set of
deterministic parameter overrides on the generator, so a whole "what-if" run is
one flag (``--scenario flu_surge``) rather than manual parameter edits.

The ``baseline`` preset reproduces the generator's default behavior exactly, so
existing runs are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Scenario:
    """A named set of generator overrides describing a what-if situation."""

    name: str
    description: str
    # Multiplier on flu-season daily intensity (baseline default: 0.35).
    flu_intensity: float = 0.35
    # Weekend dip factor (baseline default: 0.15).
    weekend_factor: float = 0.15
    # Fraction of uninsured/self-pay patients (baseline default: 0.18).
    uninsured_rate: float = 0.18
    # Additive shift to each visit's high-acuity probability (baseline: 0.0).
    acuity_shift: float = 0.0
    # Optional demand spike: (start_day, end_day, demand_multiplier) over the
    # year, e.g. (180, 210, 2.5) triples arrivals for a ~month mid-year.
    outbreak: Optional[Tuple[int, int, float]] = None


# The preset registry. ``baseline`` must keep the generator's original defaults.
SCENARIOS: Dict[str, Scenario] = {
    "baseline": Scenario(
        name="baseline",
        description="Default demand patterns (unchanged generator behavior).",
    ),
    "flu_surge": Scenario(
        name="flu_surge",
        description="Severe flu season: strong winter seasonality and higher acuity.",
        flu_intensity=0.85,
        acuity_shift=0.08,
    ),
    "outbreak_spike": Scenario(
        name="outbreak_spike",
        description="A sharp ~month-long mid-year outbreak spike with higher acuity.",
        outbreak=(180, 210, 2.5),
        acuity_shift=0.10,
    ),
    "mild_winter": Scenario(
        name="mild_winter",
        description="Weak seasonality: a quiet winter with little flu-driven demand.",
        flu_intensity=0.10,
    ),
    "high_uninsured": Scenario(
        name="high_uninsured",
        description="Higher uninsured share, stressing the insurance/acuity signal.",
        uninsured_rate=0.35,
    ),
}


def get_scenario(name: str) -> Scenario:
    """Resolve a scenario by name (case-insensitive). Defaults handled by caller."""
    key = (name or "baseline").strip().lower()
    if key not in SCENARIOS:
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {', '.join(sorted(SCENARIOS))}"
        )
    return SCENARIOS[key]


def list_scenarios() -> List[str]:
    """Return the available scenario names."""
    return sorted(SCENARIOS)
