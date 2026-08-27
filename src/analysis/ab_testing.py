"""
A/B Testing Framework
=====================

Evaluate a two-arm intervention experiment (control vs treatment) for the kinds
of interventions this project targets — e.g. does a staffing change or a
preventive-care outreach reduce non-urgent ER visits?

Two test types:

  * ``proportion`` — binary outcome per subject (e.g. returned within 30 days).
    Uses a two-proportion z-test and reports lift and relative lift.
  * ``mean`` — continuous outcome per subject (e.g. wait time, cost). Uses
    Welch's two-sample t-test (unequal variances) and Cohen's d effect size.

Each result reports the effect, a p-value, whether it is significant at
``alpha``, and a plain-language decision.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ABResult:
    """Outcome of an A/B comparison."""

    test_type: str
    control_stat: float
    treatment_stat: float
    absolute_effect: float
    relative_effect: float
    effect_size: float  # Cohen's h (proportion) or Cohen's d (mean)
    p_value: float
    significant: bool
    n_control: int
    n_treatment: int
    decision: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _decision(significant: bool, effect: float) -> str:
    if not significant:
        return "No significant difference — do not roll out on this evidence."
    direction = "increase" if effect > 0 else "decrease"
    return f"Significant {direction} in the treatment arm — consider rolling out."


class ABTest:
    """Two-arm experiment analysis."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.logger = logger

    # ------------------------------------------------------------------ #
    def proportion_test(
        self, control: Sequence[int], treatment: Sequence[int]
    ) -> ABResult:
        """Two-proportion z-test for binary (0/1) outcomes."""
        c = np.asarray(control, dtype=float)
        t = np.asarray(treatment, dtype=float)
        n_c, n_t = len(c), len(t)
        p_c, p_t = c.mean(), t.mean()

        # Pooled two-proportion z-test.
        p_pool = (c.sum() + t.sum()) / (n_c + n_t)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
        z = (p_t - p_c) / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        # Cohen's h effect size for proportions.
        h = 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))
        significant = p_value < self.alpha
        return ABResult(
            test_type="proportion",
            control_stat=round(float(p_c), 4),
            treatment_stat=round(float(p_t), 4),
            absolute_effect=round(float(p_t - p_c), 4),
            relative_effect=round(float((p_t - p_c) / p_c), 4) if p_c else float("nan"),
            effect_size=round(float(h), 4),
            p_value=round(p_value, 6),
            significant=significant,
            n_control=n_c,
            n_treatment=n_t,
            decision=_decision(significant, float(p_t - p_c)),
        )

    def mean_test(
        self, control: Sequence[float], treatment: Sequence[float]
    ) -> ABResult:
        """Welch's two-sample t-test for continuous outcomes."""
        c = np.asarray(control, dtype=float)
        t = np.asarray(treatment, dtype=float)
        n_c, n_t = len(c), len(t)
        m_c, m_t = c.mean(), t.mean()

        _, p_value = stats.ttest_ind(t, c, equal_var=False)
        p_value = float(p_value)

        # Cohen's d with a pooled standard deviation.
        pooled_sd = np.sqrt(
            ((n_c - 1) * c.var(ddof=1) + (n_t - 1) * t.var(ddof=1)) / (n_c + n_t - 2)
        )
        d = (m_t - m_c) / pooled_sd if pooled_sd > 0 else 0.0
        significant = p_value < self.alpha
        return ABResult(
            test_type="mean",
            control_stat=round(float(m_c), 4),
            treatment_stat=round(float(m_t), 4),
            absolute_effect=round(float(m_t - m_c), 4),
            relative_effect=round(float((m_t - m_c) / m_c), 4) if m_c else float("nan"),
            effect_size=round(float(d), 4),
            p_value=round(p_value, 6),
            significant=significant,
            n_control=n_c,
            n_treatment=n_t,
            decision=_decision(significant, float(m_t - m_c)),
        )
