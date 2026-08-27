"""
Data Drift Monitoring
=====================

Detects distribution shift between a *reference* dataset (what a model was
trained on) and a *current* dataset (new incoming data), so retraining can be
triggered before silent performance decay.

Two complementary signals per numeric feature:

  * **PSI** (Population Stability Index) - a binned-distribution divergence.
      PSI < 0.1   : no significant shift
      0.1<=PSI<0.2: moderate shift
      PSI >= 0.2  : significant shift
  * **KS** (Kolmogorov-Smirnov) - max gap between the two empirical CDFs, with a
      p-value; p < 0.05 flags a shift.

A feature is considered drifted if PSI >= psi_threshold OR the KS p-value is
below ks_alpha. The monitor can also persist / load a compact reference profile
so runs can be compared over time without keeping raw data around.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Drift outcome for a single feature."""

    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    drifted: bool


@dataclass
class DriftReport:
    """Aggregate drift outcome across features."""

    n_features: int
    n_drifted: int
    drift_share: float
    drifted_features: List[str]
    results: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DriftMonitor:
    """Compute PSI + KS drift between reference and current data."""

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        n_bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.n_bins = n_bins
        self.logger = logger

    # ------------------------------------------------------------------ #
    @staticmethod
    def population_stability_index(
        reference: np.ndarray, current: np.ndarray, n_bins: int = 10
    ) -> float:
        """PSI between two samples using quantile bins of the reference."""
        reference = np.asarray(reference, dtype=float)
        current = np.asarray(current, dtype=float)
        # Quantile edges from the reference; dedupe for near-constant features.
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf

        ref_counts, _ = np.histogram(reference, bins=edges)
        cur_counts, _ = np.histogram(current, bins=edges)
        eps = 1e-6
        ref_pct = ref_counts / max(ref_counts.sum(), 1) + eps
        cur_pct = cur_counts / max(cur_counts.sum(), 1) + eps
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def _feature_drift(
        self, ref: np.ndarray, cur: np.ndarray, name: str
    ) -> DriftResult:
        psi = self.population_stability_index(ref, cur, self.n_bins)
        try:
            ks_stat, ks_p = stats.ks_2samp(ref, cur)
        except ValueError:  # pragma: no cover - empty input guard
            ks_stat, ks_p = 0.0, 1.0
        drifted = (psi >= self.psi_threshold) or (ks_p < self.ks_alpha)
        return DriftResult(
            feature=name,
            psi=round(float(psi), 4),
            ks_statistic=round(float(ks_stat), 4),
            ks_pvalue=round(float(ks_p), 4),
            drifted=bool(drifted),
        )

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftReport:
        """Compare numeric features shared by both frames and summarize drift."""
        if features is None:
            numeric = reference.select_dtypes(include=[np.number]).columns
            features = [c for c in numeric if c in current.columns]

        results: List[DriftResult] = []
        for name in features:
            ref = reference[name].dropna().to_numpy()
            cur = current[name].dropna().to_numpy()
            if len(ref) < 2 or len(cur) < 2:
                continue
            results.append(self._feature_drift(ref, cur, name))

        drifted = [r.feature for r in results if r.drifted]
        n = len(results)
        return DriftReport(
            n_features=n,
            n_drifted=len(drifted),
            drift_share=round(len(drifted) / n, 4) if n else 0.0,
            drifted_features=drifted,
            results=[asdict(r) for r in results],
        )

    # ------------------------------------------------------------------ #
    # Reference-profile persistence                                      #
    # ------------------------------------------------------------------ #
    def build_profile(
        self, df: pd.DataFrame, features: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """Store a compact sample (quantile grid) per numeric feature."""
        if features is None:
            features = list(df.select_dtypes(include=[np.number]).columns)
        grid = np.linspace(0, 1, 101)
        profile: Dict[str, List[float]] = {}
        for name in features:
            col = df[name].dropna().to_numpy()
            if len(col) >= 2:
                profile[name] = [float(v) for v in np.quantile(col, grid)]
        return profile

    def save_profile(self, df: pd.DataFrame, path: str) -> str:
        """Persist a reference profile to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        profile = self.build_profile(df)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile, fh)
        self.logger.info("Saved drift reference profile -> %s", path)
        return path

    def compare_to_profile(self, current: pd.DataFrame, path: str) -> DriftReport:
        """Compare current data against a saved reference profile."""
        with open(path, encoding="utf-8") as fh:
            profile = json.load(fh)
        reference = pd.DataFrame({name: vals for name, vals in profile.items()})
        return self.compare(reference, current, features=list(profile.keys()))
