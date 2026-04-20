from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(frozen=True)
class PolicyConfig:
    p_clean: float
    k_values: List[int]
    k_probs: List[float]
    type_weights: Dict[str, float]

    # Beta mixture severity sampling
    beta_alpha: float = 2.0
    beta_beta: float = 5.0
    hard_mix_p: float = 0.10
    hard_alpha: float = 5.0
    hard_beta: float = 2.0
    min_sev: float = 0.02
    max_sev: float = 0.98

    # Couplings
    low_light_implies_noise_p: float = 0.70
    rain_implies_fog_p: float = 0.35
    exposure_implies_glare_p: float = 0.25

    # Constraints
    max_optical_in_sample: int = 2


def _normalize_type_probs(issues: List[str], type_weights: Dict[str, float]) -> np.ndarray:
    w = np.array([float(type_weights.get(k, 1.0)) for k in issues], dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    return w / w.sum()


class MixturePolicy:
    def __init__(
        self,
        issues: List[str],
        groups: Optional[Dict[str, List[str]]],
        cfg: PolicyConfig,
        seed: int = 0,
    ):
        self.issues = list(issues)
        self.groups = groups or {}
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        self.type_probs = _normalize_type_probs(self.issues, cfg.type_weights)

        kv = np.array(cfg.k_values, dtype=np.int64)
        kp = np.array(cfg.k_probs, dtype=np.float64)
        self.k_values = kv
        self.k_probs = kp / kp.sum()

        self.optical_set = set(self.groups.get("optical", []))

    def _sample_severity(self) -> float:
        c = self.cfg
        if self.rng.random() < c.hard_mix_p:
            s = self.rng.beta(c.hard_alpha, c.hard_beta)
        else:
            s = self.rng.beta(c.beta_alpha, c.beta_beta)
        return float(np.clip(s, c.min_sev, c.max_sev))

    def _apply_couplings(self, chosen: List[str]) -> List[str]:
        c = self.cfg
        chosen_set = set(chosen)

        if "low_light" in chosen_set and "sensor_noise" not in chosen_set:
            if self.rng.random() < c.low_light_implies_noise_p:
                chosen.append("sensor_noise")
                chosen_set.add("sensor_noise")

        if "rain" in chosen_set and "haze_fog" not in chosen_set:
            if self.rng.random() < c.rain_implies_fog_p:
                chosen.append("haze_fog")
                chosen_set.add("haze_fog")

        if "exposure_shift" in chosen_set and "glare_flare" not in chosen_set:
            if self.rng.random() < c.exposure_implies_glare_p:
                chosen.append("glare_flare")
                chosen_set.add("glare_flare")

        return chosen

    def _enforce_constraints(self, chosen: List[str]) -> List[str]:
        c = self.cfg
        optical = [x for x in chosen if x in self.optical_set]
        non_optical = [x for x in chosen if x not in self.optical_set]

        if len(optical) <= c.max_optical_in_sample:
            return chosen

        keep_optical = list(self.rng.choice(optical, size=c.max_optical_in_sample, replace=False))
        return non_optical + keep_optical

    def sample_plan(self) -> Dict[str, float]:
        """Return {issue_name: severity} or {} for clean."""
        if self.rng.random() < self.cfg.p_clean:
            return {}

        K = int(self.rng.choice(self.k_values, p=self.k_probs))
        if K <= 0:
            return {}

        chosen = list(self.rng.choice(
            self.issues, size=min(K, len(self.issues)), replace=False, p=self.type_probs
        ))
        chosen = self._apply_couplings(chosen)
        chosen = self._enforce_constraints(chosen)

        return {issue: self._sample_severity() for issue in chosen}
