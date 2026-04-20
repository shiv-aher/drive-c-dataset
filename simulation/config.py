from __future__ import annotations
from typing import Tuple
import yaml

from simulation.registry import load_taxonomy
from simulation.build_degradations import build_degradations
from simulation.composer import Composer
from simulation.simulator import Simulator
from simulation.policy import MixturePolicy, PolicyConfig


def build_simulator_and_policy(
    taxonomy_yaml: str,
    policy_yaml: str,
    seed: int = 0,
) -> Tuple[Simulator, MixturePolicy]:
    taxonomy = load_taxonomy(taxonomy_yaml)

    degradations = build_degradations()
    composer = Composer(degradations=degradations, seed=seed)
    simulator = Simulator(taxonomy=taxonomy, composer=composer)

    with open(policy_yaml, "r") as f:
        pcfg = yaml.safe_load(f)

    cfg = PolicyConfig(
        p_clean=pcfg["p_clean"],
        k_values=pcfg["k_dist"]["values"],
        k_probs=pcfg["k_dist"]["probs"],
        type_weights=pcfg["type_weights"],
        beta_alpha=pcfg["severity"]["beta_alpha"],
        beta_beta=pcfg["severity"]["beta_beta"],
        hard_mix_p=pcfg["severity"]["hard_mix_p"],
        hard_alpha=pcfg["severity"]["hard_alpha"],
        hard_beta=pcfg["severity"]["hard_beta"],
        min_sev=pcfg["severity"]["min_sev"],
        max_sev=pcfg["severity"]["max_sev"],
        low_light_implies_noise_p=pcfg["couplings"]["low_light_implies_noise_p"],
        rain_implies_fog_p=pcfg["couplings"]["rain_implies_fog_p"],
        exposure_implies_glare_p=pcfg["couplings"]["exposure_implies_glare_p"],
        max_optical_in_sample=pcfg["constraints"]["max_optical_in_sample"],
    )

    policy = MixturePolicy(
        issues=taxonomy.issues,
        groups=taxonomy.groups,
        cfg=cfg,
        seed=seed,
    )

    return simulator, policy
