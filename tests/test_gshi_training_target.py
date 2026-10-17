from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from simulation.gshi_utils import compute_gshi as compute_reference_gshi
from simulation.gshi_utils import load_gshi_config
from simulation.simulator import compute_gshi as compute_training_gshi


def test_training_and_reference_gshi_implementations_match() -> None:
    taxonomy_path = REPO / "configs/taxonomy/camera_issues.yaml"
    raw = yaml.safe_load(taxonomy_path.read_text())
    issues = list(raw["issues"])
    issue_to_group = {
        issue: group
        for group, group_issues in raw["groups"].items()
        for issue in group_issues
    }
    config = load_gshi_config(str(taxonomy_path))

    plans = [{}]
    plans.extend(
        {issue: severity}
        for issue in issues
        for severity in (0.08, 0.35, 0.75)
    )
    plans.extend(
        [
            {"haze_fog": 0.35, "sensor_noise": 0.18},
            {"motion_blur": 0.55, "lens_occlusion": 0.08},
        ]
    )

    for plan in plans:
        severities = np.zeros(len(issues), dtype=np.float32)
        for issue, severity in plan.items():
            severities[issues.index(issue)] = severity
        training_value = compute_training_gshi(
            severities, issues, issue_to_group, raw["gshi"]
        )
        reference_value = compute_reference_gshi(plan, config)
        assert np.isclose(training_value, reference_value, rtol=0, atol=1e-7), (
            plan,
            training_value,
            reference_value,
        )
