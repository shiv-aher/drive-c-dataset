from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import os
import yaml
from pathlib import Path


@dataclass(frozen=True)
class Taxonomy:
    issues: List[str]                      # canonical order (len=12)
    groups: Dict[str, List[str]]           # group -> [issue,...]
    issue_to_group: Dict[str, str]         # issue -> group
    gshi: Dict[str, Any]                   # gshi config block


def load_taxonomy(yaml_path: str) -> Taxonomy:
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Taxonomy YAML not found: {yaml_path}")

    path = Path(yaml_path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        cfg = yaml.safe_load(f)

    groups: Dict[str, List[str]] = cfg.get("groups", {}) or {}
    issues_cfg: Dict[str, Any] = cfg.get("issues", {}) or {}

    # canonical order: preserve YAML group order
    issues: List[str] = []
    for _, glist in groups.items():
        for issue in glist:
            if issue not in issues:
                issues.append(issue)

    # include leftover issues (if any)
    for issue in issues_cfg.keys():
        if issue not in issues:
            issues.append(issue)

    issue_to_group: Dict[str, str] = {}
    for gname, glist in groups.items():
        for issue in glist:
            issue_to_group[issue] = gname

    gshi = cfg.get("gshi", {}) or {}

    return Taxonomy(
        issues=issues,
        groups=groups,
        issue_to_group=issue_to_group,
        gshi=gshi,
    )


def issue_index_map(issues: List[str]) -> Dict[str, int]:
    return {name: i for i, name in enumerate(issues)}
