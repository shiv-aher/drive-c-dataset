import yaml
import numpy as np


class GSHIConfig:
    def __init__(self, weights, group_scale, issue_to_group, eps,
                 floor=1e-3, ceil=0.99, beta=0.85):
        self.weights = weights
        self.group_scale = group_scale
        self.issue_to_group = issue_to_group
        self.eps = eps
        self.floor = floor   # never return < floor
        self.ceil = ceil     # never return > ceil
        self.beta = beta     # <1 softens collapse, >1 makes harsher


def load_gshi_config(yaml_path: str) -> GSHIConfig:
    with open(yaml_path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    groups = cfg["groups"]
    gshi_cfg = cfg["gshi"]

    issue_to_group = {}
    for g, issues in groups.items():
        for i in issues:
            issue_to_group[i] = g

    # Optional fields with defaults
    floor = float(gshi_cfg.get("floor", 1e-3))
    ceil  = float(gshi_cfg.get("ceil", 0.99))
    beta  = float(gshi_cfg.get("beta", 0.85))

    return GSHIConfig(
        weights=gshi_cfg["weights"],
        group_scale=gshi_cfg["group_scale"],
        issue_to_group=issue_to_group,
        eps=float(gshi_cfg["eps"]),
        floor=floor,
        ceil=ceil,
        beta=beta,
    )


def compute_gshi(severity_dict: dict, cfg: GSHIConfig) -> float:
    """
    Stable + CVPR-friendly:
      log h = sum_i (w_i*g_i) * log( clamp(1 - s_i, eps, 1) )
      h = exp(beta * log h)
      return clip(h, floor, ceil)
    """
    logh = 0.0

    for issue, s in severity_dict.items():
        s = float(np.clip(s, 0.0, 1.0))

        w = float(cfg.weights[issue])
        g = float(cfg.group_scale[cfg.issue_to_group[issue]])
        eff = w * g

        base = np.clip(1.0 - s, cfg.eps, 1.0)
        logh += eff * float(np.log(base))

    h = float(np.exp(cfg.beta * logh))

    # never exactly 0 or 1
    h = float(np.clip(h, cfg.floor, cfg.ceil))
    return h
