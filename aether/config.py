"""Config loading helpers: merges base + camera + kinematic + scenario into a
single OmegaConf node.

The custom ``${aether:<rel>}`` resolver expands to ``<aether-repo-root>/<rel>``
so that scenario yamls can ship portable, repo-relative paths to vendored
checkpoints / data without hard-coding ``/home/<user>/...``.
"""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]

if not OmegaConf.has_resolver("aether"):
    OmegaConf.register_new_resolver(
        "aether",
        lambda rel: str((_REPO_ROOT / rel).resolve()),
        replace=False,
    )


def load_cfg(scenario_path: str, base_path: str, camera_path: str,
             kinematic_path: str) -> DictConfig:
    scenario = OmegaConf.load(scenario_path)
    parent = scenario.get("extends")
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = Path(scenario_path).resolve().parent / parent_path
        del scenario["extends"]
        scenario = OmegaConf.merge(OmegaConf.load(parent_path), scenario)
    cfg = OmegaConf.merge(
        {"base":      OmegaConf.load(base_path)},
        {"camera":    OmegaConf.load(camera_path)},
        {"kinematic": OmegaConf.load(kinematic_path)},
        {"scenario":  scenario},
    )
    # Stash the scenario yaml's directory so relative sidecar paths
    # (e.g. traffic_log: traffic/foo.npz) can be resolved later.
    OmegaConf.set_struct(cfg, False)
    cfg.scenario._yaml_dir = str(Path(scenario_path).resolve().parent)
    return cfg
