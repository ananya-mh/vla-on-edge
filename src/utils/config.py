"""Loads the YAML configs and resolves them into one run configuration.

Configs stay plain dicts: they are only read, and printing or saving them as JSON
is trivial.
"""

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_yaml(path: Path) -> dict:
    """Read a YAML file into a dict."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_model_spec(model_spec: str) -> tuple[str, str | None]:
    """Split "name:variant" into (name, variant). The variant is optional."""
    name, _, variant = model_spec.partition(":")
    return name, variant or None


def check_trajectory_spec(name: str, trajectory_spec: dict) -> None:
    """Raise ValueError if a model's trajectory spec is internally inconsistent."""
    implied_horizon = trajectory_spec["waypoint_count"] * trajectory_spec["spacing_s"]
    if abs(implied_horizon - trajectory_spec["horizon_s"]) > 1e-6:
        raise ValueError(
            f"model {name!r}: waypoint_count * spacing_s = {implied_horizon} s, "
            f"but horizon_s is {trajectory_spec['horizon_s']} s"
        )


def load_model_config(config_dir: Path, name: str, variant: str | None) -> dict:
    """Return the models.yaml entry for `name`, checking the requested variant."""
    models = load_yaml(config_dir / "models.yaml")["models"]
    if name not in models:
        raise ValueError(f"unknown model {name!r}; known models: {sorted(models)}")
    model_config = {"name": name, "variant": "bf16", **models[name]}
    if variant is not None and variant != model_config["variant"]:
        raise ValueError(f"model {name!r} has variant {model_config['variant']!r}, not {variant!r}")
    check_trajectory_spec(name, model_config["trajectory"])
    return model_config


def load_scenarios(config_dir: Path) -> dict[str, dict]:
    """Return every scenario from scenarios.yaml, merged with the defaults and named."""
    data = load_yaml(config_dir / "scenarios.yaml")
    scenarios = {}
    for name, settings in data["scenarios"].items():
        scenarios[name] = {**data["defaults"], **settings, "name": name}
    return scenarios


def apply_overrides(base: dict, overrides: dict) -> dict:
    """Return a copy of `base` with `overrides` applied, merging nested sections one level deep.

    For example, {"safety": {"enabled": True}} changes only safety.enabled and keeps
    the other safety settings.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = {**base[key], **value}
        else:
            merged[key] = value
    return merged


def resolve_run_config(model_spec: str, experiment_name: str, sim: str, config_dir: Path = CONFIG_DIR) -> dict:
    """Combine model, agent, experiment, and scenario configs into one dict for a run."""
    name, variant = parse_model_spec(model_spec)
    experiment = load_yaml(config_dir / "experiments" / f"{experiment_name}.yaml")
    all_scenarios = load_scenarios(config_dir)

    scenarios = []
    for scenario_name in experiment["scenarios"]:
        if scenario_name not in all_scenarios:
            raise ValueError(f"experiment {experiment_name!r} uses unknown scenario {scenario_name!r}")
        overrides = experiment.get("scenario_overrides", {})
        scenarios.append({**all_scenarios[scenario_name], **overrides})

    return {
        "model": load_model_config(config_dir, name, variant),
        # Experiment-level overrides apply to every model alike, so fairness is kept.
        "agent": apply_overrides(load_yaml(config_dir / "agent.yaml"), experiment.get("agent_overrides", {})),
        "experiment": {"name": experiment_name, **experiment},
        "scenarios": scenarios,
        "sim": sim,
    }
