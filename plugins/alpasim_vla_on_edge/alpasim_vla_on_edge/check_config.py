"""Checks AlpaSim's generated configs for our wrapper without a GPU or Docker.

Run with AlpaSim's Python on a wizard log dir made with `wizard.run_method=NONE`
(or `deploy=docker_build_only`):
    python -m alpasim_vla_on_edge.check_config <log_dir>

It repeats the two steps the driver does at startup that can fail on config: the
strict schema merge of driver-config.yaml and the plugin registry lookup of
model_type. It also checks the docker-compose driver service for our mount,
environment variable, and install command.
"""

import sys
from pathlib import Path

import yaml
from alpasim_driver.schema import DriverConfig
from alpasim_plugins.plugins import models as model_registry
from omegaconf import OmegaConf

from alpasim_vla_on_edge.logged_model import LOG_DIR_VARIABLE

EXPECTED_MODEL_TYPE = "vla_on_edge_alpamayo1_5"


def check_driver_config(log_dir: Path) -> list[str]:
    """Return problems with driver-config.yaml (empty if fine)."""
    loaded = OmegaConf.load(log_dir / "driver-config.yaml")
    config = OmegaConf.merge(OmegaConf.structured(DriverConfig), loaded)  # same as the driver's main()
    problems = []
    if config.model.model_type != EXPECTED_MODEL_TYPE:
        problems.append(f"driver model_type is {config.model.model_type!r}, expected {EXPECTED_MODEL_TYPE!r}")
    model_registry.get(config.model.model_type)  # raises if the entry point is not installed
    if config.inference.max_batch_size != 1:
        problems.append(f"max_batch_size is {config.inference.max_batch_size}; latency is reported at batch size 1")
    return problems


def check_compose_driver_service(log_dir: Path) -> list[str]:
    """Return problems with the driver service in docker-compose.yaml (empty if fine)."""
    compose = yaml.safe_load((log_dir / "docker-compose.yaml").read_text())
    drivers = [service for name, service in compose["services"].items() if name.startswith("driver")]
    if len(drivers) != 1:
        return [f"expected one driver service, found {len(drivers)}"]
    driver = drivers[0]
    problems = []
    command = " ".join(driver["command"])
    for flag in ("--offline", "--no-deps", "/mnt/vla_on_edge/dist/wheels/"):
        if flag not in command:
            problems.append(f"driver command does not install our wheels with {flag}")
    if not any(volume.endswith(":/mnt/vla_on_edge:ro") for volume in driver["volumes"]):
        problems.append("our repo is not mounted read-only at /mnt/vla_on_edge")
    environment = driver.get("environment", [])
    if not any(entry.startswith(f"{LOG_DIR_VARIABLE}=/mnt/output/") for entry in environment):
        problems.append(f"{LOG_DIR_VARIABLE} is not set under /mnt/output (the mounted log dir)")
    if "VLA_ON_EDGE_CONFIG_DIR=/mnt/vla_on_edge/configs" not in environment:
        problems.append("VLA_ON_EDGE_CONFIG_DIR does not point at the mounted configs/")
    commit = next((entry.split("=", 1)[1] for entry in environment if entry.startswith("VLA_ON_EDGE_GIT_COMMIT=")), None)
    if commit in (None, "", "unknown"):
        problems.append("VLA_ON_EDGE_GIT_COMMIT was not set on the host; run metadata would lack the commit")
    return problems


def main() -> None:
    """Print each problem and exit non-zero if there are any."""
    log_dir = Path(sys.argv[1])
    problems = check_driver_config(log_dir) + check_compose_driver_service(log_dir)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        sys.exit(1)
    print(f"OK: {log_dir} runs {EXPECTED_MODEL_TYPE} with our logging")


if __name__ == "__main__":
    main()
