"""Checks AlpaSim's generated configs for a run without a GPU or Docker.

Run with AlpaSim's Python on a wizard log dir made with `wizard.run_method=NONE`
(or `deploy=docker_build_only`):
    python -m alpasim_vla_on_edge.check_config <log_dir>             # our wrapper
    python -m alpasim_vla_on_edge.check_config <log_dir> --builtin   # stage 1

It repeats the two steps the driver does at startup that can fail on config: the
strict schema merge of driver-config.yaml and the plugin registry lookup of
model_type. It also checks the docker-compose driver service: for both runs, that
it is offline against the mounted Hugging Face cache; for the wrapper, our mount,
environment variables, and install command.
"""

import sys
from pathlib import Path

import yaml
from alpasim_driver.schema import DriverConfig
from alpasim_plugins.plugins import models as model_registry
from omegaconf import OmegaConf

from alpasim_vla_on_edge.logged_model import LOG_DIR_VARIABLE

WRAPPER_MODEL_TYPE = "vla_on_edge_alpamayo1_5"
BUILTIN_MODEL_TYPE = "alpamayo1_5"  # AlpaSim's own driver, wrapped by ours


def check_driver_config(log_dir: Path, expected_model_type: str) -> list[str]:
    """Return problems with driver-config.yaml (empty if fine)."""
    loaded = OmegaConf.load(log_dir / "driver-config.yaml")
    config = OmegaConf.merge(OmegaConf.structured(DriverConfig), loaded)  # same as the driver's main()
    problems = []
    if config.model.model_type != expected_model_type:
        problems.append(f"driver model_type is {config.model.model_type!r}, expected {expected_model_type!r}")
    model_registry.get(config.model.model_type)  # raises if the entry point is not installed
    if config.inference.max_batch_size != 1:
        problems.append(f"max_batch_size is {config.inference.max_batch_size}; latency is reported at batch size 1")
    return problems


def load_driver_service(log_dir: Path) -> dict | None:
    """Return the one driver service in docker-compose.yaml, or None if there is not exactly one."""
    compose = yaml.safe_load((log_dir / "docker-compose.yaml").read_text())
    drivers = [service for name, service in compose["services"].items() if name.startswith("driver")]
    return drivers[0] if len(drivers) == 1 else None


def check_offline_cache(driver: dict) -> list[str]:
    """The container has no token, so it must only read the host's Hugging Face cache."""
    problems = []
    if "HF_HUB_OFFLINE=1" not in driver.get("environment", []):
        problems.append("HF_HUB_OFFLINE=1 is not set; the driver could download at startup (401 on gated repos)")
    if not any(volume.endswith(":/root/.cache/huggingface") for volume in driver["volumes"]):
        problems.append("the host Hugging Face cache is not mounted at /root/.cache/huggingface")
    return problems


def check_wrapper_driver_service(driver: dict) -> list[str]:
    """Return problems with our wrapper's additions to the driver service (empty if fine)."""
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
    if not any(entry.startswith("VLA_ON_EDGE_COMPOSE_VERSION=") for entry in environment):
        problems.append("VLA_ON_EDGE_COMPOSE_VERSION is not passed in; run metadata would lack it")
    return problems


def check_run(log_dir: Path, builtin: bool) -> list[str]:
    """Return every problem with the generated configs (empty if fine)."""
    model_type = BUILTIN_MODEL_TYPE if builtin else WRAPPER_MODEL_TYPE
    problems = check_driver_config(log_dir, model_type)
    driver = load_driver_service(log_dir)
    if driver is None:
        return problems + ["expected exactly one driver service in docker-compose.yaml"]
    problems += check_offline_cache(driver)
    if not builtin:
        problems += check_wrapper_driver_service(driver)
    return problems


def main() -> None:
    """Print each problem and exit non-zero if there are any."""
    log_dir = Path(sys.argv[1])
    builtin = sys.argv[2:] == ["--builtin"]
    problems = check_run(log_dir, builtin)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        sys.exit(1)
    if builtin:
        print(f"OK: {log_dir} runs AlpaSim's {BUILTIN_MODEL_TYPE} offline against the mounted cache")
    else:
        print(f"OK: {log_dir} runs {WRAPPER_MODEL_TYPE} with our logging, offline")


if __name__ == "__main__":
    main()
