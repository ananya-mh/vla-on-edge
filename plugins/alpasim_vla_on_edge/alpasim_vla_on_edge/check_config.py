"""Checks AlpaSim's generated configs for a run without a GPU or Docker.

Run with AlpaSim's Python on a wizard log dir made with `wizard.run_method=NONE`
(or `deploy=docker_build_only`):
    python -m alpasim_vla_on_edge.check_config <log_dir> [--model M] [--layout L] [--builtin] [--experiment E]
M is alpamayo1_5 (default) or alpamayo2_super; L is 1gpu (default) or 2gpu;
--builtin checks a stage 1 run (AlpaSim's own driver) instead of our wrapper;
--experiment checks a labelled experiment's settings (see EXPERIMENTS).

It repeats the two steps the driver does at startup that can fail on config: the
strict schema merge of driver-config.yaml and the plugin registry lookup of
model_type. It also checks the docker-compose services: for both runs, that the
driver is offline against the mounted Hugging Face cache and that each service is on
the GPU the layout says; for the wrapper, our mount, environment variables, install
command, and video layouts (eval-config.yaml).
"""

import argparse
import sys
from pathlib import Path

import yaml
from alpasim_driver.schema import DriverConfig
from alpasim_plugins.plugins import models as model_registry
from omegaconf import OmegaConf

from alpasim_vla_on_edge.logged_model import LOG_DIR_VARIABLE

# models.yaml key -> (our wrapper's model_type, AlpaSim's built-in model_type it wraps)
MODEL_TYPES = {
    "alpamayo1_5": ("vla_on_edge_alpamayo1_5", "alpamayo1_5"),
    "alpamayo2_super": ("vla_on_edge_alpamayo2_super", "alpamayo2"),
}
# GPU ids per service for each layout: AlpaSim's topology/1gpu.yaml, and our
# configs/alpasim/topology/vla_on_edge_2gpu.yaml (the driver alone on GPU 0).
GPU_LAYOUTS = {
    "1gpu": {"driver": ["0"], "renderer": ["0"], "physics": ["0"]},
    "2gpu": {"driver": ["0"], "renderer": ["1"], "physics": ["1"]},
}


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


def load_services(log_dir: Path) -> dict[str, dict]:
    """Return docker-compose.yaml's services by name (e.g. driver-0, renderer-0)."""
    return yaml.safe_load((log_dir / "docker-compose.yaml").read_text())["services"]


def find_driver_service(services: dict[str, dict]) -> dict | None:
    """Return the one driver service, or None if there is not exactly one."""
    drivers = [service for name, service in services.items() if name.startswith("driver")]
    return drivers[0] if len(drivers) == 1 else None


def gpu_ids(service: dict) -> list[str]:
    """Return the GPU ids docker compose gives a service (empty if none)."""
    devices = service.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
    return [gpu_id for device in devices for gpu_id in device.get("device_ids", [])]


def check_gpu_layout(services: dict[str, dict], layout: str) -> list[str]:
    """Each GPU service must be exactly on the layout's GPUs (e.g. 2 Super alone on GPU 0)."""
    problems = []
    for prefix, expected in GPU_LAYOUTS[layout].items():
        found = [gpu_ids(service) for name, service in services.items() if name.startswith(prefix)]
        if found != [expected]:
            problems.append(f"{prefix} GPUs are {found}, expected one {prefix} service on {expected} ({layout})")
    return problems


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


def check_video_layouts(log_dir: Path) -> list[str]:
    """Our wrapper runs render the reasoning overlay video next to AlpaSim's default one."""
    layouts = yaml.safe_load((log_dir / "eval-config.yaml").read_text())["video"]["video_layouts"]
    expected = ["default", "reasoning_overlay"]
    if [str(layout).lower() for layout in layouts] != expected:
        return [f"eval video_layouts are {layouts}, expected {expected}"]
    return []


# Labelled experiments (configs/alpasim/vla_on_edge_experiment/<name>.yaml): the driver
# settings and cameras each must produce.
COMPARISON_6CAM_CAMERAS = [
    "camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
    "camera_rear_left_70fov", "camera_rear_right_70fov", "camera_front_tele_30fov",
]
EXPERIMENTS = {
    "alpamayo2_comparison_6cam": {
        "use_cameras": COMPARISON_6CAM_CAMERAS,
        "num_trajectory_samples": 6,
        "trajectory_candidate_microbatch_size": 3,
        "selection_strategy": "CLOSEST_LATERAL",
        "environment": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    },
}


def check_experiment(log_dir: Path, driver: dict, experiment: str, microbatch: int | None = None) -> list[str]:
    """The run uses exactly the experiment's driver settings, cameras, and allocator setting.

    `microbatch` overrides the expected candidate microbatch size (run_stages.sh's
    EXPERIMENT_MICROBATCH, the out-of-memory fallback); everything else must match.
    """
    expected = dict(EXPERIMENTS[experiment])
    if microbatch is not None:
        expected["trajectory_candidate_microbatch_size"] = microbatch
    config = yaml.safe_load((log_dir / "driver-config.yaml").read_text())
    rendered = yaml.safe_load((log_dir / "wizard-config.yaml").read_text())["runtime"]["simulation_config"]["cameras"]
    found = {
        "use_cameras": config["inference"]["use_cameras"],
        "num_trajectory_samples": config["model"]["num_trajectory_samples"],
        "trajectory_candidate_microbatch_size": config["model"]["trajectory_candidate_microbatch_size"],
        "selection_strategy": config["model"]["trajectory_selection"]["strategy"],
    }
    problems = [f"{experiment}: {key} is {found[key]!r}, expected {value!r}"
                for key, value in expected.items() if key in found and found[key] != value]
    missing = set(expected["use_cameras"]) - {camera["logical_id"] for camera in rendered}
    if missing:
        problems.append(f"{experiment}: the renderer is not asked for {sorted(missing)}")
    if expected["environment"] not in driver.get("environment", []):
        problems.append(f"{experiment}: {expected['environment']} is not set in the driver container")
    return problems


def check_run(
    log_dir: Path, model: str, layout: str, builtin: bool, experiment: str | None = None, microbatch: int | None = None
) -> list[str]:
    """Return every problem with the generated configs (empty if fine)."""
    wrapper_type, builtin_type = MODEL_TYPES[model]
    problems = check_driver_config(log_dir, builtin_type if builtin else wrapper_type)
    services = load_services(log_dir)
    problems += check_gpu_layout(services, layout)
    driver = find_driver_service(services)
    if driver is None:
        return problems + ["expected exactly one driver service in docker-compose.yaml"]
    problems += check_offline_cache(driver)
    if not builtin:
        problems += check_wrapper_driver_service(driver)
        problems += check_video_layouts(log_dir)
    if experiment is not None:
        problems += check_experiment(log_dir, driver, experiment, microbatch)
    return problems


def main() -> None:
    """Print each problem and exit non-zero if there are any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--model", choices=sorted(MODEL_TYPES), default="alpamayo1_5")
    parser.add_argument("--layout", choices=sorted(GPU_LAYOUTS), default="1gpu")
    parser.add_argument("--builtin", action="store_true", help="AlpaSim's own driver (stage 1), not our wrapper")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), help="a labelled experiment's settings")
    parser.add_argument("--microbatch", type=int, help="expected candidate microbatch size, if not the experiment's")
    args = parser.parse_args()
    problems = check_run(args.log_dir, args.model, args.layout, args.builtin, args.experiment, args.microbatch)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    if problems:
        sys.exit(1)
    wrapper_type, builtin_type = MODEL_TYPES[args.model]
    if args.builtin:
        print(f"OK: {args.log_dir} runs AlpaSim's {builtin_type} offline against the mounted cache ({args.layout})")
    else:
        label = f", experiment {args.experiment}" if args.experiment else ""
        print(f"OK: {args.log_dir} runs {wrapper_type} with our logging, offline ({args.layout}{label})")


if __name__ == "__main__":
    main()
