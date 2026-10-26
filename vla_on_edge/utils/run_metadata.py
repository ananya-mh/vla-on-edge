"""Collects the metadata saved with every run, so results can be traced and reproduced."""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Libraries whose versions affect results. Add vLLM, the MCP SDK, etc. when they are used.
TRACKED_PACKAGES = ["numpy", "PyYAML"]


def run_command(command: list[str]) -> str | None:
    """Run a command and return its stripped stdout, or None if it fails or is missing."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_info() -> dict:
    """Return the current commit and whether there are uncommitted changes.

    Inside AlpaSim's driver container our code runs from an installed wheel, not a
    checkout, so the host records the commit and passes it in as
    VLA_ON_EDGE_GIT_COMMIT / VLA_ON_EDGE_GIT_DIRTY (scripts/run_stages.sh).
    """
    if os.environ.get("VLA_ON_EDGE_GIT_COMMIT"):
        return {
            "commit": os.environ["VLA_ON_EDGE_GIT_COMMIT"],
            "dirty": os.environ.get("VLA_ON_EDGE_GIT_DIRTY") == "true",
            "source": "host environment",
        }
    commit = run_command(["git", "rev-parse", "HEAD"])
    status = run_command(["git", "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(status) if status is not None else None, "source": "git"}


def host_tool_versions() -> dict[str, str | None]:
    """Return versions of host tools that run the simulator, as recorded by the host.

    Docker is not visible inside AlpaSim's driver container, so scripts/run_stages.sh
    passes the Compose version in as VLA_ON_EDGE_COMPOSE_VERSION.
    """
    compose = os.environ.get("VLA_ON_EDGE_COMPOSE_VERSION") or run_command(["docker", "compose", "version", "--short"])
    return {"docker_compose": compose}


def gpu_names() -> list[str]:
    """Return the names of visible NVIDIA GPUs (empty if there are none or nvidia-smi is missing)."""
    output = run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not output:
        return []
    return [line.strip() for line in output.splitlines()]


def library_versions() -> dict[str, str | None]:
    """Return installed versions of the tracked packages (None if not installed)."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_metadata(run_config: dict) -> dict:
    """Return the metadata record for a run with this resolved config."""
    model_config = run_config["model"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "model": model_config["name"],
        "variant": model_config["variant"],
        # Resolution the model resizes images to internally, where known (filled in models.yaml).
        "model_internal_image_resolution": model_config.get("internal_image_resolution"),
        "gpus": gpu_names(),
        "platform": platform.platform(),
        "library_versions": library_versions(),
        "host_tool_versions": host_tool_versions(),
        "resolved_config": run_config,
    }


def save_run_metadata(run_dir: Path, run_metadata: dict) -> Path:
    """Write metadata.json into the run directory and return its path."""
    path = run_dir / "metadata.json"
    path.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    return path
