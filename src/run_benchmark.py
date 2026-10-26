"""Command-line entry point: runs one model on every scenario and seed of an experiment.

Example:
    python -m src.run_benchmark --model dummy:bf16 --experiment smoke_test --sim mock

Each run writes to results/runs/<timestamp>_<model>_<variant>_<experiment>/:
    metadata.json   git commit, resolved config, versions, GPU
    steps.jsonl     one record per decision step
    episodes.jsonl  the simulator's summary for each episode
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.adapters.base import ModelAdapter
from src.adapters.dummy_adapter import DummyAdapter
from src.agent.loop import run_episode
from src.sim.base import SimEnv
from src.sim.mock_env import MockEnv
from src.sim.real_env import RealSimEnv
from src.utils.config import resolve_run_config
from src.utils.logging import LOG_FORMAT_VERSION, append_jsonl
from src.utils.run_metadata import build_run_metadata, save_run_metadata

# Maps the `adapter` field in models.yaml to an adapter class.
ADAPTERS = {"dummy": DummyAdapter}

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "runs"


def create_adapter(model_config: dict, agent_config: dict) -> ModelAdapter:
    """Instantiate the adapter named in the model's config entry."""
    adapter_name = model_config["adapter"]
    if adapter_name not in ADAPTERS:
        raise ValueError(f"unknown adapter {adapter_name!r}; known adapters: {sorted(ADAPTERS)}")
    return ADAPTERS[adapter_name](model_config, agent_config)


def create_env(sim: str, agent_config: dict) -> SimEnv:
    """Instantiate the requested simulator."""
    if sim == "mock":
        return MockEnv(agent_config)
    return RealSimEnv(agent_config)


def make_run_dir(results_dir: Path, run_config: dict) -> Path:
    """Create and return a new, uniquely named run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = run_config["model"]
    name = f"{timestamp}_{model['name']}_{model['variant']}_{run_config['experiment']['name']}"
    run_dir = results_dir / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model name from models.yaml, optionally name:variant")
    parser.add_argument("--experiment", required=True, help="experiment name from configs/experiments/")
    parser.add_argument("--sim", choices=["mock", "real"], default="mock")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved config and episodes, then exit")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path | None:
    """Run the benchmark and return the run directory (None for a dry run)."""
    args = parse_args(argv)
    run_config = resolve_run_config(args.model, args.experiment, args.sim)
    seeds = run_config["experiment"]["seeds"]

    if args.dry_run:
        print(json.dumps(run_config, indent=2))
        episodes = [f"{scenario['name']}/seed={seed}" for scenario in run_config["scenarios"] for seed in seeds]
        print(f"\nDry run: {len(episodes)} episodes: {', '.join(episodes)}")
        return None

    env = create_env(args.sim, run_config["agent"])
    adapter = create_adapter(run_config["model"], run_config["agent"])
    run_dir = make_run_dir(args.results_dir, run_config)
    save_run_metadata(run_dir, build_run_metadata(run_config))

    for scenario in run_config["scenarios"]:
        for seed in seeds:
            episode_metrics = run_episode(
                env,
                adapter,
                scenario,
                seed,
                run_config["agent"],
                run_config["model"],
                run_dir / "steps.jsonl",
            )
            episode_record = {
                "log_format_version": LOG_FORMAT_VERSION,
                "scenario": scenario["name"],
                "seed": seed,
                "tags": scenario["tags"],
                **episode_metrics,
            }
            append_jsonl(run_dir / "episodes.jsonl", episode_record)
            print(f"{scenario['name']} seed={seed}: {episode_metrics}")

    print(f"Results written to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
