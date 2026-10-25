"""Save a few mock-simulator camera frames as PNG files for a visual check.

Usage: .venv/bin/python scripts/save_mock_samples.py  (writes to results/samples/)
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # so `src` imports work when run as a script

from src.adapters.base import Action, MetaAction  # noqa: E402
from src.sim.mock_env import MockEnv  # noqa: E402
from src.utils.config import CONFIG_DIR, load_scenarios, load_yaml  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "samples"

# (scenario, seed, actions to take before capturing). Chosen to show an obstacle
# ahead, scattered obstacles, a narrow corridor, and a turned (rotated) view.
SAMPLES = [
    ("blocked_lane", 0, [MetaAction.KEEP_LANE] * 6),
    ("scattered_obstacles", 0, [MetaAction.KEEP_LANE] * 8),
    ("narrow_corridor", 1, [MetaAction.KEEP_LANE] * 8),
    ("open_corridor", 0, [MetaAction.STEER_LEFT] * 6),
]


def write_png(path: Path, image: np.ndarray) -> None:
    """Write an H x W x 3 uint8 image as a PNG using only the standard library."""
    height, width, _ = image.shape
    # Each PNG row starts with a filter-type byte; 0 means "no filter".
    rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    path.write_bytes(png)


def main() -> None:
    """Run each sample scenario for a few steps and save its camera views."""
    agent_config = load_yaml(CONFIG_DIR / "agent.yaml")
    scenarios = load_scenarios(CONFIG_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    env = MockEnv(agent_config)

    for scenario_name, seed, meta_actions in SAMPLES:
        env.reset(scenarios[scenario_name], seed)
        for meta_action in meta_actions:
            env.step(Action(meta_action=meta_action))
        observation = env.observe()
        for view_name, image in observation.camera_frames.items():
            path = OUTPUT_DIR / f"{scenario_name}_seed{seed}_step{observation.step_index}_{view_name}.png"
            write_png(path, image)
            print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
