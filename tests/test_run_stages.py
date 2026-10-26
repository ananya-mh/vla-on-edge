"""Runs scripts/run_stages.sh end to end with a stub `uv` instead of AlpaSim.

The stub writes fake AlpaSim outputs (and plugin logs when our wrapper overrides
are passed) for real runs, and calls AlpaSim's real wizard in generate-only mode
for `wizard.run_method=NONE`, so stage 2's config check sees real configs.
Needs the local AlpaSim clone with its environment set up.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from vla_on_edge.utils.config import CONFIG_DIR, load_alpasim_scenes

REPO = Path(__file__).resolve().parents[1]
ALPASIM_PYTHON = REPO / "third_party/alpasim/.venv/bin/python"

STUB = '''#!{python}
import os, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
from vla_on_edge.eval.fake_results import STORIES, write_alpasim_outputs, write_plugin_logs
import random

args = sys.argv[1:]
real_uv = {real_uv!r}
if args[:3] != ["run", "--no-sync", "alpasim_wizard"]:
    os.execv(real_uv, [real_uv, *args])  # e.g. `uv build` from build_wheels.sh
overrides = args[3:]
if "wizard.run_method=NONE" in overrides:
    # Real wizard, generate-only, no scene download.
    kept = [o for o in overrides if not o.startswith("scenes.scene_ids=")]
    kept = ["deploy=docker_build_only" if o == "deploy=local" else o for o in kept]
    os.execv(real_uv, [real_uv, "run", "--no-sync", "alpasim_wizard", *kept])
values = dict(o.split("=", 1) for o in overrides if "=" in o and not o.startswith("+"))
log_dir = Path(values["wizard.log_dir"])
scene = values["scenes.scene_ids"].strip("[]")
if os.environ.get("FAIL_SCENE") == scene:
    (log_dir / "txt-logs").mkdir(parents=True)
    (log_dir / "txt-logs" / "runtime.log").write_text("simulated failure")
    sys.exit(3)
write_alpasim_outputs(log_dir, scene, STORIES[1])
if "+vla_on_edge=alpamayo1_5" in overrides:
    write_plugin_logs(log_dir, STORIES[1], random.Random(0))
'''

pytestmark = pytest.mark.skipif(not ALPASIM_PYTHON.exists(), reason="local AlpaSim environment not set up")


@pytest.fixture
def stub_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uv"
    stub.write_text(STUB.format(python=sys.executable, repo=str(REPO), real_uv=shutil.which("uv")))
    stub.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUN_ID": "test",
        "WORKDIR": str(tmp_path / "work"),
        "RESULTS_ROOT": str(tmp_path / "results"),
    }


def run_stage(stage: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(REPO / "scripts/run_stages.sh"), stage], env=env,
                          capture_output=True, text=True, timeout=300)


def test_all_three_stages_produce_results_and_a_report(stub_env):
    results = Path(stub_env["RESULTS_ROOT"]) / "vm_test"
    smoke = load_alpasim_scenes(CONFIG_DIR)["smoke_scene"]
    for stage in ("1", "2", "3"):
        completed = run_stage(stage, stub_env)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    assert (results / "stage1/alpamayo1_5_builtin" / smoke / "aggregate/results-summary.json").exists()
    assert not (results / "stage1/alpamayo1_5_builtin" / smoke / "vla_on_edge").exists()
    assert (results / "stage2/alpamayo1_5" / smoke / "vla_on_edge/steps.jsonl").exists()
    scene_count = len(load_alpasim_scenes(CONFIG_DIR)["scenes"])
    assert len(list((results / "stage3/alpamayo1_5").iterdir())) == scene_count
    report = (Path(stub_env["RESULTS_ROOT"]) / "report.md").read_text()
    assert "## Worst cases" in report
    assert (results / "stage3/report.md").exists()


def test_a_failing_scene_stops_the_stage_and_keeps_its_logs(stub_env):
    second_scene = load_alpasim_scenes(CONFIG_DIR)["scenes"][1]["id"]
    completed = run_stage("3", {**stub_env, "FAIL_SCENE": second_scene})
    assert completed.returncode != 0
    assert "STAGE FAILED" in completed.stderr
    stage_dir = Path(stub_env["RESULTS_ROOT"]) / "vm_test/stage3/alpamayo1_5"
    assert len(list(stage_dir.iterdir())) == 2  # the good first scene and the failed one
    assert (stage_dir / second_scene / "txt-logs/runtime.log").exists()
