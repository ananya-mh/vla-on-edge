"""Tests for scripts/gpu_monitor.py with a fake nvidia-smi on PATH."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from vla_on_edge.eval.fake_results import write_fake_run
from vla_on_edge.eval.report import build_report

REPO = Path(__file__).resolve().parents[1]
FAKE_NVIDIA_SMI = """#!/bin/sh
case "$1" in
  --query-gpu=index) echo "0" ;;
  --query-gpu=*) echo "0, GPU-aaaa, 61000, 81559" ;;
  --query-compute-apps=*) echo "GPU-aaaa, 4242, 40960"; echo "GPU-aaaa, 4343, [N/A]" ;;
esac
"""


def run_monitor(results_dir: Path, tmp_path: Path, seconds: float) -> subprocess.CompletedProcess:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "nvidia-smi").write_text(FAKE_NVIDIA_SMI)
    (bin_dir / "nvidia-smi").chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    process = subprocess.Popen(
        [sys.executable, str(REPO / "scripts/gpu_monitor.py"), str(results_dir), "--interval-s", "0.2"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    time.sleep(seconds)
    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=10)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def test_monitor_logs_samples_and_writes_peaks(tmp_path):
    results = tmp_path / "vm_test"
    results.mkdir()
    (results / "current_stage").write_text("stage2")
    completed = run_monitor(results, tmp_path, seconds=1.0)
    assert completed.returncode == 0, completed.stderr
    rows = (results / "gpu_memory.csv").read_text().splitlines()
    assert rows[0] == "time_s,stage,kind,gpu,pid,container,used_mib,total_mib"
    assert any(",stage2,gpu,0,,,61000.0,81559.0" in row for row in rows)
    assert any(",stage2,process,0,4242,host,40960.0," in row for row in rows)  # no /proc cgroup on macOS
    peaks = json.loads((results / "gpu_memory_peaks.json").read_text())
    assert peaks == {"stage2": {"gpu 0": 61000.0, "host": 40960.0}}  # the [N/A] row is skipped
    assert "GPU memory peaks, stage2" in completed.stdout


def test_peaks_from_earlier_stages_are_kept(tmp_path):
    results = tmp_path / "vm_test"
    results.mkdir()
    (results / "gpu_memory_peaks.json").write_text(json.dumps({"stage1": {"gpu 0": 50000.0}}))
    (results / "current_stage").write_text("stage2")
    run_monitor(results, tmp_path, seconds=0.6)
    peaks = json.loads((results / "gpu_memory_peaks.json").read_text())
    assert set(peaks) == {"stage1", "stage2"}


def test_monitor_without_nvidia_smi_exits_quietly(tmp_path):
    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    completed = subprocess.run([sys.executable, str(REPO / "scripts/gpu_monitor.py"), str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0
    assert "not logging GPU memory" in completed.stderr


def test_report_includes_monitor_peaks(tmp_path):
    stage_dir = write_fake_run(tmp_path / "vm_test" / "stage3")
    (tmp_path / "vm_test" / "gpu_memory_peaks.json").write_text(
        json.dumps({"stage3": {"gpu 0": 61440.0, "vm-driver-0-1": 40960.0}}))
    report = build_report(stage_dir, tmp_path / "report.md").read_text()
    assert "## GPU memory (nvidia-smi peaks)" in report
    assert "| vm-driver-0-1 | 40.0 |" in report
