"""Samples GPU memory with nvidia-smi every few seconds while the GPU stages run.

Usage: python scripts/gpu_monitor.py <results_dir> [--interval-s 5]
(started and stopped by scripts/run_stages.sh)

Writes <results_dir>/gpu_memory.csv with one row per GPU and one per GPU process
for each sample. Each process is tagged with its Docker container (AlpaSim runs the
driver and the renderer in separate containers), so model and renderer memory can
be told apart. On SIGTERM/SIGINT it writes <results_dir>/gpu_memory_peaks.json:
peak MiB per stage, for each GPU and each container. The current stage is read
from <results_dir>/current_stage, which run_stages.sh updates.
"""

import argparse
import csv
import json
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
CSV_FIELDS = ["time_s", "stage", "kind", "gpu", "pid", "container", "used_mib", "total_mib"]


def query_nvidia_smi(query: str, fields: str) -> list[list[str]]:
    """Run one nvidia-smi query and return its CSV rows (empty on any failure)."""
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-{query}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[value.strip() for value in line.split(",")] for line in output.splitlines() if line.strip()]


def to_mib(value: str) -> float | None:
    """nvidia-smi prints '[N/A]' when per-process memory is unavailable."""
    try:
        return float(value)
    except ValueError:
        return None


def container_for_pid(pid: str, cache: dict[str, str]) -> str:
    """Return the Docker container name a host PID runs in, or 'host'."""
    if pid in cache:
        return cache[pid]
    name = "host"
    try:
        match = CONTAINER_ID.search(Path(f"/proc/{pid}/cgroup").read_text())
    except OSError:
        match = None
    if match:
        inspected = subprocess.run(["docker", "inspect", "--format", "{{.Name}}", match.group(0)],
                                   capture_output=True, text=True)
        name = inspected.stdout.strip().lstrip("/") or match.group(0)[:12]
    cache[pid] = name
    return name


def take_sample(stage: str, cache: dict[str, str]) -> list[dict]:
    """Return this moment's rows: one per GPU, one per GPU process."""
    now = round(time.time(), 1)
    rows = []
    gpu_by_uuid = {}
    for index, uuid, used, total in query_nvidia_smi("gpu", "index,uuid,memory.used,memory.total"):
        gpu_by_uuid[uuid] = index
        rows.append({"time_s": now, "stage": stage, "kind": "gpu", "gpu": index, "pid": "",
                     "container": "", "used_mib": to_mib(used), "total_mib": to_mib(total)})
    for uuid, pid, used in query_nvidia_smi("compute-apps", "gpu_uuid,pid,used_memory"):
        rows.append({"time_s": now, "stage": stage, "kind": "process", "gpu": gpu_by_uuid.get(uuid, ""),
                     "pid": pid, "container": container_for_pid(pid, cache), "used_mib": to_mib(used),
                     "total_mib": ""})
    return rows


def update_peaks(peaks: dict, rows: list[dict]) -> None:
    """Keep the maximum used MiB per stage, per GPU and per container."""
    for row in rows:
        if row["used_mib"] is None:
            continue
        key = f"gpu {row['gpu']}" if row["kind"] == "gpu" else row["container"]
        stage_peaks = peaks.setdefault(row["stage"], {})
        stage_peaks[key] = max(stage_peaks.get(key, 0.0), row["used_mib"])


def current_stage(results_dir: Path) -> str:
    """Return the stage run_stages.sh is in, from <results_dir>/current_stage."""
    try:
        return (results_dir / "current_stage").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


def main() -> None:
    """Sample until stopped, then write the peaks file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--interval-s", type=float, default=5.0)
    args = parser.parse_args()
    if not query_nvidia_smi("gpu", "index"):
        print("gpu_monitor: nvidia-smi not available; not logging GPU memory", file=sys.stderr)
        return

    stopping = False

    def stop(signal_number, frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.results_dir / "gpu_memory.csv"
    write_header = not csv_path.exists()
    # Merge with earlier invocations (stages 1, 2, 3 are often run separately).
    peaks_path = args.results_dir / "gpu_memory_peaks.json"
    peaks: dict = json.loads(peaks_path.read_text()) if peaks_path.exists() else {}
    cache: dict[str, str] = {}
    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        while not stopping:
            rows = take_sample(current_stage(args.results_dir), cache)
            writer.writerows(rows)
            file.flush()  # keep the CSV useful even if the VM dies mid-run
            update_peaks(peaks, rows)
            deadline = time.monotonic() + args.interval_s
            while not stopping and time.monotonic() < deadline:
                time.sleep(0.2)
    peaks_path.write_text(json.dumps(peaks, indent=2, sort_keys=True))
    for stage, stage_peaks in sorted(peaks.items()):
        summary = ", ".join(f"{key}: {value / 1024:.1f} GiB" for key, value in sorted(stage_peaks.items()))
        print(f"GPU memory peaks, {stage}: {summary}")


if __name__ == "__main__":
    main()
