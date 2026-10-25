"""Writes JSON Lines files: one JSON object per line, appended as the run progresses."""

import json
from pathlib import Path


def append_jsonl(path: Path, record: dict) -> None:
    """Append one record to a JSONL file, creating the file if needed.

    The file is reopened on every write so a crashed run still leaves complete lines.
    """
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read every record from a JSONL file."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
