"""Caches every Hugging Face file AlpaSim's Alpamayo 1.5 driver loads, then proves it loads offline.

Usage (by scripts/setup_vm.sh, with AlpaSim's Python and HF_TOKEN set):
    python scripts/prepare_hf_cache.py
Files go to the standard cache ($HF_HOME, else ~/.cache/huggingface), which AlpaSim
mounts into the driver container; our driver config sets HF_HUB_OFFLINE=1 there.

What is needed and why (traced in third_party/alpasim at affc2eab and its venv):
- nvidia/Alpamayo-1.5-10B, all files. AlpaSim's driver calls
  Alpamayo1_5.from_pretrained (src/driver/src/alpasim_driver/models/alpamayo1_5_model.py:86).
  The checkpoint holds the VLM backbone weights too: its model.safetensors.index.json
  has 750 `vlm.*` tensors next to the `expert.*` ones.
- The base VLM named in that checkpoint's config.json (`vlm_name_or_path`, expected
  nvidia/Cosmos-Reason2-8B), config and processor files only. ReasoningVLA.__init__
  builds the backbone from its config (alpamayo1_5/models/base_model.py:374-390,
  Qwen3VLConfig.from_pretrained) and its processor (:258-266); the weights then come
  from the checkpoint above. Its own weights are loaded only by
  from_pretrained_submodules (:413-421), which nothing in AlpaSim or alpamayo1_5 calls.
- Qwen/Qwen3-VL-8B-Instruct, config and processor files only, although Alpamayo 1.5
  does not use it: it is the default `vlm_name_or_path` (base_model.py:211), and
  loading the config builds a default-constructed copy, whose __init__ loads that
  processor. Transformers logs `f"Model config {config}"` (configuration_utils.py:839,
  formatted even when info logging is off), so __repr__ -> to_json_string ->
  to_diff_dict runs `self.__class__()` (:893). Found with trace_hub_calls below.
- Qwen/Qwen3-VL-2B-Instruct, processor files only. The driver calls
  helper.get_processor (alpamayo1_5_model.py:91), i.e.
  AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME) (alpamayo1_5/helper.py:25, 197).

The offline check makes the driver's own two calls (alpamayo1_5_model.py:86-91; the
driver makes no other Hugging Face calls) in a fresh process set up like the driver
container: HF_HUB_OFFLINE=1 and no token. Transformers also asks for optional files
that may not exist (e.g. processor_config.json, absent from Cosmos-Reason2-8B):
offline, a missing optional file is skipped; online without a token, a gated repo
answers 401 for it and loading crashes. The check loads the weights on the CPU, so it
needs about 25 GB of free RAM.
"""

import json
import os
import subprocess
import sys
import traceback

from huggingface_hub import hf_hub_download, snapshot_download

CHECKPOINT = "nvidia/Alpamayo-1.5-10B"
EXPECTED_BASE_VLM = "nvidia/Cosmos-Reason2-8B"
PROCESSOR_REPO = "Qwen/Qwen3-VL-2B-Instruct"
CONFIG_DEFAULT_VLM = "Qwen/Qwen3-VL-8B-Instruct"  # base_model.py:211; see the docstring
# Configs, tokenizer, processor, chat template; no *.safetensors weights.
NON_WEIGHT_PATTERNS = ["*.json", "*.txt", "*.jinja", "*.model", "tokenizer*"]


def read_base_vlm(config_path: str) -> str | None:
    """Return `vlm_name_or_path` from an Alpamayo checkpoint's config.json."""
    with open(config_path, encoding="utf-8") as file:
        return json.load(file).get("vlm_name_or_path")


def check_base_vlm(base_vlm: str | None, expected: str = EXPECTED_BASE_VLM) -> None:
    """Stop if the checkpoint names a different base VLM than the one we cache."""
    if base_vlm != expected:
        sys.exit(f"Alpamayo 1.5 config.json names base VLM {base_vlm!r}, expected {expected!r}; "
                 "update EXPECTED_BASE_VLM after checking alpamayo1_5's loading code")


def download() -> str:
    """Download everything the driver needs; return the base VLM's repo id."""
    print("checkpoint:", snapshot_download(CHECKPOINT))
    base_vlm = read_base_vlm(hf_hub_download(CHECKPOINT, "config.json"))
    print("vlm_name_or_path:", base_vlm)
    check_base_vlm(base_vlm)
    for repo_id in (base_vlm, CONFIG_DEFAULT_VLM, PROCESSOR_REPO):
        print(f"{repo_id} (no weights):", snapshot_download(repo_id, allow_patterns=NON_WEIGHT_PATTERNS))
    return base_vlm


def container_like_env() -> dict[str, str]:
    """This environment as the driver container sees it: offline, with no token."""
    env = {name: value for name, value in os.environ.items()
           if name not in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    env["HF_HUB_OFFLINE"] = "1"
    # HF_TOKEN may also have been saved to $HF_HOME/token, which the container mounts.
    env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    return env


def call_site(depth: int = 6) -> str:
    """The calling frames, innermost last, as `folder/file.py:line function`."""
    frames = traceback.extract_stack()[:-3][-depth:]  # drop call_site and the tracing wrapper
    return " <- ".join(f"{'/'.join(frame.filename.split('/')[-2:])}:{frame.lineno} {frame.name}"
                       for frame in reversed(frames))


def trace_hub_calls() -> None:
    """Print the repo id and kwargs of every config, processor, and file lookup.

    Every file Transformers looks up goes through transformers.utils.hub.cached_files
    (cached_file and the shard loader call it by module global), so wrapping it, plus
    AutoConfig and AutoProcessor, shows everything the driver would read from the cache.
    """
    from transformers import AutoConfig, AutoProcessor
    from transformers.utils import hub

    def traced(name, function):
        def wrapper(*args, **kwargs):
            print(f"[hub] {name} args={args!r} kwargs={kwargs!r}\n      from {call_site()}", flush=True)
            return function(*args, **kwargs)
        return wrapper

    AutoConfig.from_pretrained = traced("AutoConfig.from_pretrained", AutoConfig.from_pretrained)
    AutoProcessor.from_pretrained = traced("AutoProcessor.from_pretrained", AutoProcessor.from_pretrained)
    hub.cached_files = traced("cached_files", hub.cached_files)


def load_like_the_driver() -> None:
    """Make the driver's Hugging Face calls (alpamayo1_5_model.py:86-91), on the CPU."""
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    trace_hub_calls()
    model = Alpamayo1_5.from_pretrained(CHECKPOINT, dtype=torch.bfloat16, attn_implementation="sdpa")
    helper.get_processor(model.tokenizer)


def verify_offline() -> None:
    """Run load_like_the_driver in a new process: HF_HUB_OFFLINE is read at import time."""
    print("loading Alpamayo 1.5 and its processors as the driver does (offline, no token; a few minutes)", flush=True)
    completed = subprocess.run([sys.executable, __file__, "--load-like-the-driver"], env=container_like_env())
    if completed.returncode != 0:
        sys.exit("offline check failed: the driver container would need a file that is not cached (see above)")
    print("offline check passed: the driver's model and processor loads work from the cache alone")


if __name__ == "__main__":
    if sys.argv[1:] == ["--load-like-the-driver"]:
        load_like_the_driver()
    else:
        download()
        verify_offline()
