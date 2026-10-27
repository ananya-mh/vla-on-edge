"""Caches every Hugging Face file an AlpaSim Alpamayo driver loads, then proves it loads offline.

Usage (by scripts/setup_vm.sh, with AlpaSim's Python and HF_TOKEN set):
    python scripts/prepare_hf_cache.py [alpamayo1_5|alpamayo2_super]   # default alpamayo1_5
Files go to the standard cache ($HF_HOME, else ~/.cache/huggingface), which AlpaSim
mounts into the driver container; our driver configs set HF_HUB_OFFLINE=1 there.

What each model needs and why (traced in third_party/alpasim at affc2eab and its venv,
transformers 4.57.1, with trace_hub_calls below).

Alpamayo 1.5:
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
  to_diff_dict runs `self.__class__()` (:893).
- Qwen/Qwen3-VL-2B-Instruct, processor files only. The driver calls
  helper.get_processor (alpamayo1_5_model.py:91), i.e.
  AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME) (alpamayo1_5/helper.py:25, 197).

Alpamayo 2 Super:
- nvidia/Alpamayo2-Super, all files (71.6 GB) and nothing else. AlpaSim's driver calls
  Alpamayo2Super.from_pretrained (alpamayo2_model.py:202) and, on every prediction,
  helper.prepare_model_inputs (:360), which calls helper.get_processor
  (alpamayo2_super/helper.py:49-65). Its config.json embeds the VLM config
  (`vlm_config`) and leaves `vlm_name_or_path` empty, so the tokenizer
  (config.py:37, models/alpamayo2_super.py:124-131) and the processor come from the
  checkpoint's own files (resolve_checkpoint_name_or_path, config.py:43-49). The
  default-constructed config built for Transformers' log message (see above) has no
  `vlm_name_or_path` and returns before loading anything (config.py:136-137).
  Traced 2026-10-26 without the weights (config, then the model built on the meta
  device, then get_processor): 35 file lookups, all in nvidia/Alpamayo2-Super; absent
  optional files: processor_config.json, chat_template.json, audio_tokenizer_config.json.

The offline check makes the driver's own Hugging Face calls in a fresh process set up
like the driver container: HF_HUB_OFFLINE=1 and no token. Transformers also asks for
optional files that may not exist (e.g. processor_config.json, absent from
Cosmos-Reason2-8B and Alpamayo2-Super): offline, a missing optional file is skipped;
online without a token, a gated repo answers 401 for it and loading crashes. Alpamayo
1.5 loads on the CPU (about 25 GB of free RAM); Alpamayo 2 Super loads on the GPU like
the driver (device_map="cuda"; about 70 GB of GPU memory, so run it while the GPU is
otherwise idle), or on the CPU without CUDA (about 75 GB of free RAM).
"""

import json
import os
import subprocess
import sys
import traceback

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = ("alpamayo1_5", "alpamayo2_super")
CHECKPOINT_1_5 = "nvidia/Alpamayo-1.5-10B"
EXPECTED_BASE_VLM = "nvidia/Cosmos-Reason2-8B"
PROCESSOR_REPO = "Qwen/Qwen3-VL-2B-Instruct"
CONFIG_DEFAULT_VLM = "Qwen/Qwen3-VL-8B-Instruct"  # base_model.py:211; see the docstring
CHECKPOINT_2_SUPER = "nvidia/Alpamayo2-Super"  # AlpaSim's src/wizard/configs/driver/alpamayo2.yaml
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


def check_self_contained(config: dict) -> None:
    """Stop unless an Alpamayo 2 Super config.json needs no other repo (see the docstring)."""
    if config.get("vlm_name_or_path") or "vlm_config" not in config:
        sys.exit(f"Alpamayo 2 Super config.json has vlm_name_or_path={config.get('vlm_name_or_path')!r} "
                 f"and {'a' if 'vlm_config' in config else 'no'} vlm_config: it may load another repo; "
                 "re-trace with --load-like-the-driver before trusting the cache")


def download_alpamayo1_5() -> None:
    """Download everything AlpaSim's alpamayo1_5 driver needs."""
    print("checkpoint:", snapshot_download(CHECKPOINT_1_5))
    base_vlm = read_base_vlm(hf_hub_download(CHECKPOINT_1_5, "config.json"))
    print("vlm_name_or_path:", base_vlm)
    check_base_vlm(base_vlm)
    for repo_id in (base_vlm, CONFIG_DEFAULT_VLM, PROCESSOR_REPO):
        print(f"{repo_id} (no weights):", snapshot_download(repo_id, allow_patterns=NON_WEIGHT_PATTERNS))


def download_alpamayo2_super() -> None:
    """Download everything AlpaSim's alpamayo2 driver needs: its checkpoint alone."""
    with open(hf_hub_download(CHECKPOINT_2_SUPER, "config.json"), encoding="utf-8") as file:
        check_self_contained(json.load(file))
    print("checkpoint (71.6 GB):", snapshot_download(CHECKPOINT_2_SUPER))


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
    """Print the repo id and kwargs of every config, tokenizer, processor, and file lookup.

    Every file Transformers looks up goes through transformers.utils.hub.cached_files
    (cached_file and the shard loader call it by module global), so wrapping it, plus
    the Auto classes, shows everything the driver would read from the cache.
    """
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from transformers.utils import hub

    def traced(name, function):
        def wrapper(*args, **kwargs):
            print(f"[hub] {name} args={args!r} kwargs={kwargs!r}\n      from {call_site()}", flush=True)
            return function(*args, **kwargs)
        return wrapper

    AutoConfig.from_pretrained = traced("AutoConfig.from_pretrained", AutoConfig.from_pretrained)
    AutoTokenizer.from_pretrained = traced("AutoTokenizer.from_pretrained", AutoTokenizer.from_pretrained)
    AutoProcessor.from_pretrained = traced("AutoProcessor.from_pretrained", AutoProcessor.from_pretrained)
    hub.cached_files = traced("cached_files", hub.cached_files)


def load_alpamayo1_5_like_the_driver() -> None:
    """Make the driver's Hugging Face calls (alpamayo1_5_model.py:86-91), on the CPU."""
    import torch
    from alpamayo1_5 import helper
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    trace_hub_calls()
    model = Alpamayo1_5.from_pretrained(CHECKPOINT_1_5, dtype=torch.bfloat16, attn_implementation="sdpa")
    helper.get_processor(model.tokenizer)


def load_alpamayo2_super_like_the_driver() -> None:
    """Make the driver's Hugging Face calls (alpamayo2_model.py:202-207 and, per prediction, :360)."""
    import torch
    from alpamayo2_super import helper
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

    trace_hub_calls()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading on {device}", flush=True)
    model = Alpamayo2Super.from_pretrained(CHECKPOINT_2_SUPER, dtype=torch.bfloat16, device_map=device)
    helper.get_processor(model.tokenizer, model.config)  # what prepare_model_inputs calls


def verify_offline(model: str) -> None:
    """Load the model in a new process: HF_HUB_OFFLINE is read at import time."""
    print(f"loading {model} and its processors as the driver does (offline, no token; a few minutes)", flush=True)
    completed = subprocess.run([sys.executable, __file__, "--load-like-the-driver", model], env=container_like_env())
    if completed.returncode != 0:
        sys.exit("offline check failed: the driver container would need a file that is not cached (see above)")
    print("offline check passed: the driver's model and processor loads work from the cache alone")


def main() -> None:
    """Download and check one model, or (in the subprocess) only load it."""
    arguments = sys.argv[1:]
    load_only = arguments[:1] == ["--load-like-the-driver"]
    if load_only:
        arguments = arguments[1:]
    model = arguments[0] if arguments else "alpamayo1_5"
    if model not in MODELS:
        sys.exit(f"unknown model {model!r}; choose one of {', '.join(MODELS)}")
    if load_only and model == "alpamayo1_5":
        load_alpamayo1_5_like_the_driver()
    elif load_only:
        load_alpamayo2_super_like_the_driver()
    else:
        if model == "alpamayo1_5":
            download_alpamayo1_5()
        else:
            download_alpamayo2_super()
        verify_offline(model)


if __name__ == "__main__":
    main()
