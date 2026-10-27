# alpasim_vla_on_edge

AlpaSim plugin that wraps AlpaSim's built-in driver models (Alpamayo 1.5 and Alpamayo 2 Super)
and logs every prediction with this repo's validator (flag-only) and JSONL logger.
Predictions are returned unchanged.

## Install (into AlpaSim's own environment)

AlpaSim's environment already has every dependency, so install without resolving:

    uv pip install --python third_party/alpasim/.venv/bin/python --no-deps -e . -e plugins/alpasim_vla_on_edge

Both installs must be editable: the plugin reads `configs/` from this checkout.
Note: `uv sync` in `third_party/alpasim` removes packages it does not manage, so
re-run the install after any sync.

## Use

Our Hydra overrides live in `configs/alpasim/vla_on_edge/`: `alpamayo1_5.yaml` and
`alpamayo2_super.yaml` each switch the driver to our wrapper
(`vla_on_edge_alpamayo1_5` or `vla_on_edge_alpamayo2_super`), and both include
`wrapper.yaml`, which mounts this repo into the driver container at
`/mnt/vla_on_edge`, installs both packages there from host-built wheels (offline,
`--no-deps`) before the driver starts, runs offline against the mounted Hugging Face
cache, and sets `VLA_ON_EDGE_LOG_DIR=/mnt/output/vla_on_edge`, which is
`<wizard.log_dir>/vla_on_edge` on the host. From `third_party/alpasim`:

    VLA_ON_EDGE_REPO=$REPO uv run alpasim_wizard deploy=local topology=1gpu driver=alpamayo1_5 \
      "hydra.searchpath=[file://$REPO/configs/alpasim]" +vla_on_edge=alpamayo1_5 \
      wizard.log_dir=$LOG_DIR "scenes.scene_ids=[$SCENE]"

For Alpamayo 2 Super use `driver=alpamayo2 +vla_on_edge=alpamayo2_super`, and on two
GPUs `topology=vla_on_edge_2gpu` (`configs/alpasim/topology/`: the model alone on
GPU 0, the renderer and physics on GPU 1). Check the generated configs without a GPU
(add `wizard.run_method=NONE` or use `deploy=docker_build_only`), then:

    .venv/bin/python -m alpasim_vla_on_edge.check_config $LOG_DIR --model alpamayo2_super --layout 2gpu

## Test

    cd plugins/alpasim_vla_on_edge && ../../third_party/alpasim/.venv/bin/python -m pytest tests
