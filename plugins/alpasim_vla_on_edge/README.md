# alpasim_vla_on_edge

AlpaSim plugin that wraps AlpaSim's built-in driver models (currently Alpamayo 1.5)
and logs every prediction with this repo's validator (flag-only) and JSONL logger.
Predictions are returned unchanged.

## Install (into AlpaSim's own environment)

AlpaSim's environment already has every dependency, so install without resolving:

    uv pip install --python third_party/alpasim/.venv/bin/python --no-deps -e . -e plugins/alpasim_vla_on_edge

Both installs must be editable: the plugin reads `configs/` from this checkout.
Note: `uv sync` in `third_party/alpasim` removes packages it does not manage, so
re-run the install after any sync.

## Use

Select the wrapper as the driver model and tell it where to log:

    driver.model.model_type: vla_on_edge_alpamayo1_5
    VLA_ON_EDGE_LOG_DIR=/path/to/logs

It writes `steps.jsonl` (one record per prediction) and `metadata.json`.
TODO: AlpaSim driver config (Hydra) for the wrapper, and mounting the log
directory into the driver container.

## Test

    cd plugins/alpasim_vla_on_edge && ../../third_party/alpasim/.venv/bin/python -m pytest tests
