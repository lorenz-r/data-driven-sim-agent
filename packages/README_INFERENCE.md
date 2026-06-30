# SMART Inference

This note describes the current SMART inference and evaluation flow in this
repository. It is meant as a practical runbook for producing rollout artifacts,
computing metrics, and recording the configuration used for a run.

## Inference

`packages/smart/infer2.py` runs SMART on either one processed scenario `.pkl`
file or all `.pkl` files in a processed scenario directory.

The inference pipeline currently:

- loads the model configuration from the checkpoint;
- normalizes the agent and map token paths to the local token files;
- preprocesses each SMART scenario with the configured `TokenProcessor`;
- generates multiple stochastic or deterministic rollouts per scenario;
- saves all scenario predictions into one collected batch `.npz` file;
- optionally saves per-scenario SMART `.npz`, AV2 parquet, and GIF
  visualizations for a limited number of scenarios.

Example:

```bash
uv run python packages/smart/infer2.py \
  --input packages/smart/data/processed/current/val/ \
  --output packages/smart/data/preds \
  --checkpoint checkpoints/last.ckpt \
  --batch-size 8 \
  --rollouts-per-scenario 16 \
  --batch-visualization-limit 2
```

### Important Inputs

- `--input`: processed SMART scenario `.pkl` file or directory of `.pkl` files.
- `--checkpoint`: SMART checkpoint used for model weights and model config.
- `--raw-data-folder`: raw AV2 scenario directory. This is only required when
  GIF visualizations are enabled.
- `--output`: output directory for batch predictions and optional scenario-level
  artifacts.
- `--device`: `cpu` or `cuda`.
- `--batch-size`: number of scenarios to run per model inference batch.

### Rollouts

`--rollouts-per-scenario` controls how many sampled futures are generated per
scenario. The default is `16`. Waymo-style evaluation often uses `32`, but `16`
is currently used as a speed/coverage compromise for larger validation batches.

Scenarios are batched for the model inference pass according to `--batch-size`.
Per-scenario preprocessing and map-token matching still run per scenario, then
the prepared graphs are collated for rollout inference.

By default, `--sample-tokens` is enabled, so repeated rollouts can differ because
tokens are sampled. If `--no-sample-tokens` is used, inference takes deterministic
top-1 tokens. In that mode, requesting more than one rollout will produce
duplicate rollouts.

### Visualization Limit

`--batch-visualization-limit` controls how many scenarios are visualized:

- `0`: no scenario-level SMART/AV2/GIF artifacts;
- positive integer: visualize at most that many scenarios;
- negative value: visualize all scenarios.

For visualized scenarios, the first rollout is used for the AV2 conversion and
GIF visualization. All rollouts are still stored in the collected batch `.npz`.

### Outputs

Inference always writes a collected batch file:

```text
<output>/batch_predictions_<run_id>.npz
```

This file contains the prediction arrays for every processed scenario, including:

- `pred_traj`: predicted xy positions with shape
  `(rollouts, agents, future_steps, 2)`;
- `pred_head`: predicted headings with shape
  `(rollouts, agents, future_steps)`;
- `pred_valid_mask`: prediction validity with shape
  `(rollouts, agents, future_steps)`;
- `gt`, `gt_head`, and `gt_valid_mask` when available;
- agent metadata such as `agent_id`, `agent_type`, and `agent_category`;
- run metadata such as checkpoint info, model config, CLI flags, runtime
  versions, and scenario ids.

For scenarios selected by the visualization limit, inference also writes
scenario-level artifacts under:

```text
<output>/<scenario_id>/
```

Depending on the requested output format and visualization settings, this may
include:

- `smart/<scenario_id>.npz`;
- `av2/scenario_<scenario_id>.parquet`;
- `gt_<scenario_id>.GIF`;
- `pred_<scenario_id>.GIF`.

## Configuration Caveat

The output metadata currently records the model config loaded from the
checkpoint plus inference CLI/runtime metadata. It does not guarantee that every
training-time choice outside the checkpoint model config is captured, such as
the exact loss implementation, tokenizer-building procedure, data manifest, or
experiment notes.

Until this is improved, record inference results together with:

- training config used to produce the checkpoint;
- token file paths used for agent and map tokens;
- metrics JSON produced by evaluation.
- optionally the data manifest (even though we will try to standardize this)

## Evaluation

`packages/smart/evaluate_rollout_npz.py` evaluates inference `.npz` artifacts.
It accepts either a single `.npz` file or a directory containing `.npz` files and
writes a JSON report with summary metrics and evaluation metadata.

Example:

```bash
uv run python packages/smart/evaluate_rollout_npz.py \
  packages/smart/data/preds/batch_predictions_20260627T111519Z_92b646d6.npz
```

The default output path is:

```text
<input_stem>_metrics.json
```

Use `--include-scenario-metrics` to include the full per-scenario metric log in
the JSON output.

### What Is Currently Evaluated

The evaluator is tied to the `infer2.py` `.npz` schema. It computes:

- forecasting metrics from ground-truth and simulated xy positions;
- kinematic likelihood metrics from positions and headings when headings are
  available;
- WOMD-style interactive metrics unless `--no-interactive` is passed.

Only target/scored agents are included in the main forecasting and kinematic
metrics. In the stored SMART data, this currently means:

```text
agent_category == 3
```

The interactive metrics receive the full scene context, but the evaluated object
mask is still restricted to the same target/scored agents.

### Masking Behavior

Evaluation uses validity masks to avoid scoring missing future ground truth or
invalid model output.

During inference, `pred_valid_mask` is built from the processed scenario
`agent.valid_mask` over the future horizon. A future prediction timestep is valid
only if:

- the agent has a valid future state at that timestep;
- the agent was valid at the last historical timestep;
- the predicted point was not filtered as out of bounds.

Predicted positions more than `1000m` from the agent's last observed position are
marked invalid.

For forecasting and kinematic metrics, the evaluator builds:

```text
eval_valid_mask = gt_valid_mask & pred_valid_mask_any_rollout & target_agent
```

where:

- `gt_valid_mask` marks future timesteps with available ground truth;
- `pred_valid_mask_any_rollout` is true when at least one rollout has a valid
  prediction for that agent and timestep;
- `target_agent` is true for `agent_category == 3`.

The actual positions passed to the forecasting and kinematic metrics are already
filtered to target agents, and the mask selects the valid target-agent timesteps.

For interactive metrics:

- `valid` is the full-scene `gt_valid_mask`;
- `sim_valid` is the per-rollout `pred_valid_mask`;
- `evaluated_object_mask` is the target-agent mask.

This means interactive metrics can use non-target agents as scene context while
still reporting scores for the target/scored agents.

## Upload Metrics To Weights & Biases

After evaluation, you can upload the metrics JSON as an artifact and inspect it visually (proceed similarly for GIF outputs created during inference):

```bash
wandb artifact put \
  --name internal_test/test/metrics \
  packages/smart/data/preds/batch_predictions_20260627T111519Z_92b646d6_metrics.json
```
