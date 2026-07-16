# KiGRAS-AM

`kigras_am` is a standalone package for sparse, observer-centric motion forecasting on Argoverse 2. The model predicts discrete kinematic action tokens for each scored or focal agent, then can roll those tokens forward through a CTRA kinematic model to produce trajectories.

## Contents

- [Pipeline Overview](#pipeline-overview)
- [Data Representation](#data-representation)
- [Dataset](#dataset)
- [Kinematic Preprocessing](#kinematic-preprocessing)
- [Batching](#batching)
- [Batch Processing](#batch-processing)
- [Observer-Local Transformation](#observer-local-transformation)
- [Normalization](#normalization)
- [Standardized Observer Indices](#standardized-observer-indices)
- [Model](#model)
- [Encoder](#encoder)
- [Decoder](#decoder)
- [Training Objective](#training-objective)
- [Closed-Loop Rollout](#closed-loop-rollout)
- [Installation](#installation)
- [Configuration](#configuration)
- [Practical Commands](#practical-commands)

## Pipeline Overview

The package pipeline is:

```text
AV2 raw scenarios
  -> ArgoDataset raw parsing
  -> kinematic preprocessing into action tokens
  -> ArgoDataset processed loading
  -> collate_fn sparse batch flattening
  -> process_sparse_batch observer-local edge features
  -> SceneEncoder MLP + sparse observer-frame attention
  -> ActionDecoder transformer decoder
  -> action-token logits
  -> optional closed-loop CTRA rollout
```

Important files:

| File | Role |
| --- | --- |
| `kigras_am/dataloader/dataset.py` | AV2 parsing, processed dataset loading, sparse `collate_fn`. |
| `scripts/preprocess_kinematics.py` | Converts trajectories into discrete CTRA action-token labels. |
| `kigras_am/utils/train_utils.py` | Converts collated batches into observer-local model inputs. |
| `kigras_am/model/scene_encoder.py` | Encodes sparse agent/map context into one scene token per observer timestep with sparse attention. |
| `kigras_am/model/decoder.py` | Transformer action decoder over observer scene tokens and previous actions. |
| `kigras_am/model/kigras.py` | LightningModule that wires processing, encoder, decoder, loss, training, and rollout. |
| `kigras_am/kinematics/ctra.py` | CTRA action quantization, inverse action recovery, and forward rollout. |

## Data Representation

The package uses AV2 motion forecasting scenarios with 50 observed timesteps and 60 prediction timesteps.

| Constant | Value | Meaning |
| --- | --- | --- |
| `OBS_LEN` | `50` | Historical timesteps. |
| `PRED_LEN` | `60` | Future prediction timesteps. |
| `CURRENT_STEP` | `49` | Last observed timestep. |
| `ACTION_LEN` | `109` | Number of transitions from 110 states. |

Agent states are represented as:

```text
[x, y, cos_heading, sin_heading, vx, vy, valid]
```

with shape:

```text
[N, 110, 7]
```

Headings are stored as unit vectors, not scalar yaw angles. This makes frame transforms use direct vector rotation instead of converting back and forth through angles.

The model predicts only observers. Observers are AV2 tracks whose category is scored or focal:

```text
TrackCategory.SCORED_TRACK -> category 2
TrackCategory.FOCAL_TRACK  -> category 3
```

All non-static tracks are kept as possible context agents. Static/background/construction/riderless-bicycle tracks are filtered out by `ArgoDataset`.

## Dataset

`ArgoDataset` supports two modes:

| Mode | Constructor argument | What happens |
| --- | --- | --- |
| Raw | `ArgoDataset(raw_path=...)` | Loads AV2 parquet scenarios and map JSON files on demand. |
| Processed | `ArgoDataset(processed_path=...)` | Loads a saved numpy split such as `train_split.npy` or `val_split.npy`. |

Raw AV2 folders are expected to contain per-scenario directories with files like:

```text
scenario_<scenario_id>.parquet
log_map_archive_<scenario_id>.json
```

For each raw scenario, `ArgoDataset` builds:

| Key | Shape | Meaning |
| --- | --- | --- |
| `agents.states` | `[N, 110, 7]` | Agent states in global coordinates. |
| `agents.agent_type_ids` | `[N]` | Object type IDs for embeddings. |
| `agents.agent_dimensions` | `[N, 3]` | Heuristic object dimensions. |
| `agents.track_ids` | list length `N` | Original AV2 track IDs. |
| `agents.track_categories` | `[N]` | AV2 track category IDs. |
| `graph.position` | `[M, 2]` | Lane centerline node positions. |
| `graph.direction_vector` | `[M, 2]` | Direction to the next centerline point. |
| `graph.lane_type` | `[M]` | Lane type IDs. |
| `graph.edge_index` | `[E_lane, 2]` before collation | Explicit lane graph topology. |
| `graph.edge_type` | `[E_lane]` | `0=forward`, `1=left`, `2=right`. |
| `scenario_id` | scalar/string | AV2 scenario ID. |
| `focal_track_id` | scalar/string | AV2 focal track ID. |

The lane graph is built from AV2 lane segments by converting centerlines into nodes and adding directed edges. Forward edges connect consecutive centerline points and lane successors. Left/right edges connect nearby nodes in adjacent lanes. The current encoder uses the flattened map nodes and their lane types; the explicit lane topology is preserved for visualization, debugging, and future model extensions.

## Kinematic Preprocessing

Training requires discrete action-token labels. These are created offline by `scripts/preprocess_kinematics.py`.

For the default CTRA setup, each state is converted from:

```text
[x, y, cos_heading, sin_heading, vx, vy, valid]
```

to:

```text
[x, y, cos_heading, sin_heading, speed]
```

`CTRAKinematics.solve(...)` recovers continuous actions for the model-relevant state window:

```text
states 48..109 -> actions 48->49, 49->50, ..., 108->109
```

Each action has layout:

```text
[acceleration, yaw_rate]
```

Then it quantizes each action onto a 2D grid. With the default config:

```text
num_accel_bins = 63
num_yaw_bins   = 63
vocab_size     = 63 * 63 = 3969
```

The action token ID is:

```text
token_id = accel_idx * NUM_YAW + yaw_idx
```

The preprocessed sample adds:

```text
agents.action_tokens: [N, 109]
```

Invalid transitions are stored as `-1`. A transition is valid only when both endpoint states are valid.

Only action-token indices `48..108` are populated by preprocessing. Other indices remain `-1`. Token `48` is kept because training uses it as the previous-action context for the first predicted action; tokens `49..108` are the supervised prediction targets.

The preprocessing output is a numpy file in the split directory:

```text
<processed_root>/<split>/<split>_split*.npy
```

For example:

```text
av2_data/kigras_am_processed/train/train_split*.npy
av2_data/kigras_am_processed/val/val_split*.npy
```

## Batching

`collate_fn` in `kigras_am/dataloader/dataset.py` builds sparse batches from scenarios with different numbers of agents and map nodes. It does not pad each scenario to a common number of agents or map nodes. Instead, it flattens all scenarios in the DataLoader batch and records index tensors.

Dimension names used throughout the package:

| Name | Meaning |
| --- | --- |
| `V_total` | Total agents across all scenarios in the batch. |
| `M_total` | Total map nodes across all scenarios in the batch. |
| `O_total` | Total observer agents across all scenarios in the batch. |
| `E_agent` | Total source-agent to observer context edges. |
| `E_map` | Total source-map-node to observer context edges. |
| `E_lane` | Total lane-topology edges. |

The collated batch contains:

| Batch key | Shape | Meaning |
| --- | --- | --- |
| `agents.states` | `[V_total, 110, 7]` | Flattened agent states. |
| `agents.type_ids` | `[V_total]` | Object type IDs. |
| `agents.dimensions` | `[V_total, 3]` | Agent dimensions. |
| `agents.categories` | `[V_total]` | Track category IDs. |
| `agents.valid` | `[V_total, 110]` | State validity mask. |
| `agents.batch_idx` | `[V_total]` | Scenario index per flattened agent. |
| `agents.action_tokens` | `[V_total, 109]` | Optional preprocessed action labels. |
| `graph.position` | `[M_total, 2]` | Flattened map-node positions. |
| `graph.direction_vector` | `[M_total, 2]` | Flattened map-node direction vectors. |
| `graph.lane_type` | `[M_total]` | Lane type per map node. |
| `graph.edge_index` | `[2, E_lane]` | Flattened lane graph topology. |
| `graph.edge_type` | `[E_lane]` | Lane graph edge types. |
| `observers.agent_idx` | `[O_total]` | Global flattened agent index per observer row. |
| `observers.local_idx` | `[O_total]` | Per-scenario agent index per observer row. |
| `observers.batch_idx` | `[O_total]` | Scenario index per observer row. |
| `observers.action_tokens` | `[O_total, 109]` | Observer-only action labels. |
| `observers.action_valid` | `[O_total, 109]` | Observer-only action validity mask. |
| `context_edges.agent_to_observer` | `[2, E_agent]` | `[source_agent_idx, observer_agent_idx]` context edges. |
| `context_edges.map_to_observer` | `[2, E_map]` | `[source_map_node_idx, observer_agent_idx]` context edges. |

For each scenario in the batch, `collate_fn` creates all-to-all context between sources and observers:

```text
every agent in the scenario -> every observer in the same scenario
every map node in the scenario -> every observer in the same scenario
```

Example with 3 agents and 1 observer in one scenario:

```text
source agents: [0, 1, 2]
observer:      [0]

agent_to_observer = [
    [0, 1, 2],
    [0, 0, 0],
]
```

When multiple scenarios are collated, indices are shifted by running global offsets so they index into the flattened tensors.

## Batch Processing

`process_sparse_batch` in `kigras_am/utils/train_utils.py` turns a collated batch into model-facing sparse tensors.

Default call:

```python
processed = process_sparse_batch(batch)
```

uses:

```text
start_t = CURRENT_STEP = 49
horizon = PRED_LEN = 60
```

so the scene encoder sees states at timesteps:

```text
49, 50, ..., 108
```

The target action tokens correspond to transitions:

```text
49->50, 50->51, ..., 108->109
```

### Observer-Local Transformation

`transform_batch_sparse` receives flattened trajectories and context edges. For agent edges, it gathers:

```python
tgt_idx, obs_idx = agent_to_observer_edges[0], agent_to_observer_edges[1]
obs_states = trajectories_flat[obs_idx]
tgt_states = trajectories_flat[tgt_idx]
```

Both `tgt_idx` and `obs_idx` are global flattened indices into `agents.states`. The edge features are still edge rows; they have not been pooled yet.

For each source-agent to observer edge, it first builds a per-timestep radius/validity mask in global coordinates. It only transforms rows that are valid for that specific observer timestep:

```text
local_x, local_y
local_cos_heading, local_sin_heading
local_vx, local_vy
valid
```

The output shape is:

```text
agent_context: [N_agent_context, 6]
agent_group_idx: [N_agent_context]
```

For map edges, it similarly filters by observer validity and map radius before transforming the static map node position and direction vector into the observer frame:

```text
map_context: [N_map_context, 4]
map_group_idx: [N_map_context]
```

The map feature dimension stores:

```text
map_context[:, 0:2] = local map-node position [x, y]
map_context[:, 2:4] = local map direction [dx, dy]
```

The group index is `observer_row * T + timestep`, which lets scatter attention collect variable-size neighborhoods without padding back to `[E, T, D]`.

### Normalization

The processor normalizes local features before the model sees them:

| Feature | Scale |
| --- | --- |
| Agent position | `POSITION_SCALE = 100.0` |
| Agent velocity | `VELOCITY_SCALE = 30.0` |
| Map position | `POSITION_SCALE = 100.0` |
| Map direction | `MAP_DIRECTION_SCALE = 10.0` |

Heading unit vectors are not scaled. Validity flags are used to create sparse rows and are not passed to the encoder as features.

### Standardized Observer Indices

The context edges store observer endpoints as global flattened agent indices. The model needs dense observer rows from `0` to `O_total - 1`. `_standardize_observer_indices` converts edge-level global observer IDs into standardized observer rows.

Example:

```text
observer_global_idx       = [4, 10]
edge_observer_global_idx  = [10, 4, 10]
standardized edge indices = [1, 0, 1]
```

The standardized observer rows are combined with timestep ids to build sparse attention groups:

```text
group_idx = observer_row * horizon + timestep
```

These group ids tell scatter attention which observer-time query each sparse context row belongs to.

The processed batch contains:

| Processed key | Shape | Meaning |
| --- | --- | --- |
| `agent_context` | `[N_agent_context, 6]` | Normalized source-agent sparse rows in observer frames. |
| `map_context` | `[N_map_context, 4]` | Normalized map sparse rows in observer frames. |
| `agent_group_idx` | `[N_agent_context]` | Observer-time attention group per agent row. |
| `map_group_idx` | `[N_map_context]` | Observer-time attention group per map row. |
| `agent_type_ids` | `[N_agent_context]` | Source-agent type per sparse agent row. |
| `map_lane_type` | `[N_map_context]` | Lane type per sparse map row. |
| `map_left_mark_type` | `[N_map_context]` | Left boundary mark type per sparse map row. |
| `map_right_mark_type` | `[N_map_context]` | Right boundary mark type per sparse map row. |
| `map_is_intersection` | `[N_map_context]` | Intersection flag per sparse map row. |
| `map_width` | `[N_map_context]` | Normalized lane width per sparse map row. |
| `observer_idx` | `[O_total]` | Global flattened agent index per observer row. |
| `observer_type_ids` | `[O_total]` | Observer object type per observer row. |
| `observer_valid` | `[O_total, T]` | Observer validity per scene timestep. |
| `action_tokens` | `[O_total, 109]` | Observer action-token labels. |
| `action_valid` | `[O_total, 109]` | Observer action-token validity. |
| `num_observers` | scalar | `O_total`. |
| `horizon` | scalar | Scene horizon `T`. |
| `normalization` | dict | Scales used for local features. |

## Model

`KiGRAS` in `kigras_am/model/kigras.py` is a PyTorch Lightning module. Its forward pass is:

```python
processed = process_sparse_batch(batch)
scene_tokens = self.encoder(processed)
logits = self.decoder(scene_tokens, prev_actions=..., prev_actions_valid=..., scene_valid=...)
```

The output is:

```text
logits: [O_total, 60, vocab_size]
```

With the default action grid, `vocab_size = 3969`.

## Encoder

`SparseSceneEncoder` in `kigras_am/model/scene_encoder.py` turns sparse observer-local context into one dense scene token per observer per timestep.

```text
agent context edges + map context edges -> scene_tokens [O_total, T, d_model]
```

The encoder embeds sparse context rows and uses scatter attention to aggregate each variable-size observer-time neighborhood without padding.

Encoder inputs:

| Input | Shape | Meaning |
| --- | --- | --- |
| `agent_context` | `[N_agent_context, 6]` | Valid source-agent rows in each observer frame. |
| `map_context` | `[N_map_context, 4]` | Valid map-node position/direction rows in each observer frame. |
| `agent_group_idx` | `[N_agent_context]` | Observer-time group per agent row. |
| `map_group_idx` | `[N_map_context]` | Observer-time group per map row. |
| `agent_type_ids` | `[N_agent_context]` | Source-agent semantic type per sparse row. |
| `map_lane_type` | `[N_map_context]` | Lane type per sparse map row. |
| `map_left_mark_type` | `[N_map_context]` | Left boundary mark type per sparse map row. |
| `map_right_mark_type` | `[N_map_context]` | Right boundary mark type per sparse map row. |
| `map_is_intersection` | `[N_map_context]` | Intersection flag per sparse map row. |
| `map_width` | `[N_map_context]` | Lane width per sparse map row. |
| `observer_type_ids` | `[O_total]` | Semantic type of each predicted observer. |
| `observer_valid` | `[O_total, T]` | Validity mask for observer timesteps. |

Encoder behavior:

1. Sparse agent rows are embedded from local kinematic features and source-agent type.
2. Sparse map rows are embedded from local map position/direction, lane type, lane-boundary marks, intersection flag, and lane width.
3. `agent_group_idx` and `map_group_idx` assign each context row to an observer-time query.
4. Sparse scatter attention updates one token per observer timestep.
5. Invalid observer timesteps are masked to zero.

The key idea is that the encoder preserves sparse context rows until attention. It never pads each observer to a fixed number of neighboring agents or map nodes.

Scatter attention is grouped by flattened observer-time ids. If four context rows have group indices:

```text
index = [0, 1, 1, 0]
```

then the pooled output groups them as:

```text
observer 0 receives rows 0 and 3
observer 1 receives rows 1 and 2
```

For every observer and timestep, the encoder attends only to sparse rows assigned to that group. Invalid or out-of-radius edge/timestep pairs were never materialized as encoder rows. The output shape is always:

```text
scene_tokens: [O_total, T, d_model]
```

## Decoder

`ActionDecoder` in `kigras_am/model/decoder.py` maps the encoder's scene tokens and previous action tokens to logits over the discrete action vocabulary.

```text
scene_tokens [O_total, T, d_model] + prev_actions [O_total, T]
  -> logits [O_total, T, vocab_size]
```

This is the temporal autoregressive attention part of the model. Observer rows act like the batch dimension and the sequence dimension is time.

Decoder inputs:

| Argument | Shape | Meaning |
| --- | --- | --- |
| `scene_tokens` | `[O_total, T, d_model]` | Encoder memory. |
| `prev_actions` | `[O_total, T]` | Teacher-forced previous action tokens or rollout history. |
| `prev_actions_valid` | `[O_total, T]` | Validity mask for previous action tokens. |
| `scene_valid` | `[O_total, T]` | Validity mask for scene memory. |

Decoder behavior:

1. Previous action tokens are embedded. Invalid positions are replaced by a start-of-sequence token.
2. Positional information is added so the decoder knows each token's timestep.
3. Causal self-attention lets each timestep read earlier action history without seeing future action tokens.
4. Causal cross-attention lets each timestep read the encoded scene up to that timestep without seeing future scene tokens.
5. Invalid scene timesteps are masked out of cross-attention.
6. The decoder output is projected to action-token logits.

The decoder does not attend over raw agents or raw map nodes. By the time data reaches the decoder, the scene has already been summarized into observer-centric tokens by the encoder.

## Training Objective

`KiGRAS.forward` prepares teacher-forced action inputs from the observer action labels.

It slices:

```python
action_window = action_tokens[:, CURRENT_STEP - 1 : CURRENT_STEP + PRED_LEN]
```

With defaults, this is:

```text
action_tokens[:, 48:109]
```

which has length `61`.

Then:

```text
prev_actions    = action_window[:, :-1]  -> token indices 48..107
target_actions  = action_window[:, 1:]   -> token indices 49..108
```

The first predicted future action is transition `49->50`. Its previous-action input is transition `48->49`, the action immediately before the prediction horizon.

Targets are masked with `target_actions_valid`. Invalid targets are replaced by `-1`, and the loss is:

```python
nn.CrossEntropyLoss(ignore_index=-1)
```

The model logs train/validation loss with `batch_size=processed["num_observers"]`.

## Closed-Loop Rollout

`KiGRAS.rollout(batch, kinematic_model)` performs autoregressive inference.

For each future timestep:

```text
t = 49, 50, ..., 108
```

it:

```text
1. Processes the current rollout batch with horizon=1.
2. Encodes the current observer-local scene token.
3. Appends that scene token to the accumulated memory.
4. Decodes the next action token from previous predicted/known actions.
5. Converts the token to [acceleration, yaw_rate].
6. Advances observer states one step with CTRA forward kinematics.
7. Writes the predicted next observer states back into the batch.
```

This makes rollout closed-loop: later scene encodings see previously predicted observer states, not ground-truth future observer states.

The rollout return value includes:

```text
tokens: [O_total, 60]
states: [O_total, 60, 7]
xy:     [O_total, 60, 2]
```

## Installation

From the repository root:

```bash
pip install -e packages/kigras_am
```

The package depends on PyTorch, PyTorch Lightning, AV2, NumPy, pandas, SciPy, Hydra/OmegaConf via the training scripts, and related project packages listed in `pyproject.toml`.

## Configuration

The main Hydra config is:

```text
configs/config.yaml
```

Default paths live in:

```text
configs/paths.yaml
```

Default path keys:

| Key | Default |
| --- | --- |
| `raw_train` | `${package_root}/av2_data/raw_av2/train` |
| `raw_val` | `${package_root}/av2_data/raw_av2/val` |
| `processed_train` | `${package_root}/av2_data/kigras_am_processed/train` |
| `processed_val` | `${package_root}/av2_data/kigras_am_processed/val` |
| `checkpoint_save_path` | `${package_root}/checkpoints` |

Model architecture defaults live in:

```text
configs/model/architecture/kigras_default_arch.yaml
```

Important defaults:

| Key | Default | Meaning |
| --- | --- | --- |
| `embedding_dim` | `32` | `d_model`. |
| `num_head` | `4` | Transformer decoder heads. |
| `encoder_num_layers` | `3` | Sparse observer-frame attention depth. |
| `encoder_ff_dim` | `256` | Encoder sparse-attention feed-forward width. |
| `decoder_num_layers` | `3` | Transformer decoder depth. |
| `decoder_ff_dim` | `256` | Transformer feed-forward width. |
| `num_action_bins` | `63` | Bins per action dimension. |
| `num_actions` | `2` | Acceleration and yaw rate. |

The model vocabulary size is computed as:

```text
vocab_size = num_action_bins ** num_actions
```

With defaults:

```text
63 ** 2 = 3969
```

## Practical Commands

Run these from the repository root with `uv run`.

### Train From No Local Data

This downloads AV2, preprocesses AM data, and trains:

```bash
uv run train.py --model kigras_am --download-first --download-size small
```

Tiny smoke run:

```bash
uv run train.py --model kigras_am --download-first --download-size single -- training.trainer.max_epochs=1
```

Outputs use the AM-specific processed-data folder:

```text
av2_data/raw_av2/{train,val}
av2_data/kigras_am_processed/{train,val}
checkpoints/last.ckpt
```

Control preprocessing parallelism with `PREPROCESS_WORKERS`:

```bash
PREPROCESS_WORKERS=8 uv run train.py --model kigras_am --download-first --download-size small
```

### Generate AM Prediction NPZ

`infer.py` expects AM preprocessed data, not raw AV2. It writes a SMART-compatible batch archive.

```bash
uv run python packages/kigras_am/infer.py \
  --input av2_data/kigras_am_processed/val \
  --checkpoint checkpoints/last.ckpt \
  --output outputs/kigras_am_preds \
  --raw-data-folder av2_data/raw_av2/val \
  --device auto \
  --batch-size 1 \
  --no-visualize
```

Smoke inference on a few scenarios:

```bash
uv run python packages/kigras_am/infer.py \
  --input av2_data/kigras_am_processed/val \
  --checkpoint checkpoints/last.ckpt \
  --output outputs/kigras_am_preds \
  --raw-data-folder av2_data/raw_av2/val \
  --limit 10 \
  --no-visualize
```

The main output is:

```text
outputs/kigras_am_preds/batch_predictions_<run_id>.npz
outputs/kigras_am_preds/inference_summary_<run_id>.json
```

AM predicts only observer agents (`SCORED_TRACK` and `FOCAL_TRACK`). The NPZ keeps full-scene ground truth/context, but non-observer prediction rows are invalid via `pred_valid_mask=false`.

### Evaluate With SMART Unified NPZ Metrics

Use the common SMART evaluator for both SMART and KiGRAS batch NPZ files:

```bash
uv run python packages/smart/evaluate_rollout_npz.py \
  outputs/kigras_am_preds/batch_predictions_<run_id>.npz \
  --output outputs/kigras_am_preds/metrics.json \
  --raw-data-folder av2_data/raw_av2/val \
  --workers 8
```

Fast forecast-only evaluation:

```bash
uv run python packages/smart/evaluate_rollout_npz.py \
  outputs/kigras_am_preds/batch_predictions_<run_id>.npz \
  --output outputs/kigras_am_preds/metrics.json \
  --no-map \
  --no-interactive
```

The evaluator scores rows with `agent_category == 3`; AM sets that only for observer agents.
