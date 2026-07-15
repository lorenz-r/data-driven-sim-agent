# AV2 Evaluation Utilities

This folder contains small scripts for inspecting and cleaning AV2-style
scenario parquet files.

## `av2-cleanup.py`

`av2-cleanup.py` removes trajectories that would otherwise overlap with another
agent. It is intended as a preprocessing step for raw or generated AV2 scenario
parquets before downstream evaluation or visualization.

The script reuses `count-collisions.py` for rounded-footprint collision
detection. For each scenario it:

1. Detects unordered collision pairs and stores the first collision timestep.
2. Builds a temporal collision graph where agents are nodes and collisions are
   edges.
3. Greedily selects the currently most-colliding removable agent.
4. Removes that agent from its first active collision timestep onward.
5. Deletes the resolved edges and repeats until no active collision edge remains,
   no removable agent is left, or `--max-removals` is reached.

By default every object type is removable. Pass `--removable-types` with an
explicit allow-list if a run should only remove specific object classes.

### Examples

Inspect planned removals without writing cleaned parquet:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val/scenario_id/scenario_scenario_id.parquet \
  /tmp/cleaned_scenario.parquet \
  --dry-run \
  --details-csv /tmp/cleanup_decisions.csv
```

Clean and verify one scenario:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val/scenario_id/scenario_scenario_id.parquet \
  data/cleaned/val/scenario_id/scenario_scenario_id.parquet \
  --verify \
  --details-csv data/cleaned/cleanup_decisions.csv
```

Clean a directory tree in parallel and copy map JSON sidecars:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val \
  data/cleaned/val \
  --workers 8 \
  --chunksize 8 \
  --copy-sidecars \
  --details-csv data/cleaned/cleanup_decisions.csv
```

Clean a directory tree and create visualization-ready SMART-style output
folders:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val \
  data/cleanup_visualizations \
  --visualize \
  --workers 4 \
  --details-csv data/cleanup_visualizations/cleanup_decisions.csv
```

With `--visualize`, each scenario is written as:

```text
data/cleanup_visualizations/<scenario_id>/
  scenario_<scenario_id>.GIF
  scenario_<scenario_id>_original.GIF
  scenario_<scenario_id>.parquet
  scenario_<scenario_id>_original.parquet
  log_map_archive_*.json
```

The cleaned scenario keeps the original filename and internal scenario id. The
source parquet is copied with `_original` added to the filename. This keeps the
scenario folder compatible with SMART's raw AV2 input discovery, which expects
the scenario parquet and map JSON directly inside each scenario directory.

Restrict removals to vehicles and buses:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val \
  data/cleaned/val \
  --removable-types vehicle bus
```

Restrict collision detection itself to vehicles and buses while still allowing
all detected collision participants to be removed:

```bash
.venv/bin/python eval-utilities/av2-cleanup.py \
  data/raw/val \
  data/cleaned/val \
  --collision-types vehicle bus
```

### Useful Options

- `--verify`: reruns collision counting on cleaned outputs. This is useful for
  audits, but roughly doubles collision-detection work.
- `--visualize`: writes SMART-style scenario folders and GIFs for the cleaned
  and original scenario.
- `--visualization-fps`: controls the GIF frame rate used by `--visualize`.
- `--remove-previous-step`: starts removal at `first_collision_timestep - 1`
  instead of the collision timestep.
- `--max-removals`: caps removals per scenario.
- `--no-numba`: disables the numba collision kernel from `count-collisions.py`.
- `--workers`: parallelizes across independent scenario parquet files.

The script refuses to overwrite the input parquet or write a directory output on
top of the input directory.
