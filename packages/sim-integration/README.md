sim-integration
================

This package provides a minimal integration runner to convert a local argoverse-style
parquet containing sequence annotations into the features DataFrame expected by the
argoverse-forecasting baselines, run a baseline (constant-velocity by default), and
optionally evaluate it.

Usage (from repository root):

    python3 packages/sim-integration/run_integration.py --parquet ~/ddsa_other/av2_mf_focal_test_annotations.parquet --run_eval

Notes:
- The runner currently implements a conservative pipeline and supports the `const_vel` baseline only.
- The parquet must contain per-sequence rows with track_id, x and y columns (common names are supported);
  the runner will attempt to infer column names.
- If the parquet contains future frames (obs_len + pred_len), a ground-truth pickle will be created and
  used by the eval helper.
- Add other baselines (nn, lstm) in sim_integration/runner.py if needed.
