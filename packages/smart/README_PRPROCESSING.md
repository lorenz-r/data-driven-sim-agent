## How to preprocess AV2 scenarios for SMART

This is a (temporary) workaround for the dependency conflicts between AV2 and SMART. The main idea is to have a separate, clean environment specifically for running the preprocessing step.

- Download argoverse2 scenarios to `smart/data/raw/train`

- Run from the root directory of SMART: `chmod +x ./scripts/run_preprocess.sh && ./scripts/run_preprocess.sh`

- This will create a local environment `.venv_av2` (unless it aready exists) and execute the preprocessing script `data_preprocess_av2.py` aftwerwards

Configure paths in `run_preprocess.sh` as necessary. 

