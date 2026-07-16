from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trainers.base_trainer import BaseTrainer


def _has_split_outputs(proc_cfg) -> bool:
    for split in proc_cfg.output.keys():
        output_dir = Path(proc_cfg.output[split])
        if not output_dir.exists():
            return False
        if not any(output_dir.glob(f"{split}_split*.npy")):
            return False
    return True


class Kigras_AM_Trainer(BaseTrainer):
    def train(self):
        print("Training with Kigras_AM_Trainer...")

        project_root = Path(__file__).resolve().parent.parent
        package_root = project_root / "packages" / "kigras_am"

        from packages.kigras_am.scripts.preprocess_kinematics import process
        from packages.kigras_am.train import run_training

        self.name = self.__class__.__name__

        # Keep AM caches separate from the older KiGRAS/HA schema. User-provided
        # Hydra overrides after "--" still win because they are appended last.
        overrides = [
            f"package_root='{project_root}'",
            f"processed_train='{project_root / 'av2_data' / 'kigras_am_processed' / 'train'}'",
            f"processed_val='{project_root / 'av2_data' / 'kigras_am_processed' / 'val'}'",
        ]
        overrides.extend(self.config["model"].get("overrides", []))

        config_dir = str(package_root / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        OmegaConf.set_struct(cfg, False)
        print(OmegaConf.to_yaml(cfg))

        force_preprocess = self.config.get("data", {}).get("download_first", False)
        if force_preprocess or not _has_split_outputs(cfg.setup.preprocessing):
            process(cfg)
        else:
            print("\nPreprocessed AM outputs already exist. Skipping preprocessing.\n")

        return run_training(
            cfg,
            wandb_logger=self.wandb_logger,
            return_metrics_path=self.config.get("output", {}).get("metrics_path", ""),
        )

    def validate(self):
        print(f"Validating with {self.name}...")

    def test(self):
        print(f"Testing with {self.name}...")

    def run_inference(self, input_data):
        print(f"Running inference with {self.name}...")
