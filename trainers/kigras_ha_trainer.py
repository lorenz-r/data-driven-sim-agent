from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from trainers.base_trainer import BaseTrainer
from packages.kigras_ha.train import run_training
from packages.kigras.scripts.preprocess_kinematics import process, split_filename
from pathlib import Path


class Kigras_HA_Trainer(BaseTrainer):
    """def __init__(self, config, wandb_logger=None):
    super().__init__(config, wandb_logger)
    """

    def train(self):
        print("Training with Kigras_HA_Trainer...")

        project_root = Path(__file__).resolve().parent.parent
        self.name = self.__class__.__name__

        download_size = self.config.get("data", {}).get("download_size", "")

        overrides = list(self.config["model"].get("overrides", []))
        overrides.append(f"package_root='{project_root}'")

        config_dir = str(project_root / "packages" / "kigras" / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        # The split filename encodes the action-bin warp (grid_power), so the
        # dataloader paths and the "already preprocessed?" check must be built
        # after composing the config — that is when grid_power is known.
        grid_power = cfg.model.kinematics.get("grid_power", 1.0)
        train_name = split_filename("train", size=download_size, grid_power=grid_power)
        val_name = split_filename("val", size=download_size, grid_power=grid_power)

        OmegaConf.set_struct(cfg, False)
        cfg.training.dataloader.train_path = (
            f"{project_root}/av2_data/kigras_processed/train/{train_name}"
        )
        cfg.training.dataloader.val_path = (
            f"{project_root}/av2_data/kigras_processed/val/{val_name}"
        )

        print(OmegaConf.to_yaml(cfg))

        # Skip preprocessing if the size- and grid_power-specific outputs exist
        proc_cfg = cfg.setup.preprocessing
        outputs_exist = all(
            (
                Path(proc_cfg.output[split])
                / split_filename(split, size=download_size, grid_power=grid_power)
            ).exists()
            for split in proc_cfg.output
        )
        if outputs_exist:
            print(
                f"\nPreprocessed outputs for '{download_size or 'default'}' already exist. Skipping preprocessing.\n"
            )
        else:
            process(cfg, size=download_size)

        return run_training(
            cfg,
            wandb_logger=self.wandb_logger,
            return_metrics_path=self.config.get("output", {}).get("metrics_path", ""),
        )
        # Implement your training loop here
        # You can use self.config for configuration and self.wandb_logger for logging

    def validate(self):
        print("Validating with Kigras_HA_Trainer...")
        # Implement your validation loop here

    def test(self):
        print("Testing with Kigras_HA_Trainer...")
        # Implement your testing loop here

    def run_inference(self, input_data):
        print("Running inference with Kigras_HA_Trainer...")
        # Implement your inference code here
