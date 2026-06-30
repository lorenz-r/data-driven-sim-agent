from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from trainers.base_trainer import BaseTrainer
from packages.kigras_ha.train import run_training
from packages.kigras.scripts.preprocess_kinematics import process
from pathlib import Path


class Kigras_HA_Trainer(BaseTrainer):
    """def __init__(self, config, wandb_logger=None):
    super().__init__(config, wandb_logger)
    """

    def train(self):
        print("Training with Kigras_HA_Trainer...")

        project_root = Path(__file__).resolve().parent.parent
        overrides = list(self.config["model"].get("overrides", []))
        overrides.append(f"package_root='{project_root}'")
        self.name = self.__class__.__name__

        config_dir = str(project_root / "packages" / "kigras" / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        print(OmegaConf.to_yaml(cfg))

        # Preprocessing: skip if output .npy files already exist (e.g. preprocessed externally)
        proc_cfg = cfg.setup.preprocessing
        outputs_exist = all(
            (Path(proc_cfg.output[split]) / f"{split}_split.npy").exists()
            for split in proc_cfg.output
        )
        if outputs_exist:
            print("Preprocessed outputs already exist. Skipping preprocessing.")
        if not outputs_exist:
            process(cfg)

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
