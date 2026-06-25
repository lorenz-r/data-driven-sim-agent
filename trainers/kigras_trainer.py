from hydra import initialize, compose
from omegaconf import OmegaConf
from trainers.base_trainer import BaseTrainer
from packages.kigras.train import run_training
from packages.kigras.scripts.preprocess_kinematics import process
from pathlib import Path


class KigrasTrainer(BaseTrainer):
    def train(self):
        project_root = Path(__file__).resolve().parent.parent
        overrides = list(self.config["model"].get("overrides", []))
        overrides.append(f"package_root={project_root}")
        self.name = self.__class__.__name__

        with initialize(config_path="../packages/kigras/configs", version_base=None):
            cfg = compose(config_name="config", overrides=overrides)

        print(OmegaConf.to_yaml(cfg))

        # Preprocessing (NOTE: should be removed once this functionality is added to train.py)
        process(cfg)

        return run_training(cfg, wandb_logger=self.wandb_logger)
    
    def validate(self):
        print(f"Validating with {self.name}...")

    def test(self):
        print(f"Testing with {self.name}...")

    def run_inference(self, input_data):
        print(f"Running inference with {self.name}...")