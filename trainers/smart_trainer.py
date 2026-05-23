import sys

sys.path.append("../")  # Add the parent directory to the system path
from packages.smart.train import run_training
from trainers.base_trainer import BaseTrainer


class SmartTrainer(BaseTrainer):
    def __init__(self, config, wandb_logger=None):
        super().__init__(config, wandb_logger)

    def train(self):
        print("Training with SmartTrainer...")
        return run_training(self.config, wandb_logger=self.wandb_logger)

    def validate(self):
        print("Validating with SmartTrainer...")

    def test(self):
        print("Testing with SmartTrainer...")

    def run_inference(self, input_data):
        print("Running inference with SmartTrainer...")
