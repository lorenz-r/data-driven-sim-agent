from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config, wandb_logger=None):
        self.config = config
        self.wandb_logger = wandb_logger

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def run_inference(self, input_data):
        pass
