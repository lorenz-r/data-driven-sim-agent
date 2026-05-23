from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

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
