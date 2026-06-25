from trainers.smart_trainer import SmartTrainer
from trainers.kigras_trainer import KigrasTrainer

TRAINER_REGISTRY = {"smart": SmartTrainer, "kigras": KigrasTrainer}
