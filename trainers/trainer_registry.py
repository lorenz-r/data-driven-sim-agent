from trainers.smart_trainer import SmartTrainer
from trainers.kigras_trainer import KigrasTrainer
from trainers.kigras_ha_trainer import Kigras_HA_Trainer
from trainers.kigras_am_trainer import Kigras_AM_Trainer

TRAINER_REGISTRY = {
    "smart": SmartTrainer,
    "kigras": KigrasTrainer,
    "kigras_ha": Kigras_HA_Trainer,
    "kigras_am": Kigras_AM_Trainer,
}
