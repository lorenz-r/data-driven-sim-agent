from core.losses.ntp import build_ntp_classification_loss
from core.losses.spatial_label_smoothing import SpatialAwareLabelSmoothingLoss

__all__ = [
    "SpatialAwareLabelSmoothingLoss",
    "build_ntp_classification_loss",
]
