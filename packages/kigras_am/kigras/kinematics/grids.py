import numpy as np
import torch


def power_spaced_bins_np(max_val: float, num: int, power: float = 1.0) -> np.ndarray:
    """Symmetric bin centers on [-max_val, max_val], finer near 0 when power > 1.

    u = linspace(-1, 1, num);  bins = max_val * sign(u) * |u|**power
    power == 1.0 exactly reproduces np.linspace(-max_val, max_val, num).

    Built by mirroring the positive half so the grid is exactly symmetric,
    hits +-max_val exactly, and (for odd num) contains an exact 0.0 center.
    """
    if power <= 0:
        raise ValueError(f"grid_power must be > 0, got {power}")
    if power == 1.0:
        return np.linspace(-max_val, max_val, num)
    half = num // 2
    if num % 2:
        pos = max_val * np.linspace(0.0, 1.0, half + 1) ** power
        return np.concatenate([-pos[:0:-1], pos])
    pos = max_val * np.linspace(1.0 / (num - 1), 1.0, half) ** power
    return np.concatenate([-pos[::-1], pos])


def power_spaced_bins(max_val: float, num: int, power: float = 1.0) -> torch.Tensor:
    """Torch version of power_spaced_bins_np; power == 1.0 exactly reproduces torch.linspace."""
    if power <= 0:
        raise ValueError(f"grid_power must be > 0, got {power}")
    if power == 1.0:
        return torch.linspace(-max_val, max_val, num)
    half = num // 2
    if num % 2:
        pos = max_val * torch.linspace(0.0, 1.0, half + 1) ** power
        return torch.cat([-pos.flip(0)[:-1], pos])
    pos = max_val * torch.linspace(1.0 / (num - 1), 1.0, half) ** power
    return torch.cat([-pos.flip(0), pos])
