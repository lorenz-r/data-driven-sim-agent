import torch
from torch import Tensor
from typing import TypeAlias

Num: TypeAlias = Tensor | float


def bounding_box(
    cx: Tensor, cy: Tensor, cos_h: Num, sin_h: Num, length: Num, width: Num
) -> Tensor:
    """Axis-aligned -> rotated bounding box corners.

    Args:
        cx, cy:   (...,)  centre position.
        cos_h,
        sin_h:    (...,)  heading unit vector.
        length,
        width:    (...,)  agent dimensions - float or Tensor.

    Returns:
        corners:  (..., 4, 2) four corners in clockwise order.
    """
    if isinstance(length, (int, float)):
        length = torch.as_tensor(length, dtype=cx.dtype, device=cx.device)
    if isinstance(width, (int, float)):
        width = torch.as_tensor(width, dtype=cx.dtype, device=cx.device)
    if isinstance(cos_h, (int, float)):
        cos_h = torch.as_tensor(cos_h, dtype=cx.dtype, device=cx.device)
    if isinstance(sin_h, (int, float)):
        sin_h = torch.as_tensor(sin_h, dtype=cx.dtype, device=cx.device)
    hl, hw = length / 2, width / 2
    local = torch.stack(
        [
            torch.stack([hl, hw], dim=-1),
            torch.stack([hl, -hw], dim=-1),
            torch.stack([-hl, -hw], dim=-1),
            torch.stack([-hl, hw], dim=-1),
        ],
        dim=-2,
    )

    rotated = torch.stack(
        [
            local[..., 0] * cos_h[..., None] - local[..., 1] * sin_h[..., None],
            local[..., 0] * sin_h[..., None] + local[..., 1] * cos_h[..., None],
        ],
        dim=-1,
    )

    return rotated + torch.stack([cx, cy], dim=-1).unsqueeze(-2)
