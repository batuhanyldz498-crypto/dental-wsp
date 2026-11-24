"""Compact WSP architectures with matched NumPy oracles and optional PyTorch modules."""

from wsp_core.architectures.cls_lenet2d import (
    cls_lenet2d_logits,
    cls_lenet2d_margin_ablation,
)
from wsp_core.architectures.seg_wsp82_2d import (
    basis_1d,
    seg_wsp82_2d_logits,
    seg_wsp82_2d_position_logits,
    seg_wsp82_2d_without_grid,
)
from wsp_core.architectures.seg_wsp82_3d import (
    seg_wsp82_3d_logits,
    seg_wsp82_3d_position_logits,
    seg_wsp82_3d_without_grid,
)

__all__ = [
    "basis_1d",
    "seg_wsp82_2d_logits",
    "seg_wsp82_2d_position_logits",
    "seg_wsp82_2d_without_grid",
    "seg_wsp82_3d_logits",
    "seg_wsp82_3d_position_logits",
    "seg_wsp82_3d_without_grid",
    "cls_lenet2d_logits",
    "cls_lenet2d_margin_ablation",
]
