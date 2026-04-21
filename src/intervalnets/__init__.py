"""Interval arithmetic utilities for neural network evaluation."""

from .affine import AffineTensor
from .interval import Interval

__all__ = ["Interval", "AffineTensor"]

try:
    from .pytorch import (
        IntervalAdd,
        IntervalCat,
        IntervalTensor,
        enable_interval_eval,
        interval_forward,
        interval_forward_refine,
    )
    from .affine_pytorch import affine_relu_transform, affine_sigmoid_transform, affine_tanh_transform
except ImportError:  # pragma: no cover - optional dependency
    pass
else:
    __all__.extend(
        [
            "IntervalTensor",
            "IntervalAdd",
            "IntervalCat",
            "enable_interval_eval",
            "interval_forward",
            "interval_forward_refine",
            "affine_relu_transform",
            "affine_tanh_transform",
            "affine_sigmoid_transform",
        ]
    )
