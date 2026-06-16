"""Interval arithmetic utilities for neural network evaluation."""

from .affine import AffineTensor
from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope

__all__ = ["Interval", "AffineTensor", "PolynomialZonotope", "PZTwoJet"]

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
    from .pytorch import affine_forward
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
            "affine_forward",
            "affine_relu_transform",
            "affine_tanh_transform",
            "affine_sigmoid_transform",
        ]
    )
