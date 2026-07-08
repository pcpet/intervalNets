"""Interval arithmetic utilities for neural network evaluation."""

from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope
from .pz_tanh import TanhApproximation, compute_tanh_polynomial, certify_tanh_residual_subdivision, tanh_pz_scalar

__all__ = [
    "Interval",
    "PolynomialZonotope",
    "PZTwoJet",
    "TanhApproximation",
    "compute_tanh_polynomial",
    "certify_tanh_residual_subdivision",
    "tanh_pz_scalar",
]

try:
    from .pytorch import (
        IntervalAdd,
        IntervalCat,
        IntervalTensor,
        enable_interval_eval,
        interval_forward,
        interval_forward_refine,
        pz_twojet_forward,
    )
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
            "pz_twojet_forward",
        ]
    )
