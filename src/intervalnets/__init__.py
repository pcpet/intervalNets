"""Interval arithmetic utilities for neural network evaluation."""

from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope
from .pz_tanh import TanhApproximation, compute_tanh_polynomial, certify_tanh_residual_subdivision, tanh_pz_scalar
from .pz_integration import IntegratedPZResult, PZIntegrationCell, integrate_over_cell, integrate_pz_over_domain

from .pz_norms import (
    pz_norm_from_integrand,
    pz_sum_squares,
    pz_twojet_l2_integrand,
    pz_twojet_l2_norm,
    pz_twojet_w12_integrand,
    pz_twojet_w12_norm,
    pz_twojet_w22_integrand,
    pz_twojet_w22_norm,
)

__all__ = [
    "Interval",
    "PolynomialZonotope",
    "PZTwoJet",
    "TanhApproximation",
    "compute_tanh_polynomial",
    "certify_tanh_residual_subdivision",
    "tanh_pz_scalar",
    "IntegratedPZResult",
    "PZIntegrationCell",
    "integrate_over_cell",
    "integrate_pz_over_domain",
    "pz_norm_from_integrand",
    "pz_sum_squares",
    "pz_twojet_l2_integrand",
    "pz_twojet_l2_norm",
    "pz_twojet_w12_integrand",
    "pz_twojet_w12_norm",
    "pz_twojet_w22_integrand",
    "pz_twojet_w22_norm",
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
