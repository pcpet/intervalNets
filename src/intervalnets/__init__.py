"""Interval arithmetic utilities for neural network evaluation."""

from .interval import Interval
from .polynomial_zonotope import (
    PZOneJet,
    PZTwoJet,
    PolynomialZonotope,
    collect_pz_diagnostics,
    pz_to_latex,
    pz_to_markdown_code,
    twojet_to_latex,
)
from .pz_tanh import (
    AffineTanhEnclosure,
    QuadraticTanhEnclosure,
    TanhApproximation,
    affine_tanh_double_prime_enclosure,
    affine_tanh_enclosure,
    affine_tanh_prime_enclosure,
    quadratic_tanh_prime_enclosure,
    certify_tanh_residual_subdivision,
    compute_tanh_polynomial,
    tanh_pz_scalar,
)
from .pz_integration import (
    IntegratedPZResult,
    PZIntegrationCell,
    integrate_over_cell,
    integrate_pz_over_domain,
    integrate_pz_onejet_squared,
    integrate_pz_value_squared,
    pz_l2norm_bounds,
    pz_sobolev_norm_bounds,
)

from .pz_norms import (
    build_pz_twojet_norm_diagnostics,
    pz_norm_from_integrand,
    pz_sum_squares,
    pz_twojet_l2_integrand,
    pz_twojet_l2_norm,
    pz_twojet_w12_integrand,
    pz_twojet_w12_norm,
    pz_twojet_w22_integrand,
    pz_twojet_w22_norm,
)

from .pinn import load_tanh_mlp_checkpoint, sequential_value_jacobian_laplacian

__all__ = [
    "Interval",
    "PolynomialZonotope",
    "PZOneJet",
    "PZTwoJet",
    "collect_pz_diagnostics",
    "pz_to_latex",
    "pz_to_markdown_code",
    "twojet_to_latex",
    "AffineTanhEnclosure",
    "QuadraticTanhEnclosure",
    "TanhApproximation",
    "affine_tanh_double_prime_enclosure",
    "affine_tanh_enclosure",
    "affine_tanh_prime_enclosure",
    "quadratic_tanh_prime_enclosure",
    "compute_tanh_polynomial",
    "certify_tanh_residual_subdivision",
    "tanh_pz_scalar",
    "IntegratedPZResult",
    "PZIntegrationCell",
    "integrate_over_cell",
    "integrate_pz_over_domain",
    "integrate_pz_onejet_squared",
    "integrate_pz_value_squared",
    "pz_l2norm_bounds",
    "pz_sobolev_norm_bounds",
    "build_pz_twojet_norm_diagnostics",
    "pz_norm_from_integrand",
    "pz_sum_squares",
    "pz_twojet_l2_integrand",
    "pz_twojet_l2_norm",
    "pz_twojet_w12_integrand",
    "pz_twojet_w12_norm",
    "pz_twojet_w22_integrand",
    "pz_twojet_w22_norm",
    "sequential_value_jacobian_laplacian",
    "load_tanh_mlp_checkpoint",
]

try:
    from .pytorch import (
        IntervalAdd,
        IntervalCat,
        IntervalTensor,
        enable_interval_eval,
        interval_forward,
        interval_forward_refine,
        pz_l2norm,
        pz_onejet_forward,
        pz_sobolev_norm,
        pz_value_forward,
        pz_twojet_forward,
        PZValueTraceRecord,
        PZValueTraceResult,
        PZOneJetTraceRecord,
        PZOneJetTraceResult,
        PZReductionConfig,
        PZTwoJetTraceRecord,
        PZTwoJetTraceResult,
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
            "pz_l2norm",
            "pz_onejet_forward",
            "pz_sobolev_norm",
            "pz_value_forward",
            "pz_twojet_forward",
            "PZValueTraceRecord",
            "PZValueTraceResult",
            "PZOneJetTraceRecord",
            "PZOneJetTraceResult",
            "PZReductionConfig",
            "PZTwoJetTraceRecord",
            "PZTwoJetTraceResult",
        ]
    )
