"""Polynomial-zonotope norm integrands and certified norm intervals."""

from __future__ import annotations

from collections import Counter
from math import inf, isfinite, nextafter, sqrt
from typing import Any, Sequence

from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope, pz_to_latex, pz_to_markdown_code, twojet_to_latex
from .pz_integration import PZIntegrationCell, integrate_over_cell, integrate_pz_over_domain, integrate_pz_twojet_squared

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _zero_scalar_like(z: PolynomialZonotope) -> PolynomialZonotope:
    if torch is not None and isinstance(z.center, torch.Tensor):
        return PolynomialZonotope.constant(torch.zeros((), dtype=z.center.dtype, device=z.center.device), num_noise=z.num_noise, noise_kinds=z.noise_kinds)
    return PolynomialZonotope.constant(0.0, num_noise=z.num_noise, noise_kinds=z.noise_kinds)


def _fallback_scalar_indices(value: Any, prefix: tuple[int, ...] = ()):  # type: ignore[no-untyped-def]
    if isinstance(value, tuple):
        for idx, item in enumerate(value):
            yield from _fallback_scalar_indices(item, prefix + (idx,))
    else:
        yield prefix


def _nested_get(value: Any, index: tuple[int, ...]):
    out = value
    for item in index:
        out = out[item]
    return out


def _scalar_entries(z: PolynomialZonotope):
    if z.shape == ():
        yield z
        return
    if torch is not None and isinstance(z.center, torch.Tensor):
        for flat_idx in range(z.center.numel()):
            multi = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx, device=z.center.device), z.center.shape))
            yield z[multi]
        return
    for index in _fallback_scalar_indices(z.center):
        yield PolynomialZonotope(_nested_get(z.center, index), {exp: _nested_get(coeff, index) for exp, coeff in z.terms.items()}, num_noise=z.num_noise, noise_kinds=z.noise_kinds)



def _pz_metadata_summary(z: PolynomialZonotope) -> dict[str, Any]:
    """Return lightweight structural metadata for a polynomial zonotope."""

    return {
        "shape": z.shape,
        "term_count": len(z.terms),
        "max_degree": max((sum(exp) for exp in z.terms), default=0),
        "num_noise": z.num_noise,
        "noise_kind_counts": dict(Counter(z.noise_kinds)),
    }


def build_pz_twojet_norm_diagnostics(
    jet: PZTwoJet,
    *,
    include_integrands: bool = True,
    render: bool = True,
    max_terms: int | None = 12,
    precision: int = 4,
) -> dict[str, Any]:
    """Build pre-norm diagnostics for a polynomial-zonotope two-jet.

    The returned dictionary exposes the raw ``Y``, ``J``, and ``H`` zonotopes
    by reference so callers can inspect the exact final two-jet before norm
    integrand construction. Optional integrands are produced with the same
    public helpers used by the norm routines, and renderer output is included
    when ``render`` is true. The input jet is never modified.
    """

    diagnostics: dict[str, Any] = {
        "jet": {"Y": jet.Y, "J": jet.J, "H": jet.H},
        "metadata": {
            "Y": _pz_metadata_summary(jet.Y),
            "J": _pz_metadata_summary(jet.J),
            "H": _pz_metadata_summary(jet.H),
        },
    }
    diagnostics["metadata"]["total_term_count"] = sum(
        diagnostics["metadata"][label]["term_count"] for label in ("Y", "J", "H")
    )
    diagnostics["metadata"]["max_degree"] = max(
        diagnostics["metadata"][label]["max_degree"] for label in ("Y", "J", "H")
    )
    diagnostics["metadata"]["shapes"] = {
        label: diagnostics["metadata"][label]["shape"] for label in ("Y", "J", "H")
    }
    diagnostics["metadata"]["noise_kind_counts"] = dict(
        Counter(kind for label in ("Y", "J", "H") for kind in getattr(jet, label).noise_kinds)
    )

    if render:
        diagnostics["rendered"] = {
            "latex": {
                "twojet": twojet_to_latex(jet, max_terms=max_terms, precision=precision),
                "Y": pz_to_latex(jet.Y, max_terms=max_terms, precision=precision),
                "J": pz_to_latex(jet.J, max_terms=max_terms, precision=precision),
                "H": pz_to_latex(jet.H, max_terms=max_terms, precision=precision),
            },
            "markdown": {
                "Y": pz_to_markdown_code(jet.Y, max_terms=max_terms, precision=precision),
                "J": pz_to_markdown_code(jet.J, max_terms=max_terms, precision=precision),
                "H": pz_to_markdown_code(jet.H, max_terms=max_terms, precision=precision),
            },
        }

    if include_integrands:
        diagnostics["integrands"] = {
            "l2_integrand": pz_twojet_l2_integrand(jet),
            "w12_integrand": pz_twojet_w12_integrand(jet),
            "w22_integrand": pz_twojet_w22_integrand(jet),
        }

    return diagnostics

def pz_sum_squares(z: PolynomialZonotope) -> PolynomialZonotope:
    """Return the algebraic sum of squares of every scalar entry in ``z``."""

    total = _zero_scalar_like(z)
    for entry in _scalar_entries(z):
        total = total + entry * entry
    return total


def pz_symmetric_hessian_sum_squares(hessian: PolynomialZonotope) -> PolynomialZonotope:
    """Return the dense Hessian square sum using Hessian symmetry.

    The dense two-jet Hessian convention stores one full symmetric matrix per
    output with shape ``(output_dim, input_dim, input_dim)``.  For that shape,
    off-diagonal entries occur twice in the full Frobenius sum, so accumulate
    only the upper-triangular entries and double the off-diagonal squares.
    Scalar-output Hessians may also be stored as ``(input_dim, input_dim)``;
    that lower-rank convention is handled analogously.
    """

    if len(hessian.shape) == 3 and hessian.shape[1] == hessian.shape[2]:
        output_dim, input_dim, _ = hessian.shape
        output_indices: range | tuple[None, ...] = range(output_dim)
    elif len(hessian.shape) == 2 and hessian.shape[0] == hessian.shape[1]:
        input_dim = hessian.shape[0]
        output_indices = (None,)
    else:
        return pz_sum_squares(hessian)

    total = _zero_scalar_like(hessian)
    for out in output_indices:
        for a in range(input_dim):
            diagonal = hessian[a, a] if out is None else hessian[out, a, a]
            total = total + diagonal * diagonal
            for b in range(a + 1, input_dim):
                off_diagonal = hessian[a, b] if out is None else hessian[out, a, b]
                total = total + 2.0 * off_diagonal * off_diagonal
    return total


def pz_twojet_l2_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared L2 integrand ``sum_i Y_i^2`` for a two-jet."""

    return pz_sum_squares(jet.Y)


def pz_twojet_w12_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared W^{1,2} integrand ``sum_i Y_i^2 + sum_ij J_ij^2``."""

    return pz_twojet_l2_integrand(jet) + pz_sum_squares(jet.J)


def pz_twojet_w22_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared W^{2,2} integrand including value, Jacobian, and Hessian."""

    return pz_twojet_w12_integrand(jet) + pz_symmetric_hessian_sum_squares(jet.H)


def _require_p2(p: float) -> None:
    if not isfinite(float(p)) or float(p) != 2.0:
        raise NotImplementedError("Polynomial-zonotope norm helpers currently support only p=2.0.")


def _sqrt_interval_nonnegative(value: Interval) -> Interval:
    lower = max(0.0, float(value.lower))
    upper = max(0.0, float(value.upper))
    return Interval.from_bounds(nextafter(sqrt(lower), -inf), nextafter(sqrt(upper), inf))


def pz_norm_from_integrand(
    integrand: PolynomialZonotope,
    cell: PZIntegrationCell | None = None,
    *,
    domain_indices: Sequence[int] | None = None,
    p: float = 2.0,
) -> Interval:
    """Integrate a squared PZ integrand exactly over domain variables and sqrt."""

    _require_p2(p)
    if cell is not None:
        integral = integrate_over_cell(integrand, cell, output="interval")
    else:
        integral = integrate_pz_over_domain(integrand, domain_indices, mode="pointwise_interval").interval_enclosure()
    return _sqrt_interval_nonnegative(integral)


def pz_twojet_l2_norm(jet: PZTwoJet, cell: PZIntegrationCell | None = None, *, p: float = 2.0) -> Interval:
    _require_p2(p)
    if cell is not None:
        return _sqrt_interval_nonnegative(integrate_pz_twojet_squared(jet, cell, "l2"))
    return pz_norm_from_integrand(pz_twojet_l2_integrand(jet), p=p)


def pz_twojet_w12_norm(jet: PZTwoJet, cell: PZIntegrationCell | None = None, *, p: float = 2.0) -> Interval:
    _require_p2(p)
    if cell is not None:
        return _sqrt_interval_nonnegative(integrate_pz_twojet_squared(jet, cell, "w12"))
    return pz_norm_from_integrand(pz_twojet_w12_integrand(jet), p=p)


def pz_twojet_w22_norm(jet: PZTwoJet, cell: PZIntegrationCell | None = None, *, p: float = 2.0) -> Interval:
    _require_p2(p)
    if cell is not None:
        return _sqrt_interval_nonnegative(integrate_pz_twojet_squared(jet, cell, "w22"))
    return pz_norm_from_integrand(pz_twojet_w22_integrand(jet), p=p)
