"""Polynomial-zonotope norm integrands and certified norm intervals."""

from __future__ import annotations

from math import inf, isfinite, nextafter, sqrt
from typing import Any, Sequence

from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope
from .pz_integration import PZIntegrationCell, integrate_over_cell, integrate_pz_over_domain

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


def pz_sum_squares(z: PolynomialZonotope) -> PolynomialZonotope:
    """Return the algebraic sum of squares of every scalar entry in ``z``."""

    total = _zero_scalar_like(z)
    for entry in _scalar_entries(z):
        total = total + entry * entry
    return total


def pz_twojet_l2_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared L2 integrand ``sum_i Y_i^2`` for a two-jet."""

    return pz_sum_squares(jet.Y)


def pz_twojet_w12_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared W^{1,2} integrand ``sum_i Y_i^2 + sum_ij J_ij^2``."""

    return pz_twojet_l2_integrand(jet) + pz_sum_squares(jet.J)


def pz_twojet_w22_integrand(jet: PZTwoJet) -> PolynomialZonotope:
    """Squared W^{2,2} integrand including value, Jacobian, and Hessian."""

    return pz_twojet_w12_integrand(jet) + pz_sum_squares(jet.H)


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
    return pz_norm_from_integrand(pz_twojet_l2_integrand(jet), cell, p=p)


def pz_twojet_w12_norm(jet: PZTwoJet, cell: PZIntegrationCell | None = None, *, p: float = 2.0) -> Interval:
    return pz_norm_from_integrand(pz_twojet_w12_integrand(jet), cell, p=p)


def pz_twojet_w22_norm(jet: PZTwoJet, cell: PZIntegrationCell | None = None, *, p: float = 2.0) -> Interval:
    return pz_norm_from_integrand(pz_twojet_w22_integrand(jet), cell, p=p)
