"""Integration helpers for polynomial zonotopes.

The helpers in this module keep the exact polynomial contribution separate from
interval uncertainty created by pointwise approximation residuals.  Pointwise
residual symbols model a fresh adversarial value at each integration point, so
integrating them by monomial moments would be unsound unless the caller opts
into a purely symbolic treatment explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .polynomial_zonotope import (
    Exponent,
    PolynomialZonotope,
    _abs_coeff,
    _add_coeff,
    _mul_coeff,
    _zero_like,
    box_monomial_moment,
)

POINTWISE_RESIDUAL_KINDS = frozenset({"pointwise_residual", "approximation_pointwise"})
SYMBOLIC_APPROXIMATION_KINDS = frozenset({"approximation_symbolic", "global_symbolic_residual"})

IntegrationMode = Literal["pointwise_interval", "symbolic"]


@dataclass(frozen=True)
class IntegratedPZResult:
    """Result of integrating a polynomial zonotope over selected variables.

    Attributes:
        polynomial: Exact integral of all terms over retained symbolic
            variables.  In the default mode, pointwise residual terms are not
            included here.
        interval_radius: Accumulated non-negative scalar/tensor radius for
            integrated pointwise residual contributions.
        measure: Measure factor of the integrated domain, e.g. ``2**d`` for
            the normalized parameter box ``[-1, 1]^d`` or a geometric volume
            supplied by the caller.
        metadata: Details about the integration mode and noise classification.
    """

    polynomial: PolynomialZonotope
    interval_radius: Any
    measure: float
    metadata: dict[str, Any]

    def interval_enclosure(self):
        """Return an interval enclosure of ``polynomial +/- interval_radius``."""

        base = self.polynomial.interval_enclosure()
        lower_radius = _mul_coeff(self.interval_radius, -1.0)
        try:
            from .pytorch import IntervalTensor

            if base.__class__ is IntervalTensor:
                return base + IntervalTensor.from_bounds(lower_radius, self.interval_radius)
        except ImportError:  # pragma: no cover
            pass

        from .interval import Interval

        return base + Interval.from_bounds(lower_radius, self.interval_radius)


def _default_domain_indices(zonotope: PolynomialZonotope) -> tuple[int, ...]:
    return tuple(index for index, kind in enumerate(zonotope.noise_kinds) if kind == "domain")


def _is_pointwise_kind(kind: str) -> bool:
    return kind in POINTWISE_RESIDUAL_KINDS


def integrate_pz_over_domain(
    zonotope: PolynomialZonotope,
    domain_indices: Sequence[int] | None = None,
    *,
    mode: IntegrationMode = "pointwise_interval",
    volume: float | None = None,
) -> IntegratedPZResult:
    """Integrate domain variables while preserving pointwise residual semantics.

    In default ``"pointwise_interval"`` mode, monomials involving pointwise
    residual symbols are converted to interval-radius contributions with the
    integrated-domain measure.  They are *not* integrated by residual-symbol
    moments.  ``"symbolic"`` mode is an explicit opt-in that treats all
    non-domain variables, including approximation variables, as retained global
    symbolic variables and therefore integrates only the selected domain powers
    by exact box moments.
    """

    if mode not in ("pointwise_interval", "symbolic"):
        raise ValueError("mode must be 'pointwise_interval' or 'symbolic'.")
    indices = _default_domain_indices(zonotope) if domain_indices is None else tuple(int(i) for i in domain_indices)
    if len(set(indices)) != len(indices):
        raise ValueError("domain_indices must not contain duplicates.")
    if any(index < 0 or index >= zonotope.num_noise for index in indices):
        raise ValueError("domain index out of range.")

    domain_set = set(indices)
    retained_indices = tuple(index for index in range(zonotope.num_noise) if index not in domain_set)
    retained_kinds = tuple(zonotope.noise_kinds[index] for index in retained_indices)
    measure = float(volume) if volume is not None else float(2 ** len(indices))
    if measure < 0.0:
        raise ValueError("volume/measure must be non-negative.")

    center = _mul_coeff(zonotope.center, measure)
    terms: dict[Exponent, Any] = {}
    radius = _zero_like(zonotope.center)
    pointwise_indices = tuple(index for index, kind in enumerate(zonotope.noise_kinds) if _is_pointwise_kind(kind))
    zero_retained = (0,) * len(retained_indices)

    for exponent, coeff in zonotope.terms.items():
        has_pointwise = any(exponent[index] for index in pointwise_indices)
        if mode == "pointwise_interval" and has_pointwise:
            # A pointwise residual may choose an unrelated value at each domain
            # point.  Bound its integral by measure times the coefficient's
            # magnitude rather than by a symbolic residual moment.
            radius = _add_coeff(radius, _mul_coeff(_abs_coeff(coeff), measure))
            continue

        moment = box_monomial_moment(tuple(exponent[index] for index in indices))
        if moment == 0.0:
            continue
        retained_exponent = tuple(exponent[index] for index in retained_indices)
        integrated_coeff = _mul_coeff(coeff, moment)
        if retained_exponent == zero_retained:
            center = _add_coeff(center, integrated_coeff)
        else:
            terms[retained_exponent] = _add_coeff(terms[retained_exponent], integrated_coeff) if retained_exponent in terms else integrated_coeff

    return IntegratedPZResult(
        polynomial=PolynomialZonotope(center, terms, num_noise=len(retained_indices), noise_kinds=retained_kinds),
        interval_radius=radius,
        measure=measure,
        metadata={
            "mode": mode,
            "domain_indices": indices,
            "pointwise_residual_indices": pointwise_indices,
            "pointwise_residual_kinds": tuple(zonotope.noise_kinds[index] for index in pointwise_indices),
        },
    )
