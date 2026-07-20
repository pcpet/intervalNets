"""Integration helpers for polynomial zonotopes.

The helpers in this module keep the exact polynomial contribution separate from
interval uncertainty created by pointwise approximation residuals. Pointwise
residual symbols model a fresh adversarial value at each integration point, so
integrating them by monomial moments would be unsound unless the caller opts
into a purely symbolic treatment explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Literal, Sequence

from .interval import Interval
from .polynomial_zonotope import (
    Exponent,
    PolynomialZonotope,
    _abs_coeff,
    _add_coeff,
    _mul_coeff,
    _to_fallback,
    _zero_like,
    box_monomial_moment,
)

try:  # pragma: no cover - optional dependency
    from .pytorch import IntervalTensor
except ImportError:  # pragma: no cover
    IntervalTensor = None  # type: ignore[assignment]

POINTWISE_RESIDUAL_KINDS = frozenset({"pointwise_residual", "approximation_pointwise"})
SYMBOLIC_APPROXIMATION_KINDS = frozenset({"approximation_symbolic", "global_symbolic_residual"})

IntegrationMode = Literal["pointwise_interval", "symbolic"]
IntegrationOutput = Literal["interval", "pz"]


@dataclass(frozen=True)
class PZIntegrationCell:
    """A parameterized cell used for geometric polynomial-zonotope integration.

    ``domain`` maps reference variables in ``[-1, 1]^n`` to physical
    coordinates. ``domain_noise_indices`` identifies exactly those reference
    variables. ``jacobian_density`` is the non-negative density multiplying the
    reference integral. For affine axis-aligned boxes this is simply the product
    of coordinate radii.
    """

    domain: PolynomialZonotope
    domain_noise_indices: tuple[int, ...]
    jacobian_density: PolynomialZonotope | float
    volume: float | Interval
    orientation: int | None
    source_box: "IntervalTensor | None"

    @classmethod
    def from_affine_box(cls, box: Interval | "IntervalTensor") -> "PZIntegrationCell":
        """Create the affine cell mapping ``[-1, 1]^n`` to an interval box."""

        domain = PolynomialZonotope.from_box(box.lower, box.upper)
        radii = tuple(_flatten_scalars(box.radius))
        density = float(prod(radii))
        dim = len(radii)
        return cls(
            domain=domain.with_noise_kinds(("domain",) * domain.num_noise),
            domain_noise_indices=tuple(range(dim)),
            jacobian_density=density,
            volume=float((2.0**dim) * density),
            orientation=1 if density >= 0.0 else -1,
            source_box=box if IntervalTensor is not None and isinstance(box, IntervalTensor) else None,
        )

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "PZIntegrationCell":
        """Create an affine integration cell from lower and upper box bounds."""

        if IntervalTensor is not None:
            return cls.from_affine_box(IntervalTensor.from_bounds(lower, upper))
        return cls.from_affine_box(Interval.from_bounds(lower, upper))

    @classmethod
    def from_fixed_orientation_domain(
        cls,
        domain: PolynomialZonotope,
        domain_noise_indices: Sequence[int],
        *,
        determinant: PolynomialZonotope | None = None,
        orientation: int | None = None,
        injectivity_certificate: Any | None = None,
        source_box: "IntervalTensor | None" = None,
    ) -> "PZIntegrationCell":
        """Restricted hook for future non-affine certified PZ domains.

        Non-affine changes of variables require a certified determinant
        polynomial plus injectivity and fixed-orientation certificates. Until the
        certificate objects are defined by the implementation, this constructor
        intentionally refuses uncertified domains.
        """

        indices = tuple(int(index) for index in domain_noise_indices)
        if determinant is None or orientation not in (-1, 1) or injectivity_certificate is None:
            raise NotImplementedError(
                "Non-affine PZ integration requires a certified determinant polynomial "
                "and injectivity/fixed-orientation certificates."
            )
        if len(set(indices)) != len(indices):
            raise ValueError("domain_noise_indices must not contain duplicates.")
        if any(index < 0 or index >= domain.num_noise for index in indices):
            raise ValueError("domain noise index out of range.")

        density = determinant * float(orientation)
        volume = integrate_pz_over_domain(density, indices, mode="pointwise_interval").interval_enclosure()
        return cls(
            domain=domain,
            domain_noise_indices=indices,
            jacobian_density=density,
            volume=volume,
            orientation=orientation,
            source_box=source_box,
        )


@dataclass(frozen=True)
class IntegratedPZResult:
    """Result of integrating a polynomial zonotope over selected variables."""

    polynomial: PolynomialZonotope
    interval_radius: Any
    measure: float
    metadata: dict[str, Any]

    def interval_enclosure(self):
        """Return an interval enclosure of ``polynomial +/- interval_radius``."""

        base = self.polynomial.interval_enclosure()
        lower_radius = _mul_coeff(self.interval_radius, -1.0)
        if IntervalTensor is not None and base.__class__ is IntervalTensor:
            return base + IntervalTensor.from_bounds(lower_radius, self.interval_radius)
        return base + Interval.from_bounds(lower_radius, self.interval_radius)


def _flatten_scalars(value: Any):
    data = _to_fallback(value)
    if isinstance(data, tuple):
        for item in data:
            yield from _flatten_scalars(item)
    else:
        yield float(data)


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
    """Integrate domain variables while preserving pointwise residual semantics."""

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


def integrate_over_cell(pz_expr: PolynomialZonotope, cell: PZIntegrationCell, *, output: IntegrationOutput = "interval"):
    """Integrate a PZ expression geometrically over an integration cell.

    ``output="pz"`` integrates only the cell domain variables and retains all
    non-domain approximation variables symbolically. ``output="interval"``
    first performs pointwise-residual-safe domain integration, then encloses the
    retained symbolic approximation-noise polynomial and adds accumulated
    pointwise residual radii.
    """

    if output not in ("interval", "pz"):
        raise ValueError("output must be 'interval' or 'pz'.")
    weighted = pz_expr * cell.jacobian_density
    if output == "pz":
        return integrate_pz_over_domain(weighted, cell.domain_noise_indices, mode="symbolic").polynomial
    return integrate_pz_over_domain(weighted, cell.domain_noise_indices, mode="pointwise_interval").interval_enclosure()
