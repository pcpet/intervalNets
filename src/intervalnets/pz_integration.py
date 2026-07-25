"""Integration helpers for polynomial zonotopes.

The helpers in this module keep the exact polynomial contribution separate from
interval uncertainty created by pointwise approximation residuals. Pointwise
residual symbols model a fresh adversarial value at each integration point, so
integrating them by monomial moments would be unsound unless the caller opts
into a purely symbolic treatment explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf, isfinite, nextafter, prod, sqrt
from numbers import Real
from typing import Any, Literal, Sequence

from .interval import Interval
from .polynomial_zonotope import (
    Exponent,
    PZTwoJet,
    PolynomialZonotope,
    _abs_coeff,
    _add_coeff,
    _mul_coeff,
    _merge_noise_kinds,
    _to_fallback,
    _zero_like,
    box_monomial_moment,
    torch,
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


TwoJetIntegrandKind = Literal["l2", "w12", "w22"]


def _scalar_coordinates(zonotope: PolynomialZonotope) -> list[PolynomialZonotope]:
    """Flatten a tensor-valued PZ without converting its scalar coefficients."""

    if zonotope.shape == ():
        return [zonotope]
    return [zonotope[index] for index in product(*(range(size) for size in zonotope.shape))]


def _twojet_weighted_coordinates(
    jet: PZTwoJet, integrand_kind: TwoJetIntegrandKind
) -> tuple[list[PolynomialZonotope], list[float]]:
    if integrand_kind not in {"l2", "w12", "w22"}:
        raise ValueError("integrand_kind must be 'l2', 'w12', or 'w22'.")

    coordinates = _scalar_coordinates(jet.Y)
    weights = [1.0] * len(coordinates)
    if integrand_kind in {"w12", "w22"}:
        jacobian = _scalar_coordinates(jet.J)
        coordinates.extend(jacobian)
        weights.extend([1.0] * len(jacobian))
    if integrand_kind == "w22":
        shape = jet.H.shape
        if len(shape) == 3 and shape[1] == shape[2]:
            output_indices: tuple[int | None, ...] = tuple(range(shape[0]))
            input_dim = shape[1]
        elif len(shape) == 2 and shape[0] == shape[1]:
            output_indices = (None,)
            input_dim = shape[0]
        else:
            raise ValueError("w22 direct integration requires a square stored Hessian.")
        for output in output_indices:
            for row in range(input_dim):
                for column in range(row, input_dim):
                    index = (row, column) if output is None else (output, row, column)
                    coordinates.append(jet.H[index])
                    weights.append(1.0 if row == column else 2.0)
    return coordinates, weights


def _validate_twojet_metadata(jet: PZTwoJet) -> tuple[int, tuple[str, ...]]:
    components = (jet.Y, jet.J, jet.H)
    num_noise = components[0].num_noise
    if any(component.num_noise != num_noise for component in components[1:]):
        raise ValueError("Y, J, and H must have identical num_noise metadata.")
    noise_kinds = components[0].noise_kinds
    for component in components[1:]:
        noise_kinds = _merge_noise_kinds(noise_kinds, component.noise_kinds)
    return num_noise, noise_kinds


def _coefficient_matrix(
    coordinates: Sequence[PolynomialZonotope], union_support: Sequence[Exponent]
) -> tuple[Any, Any]:
    centers = [coordinate.center for coordinate in coordinates]
    matrix = [
        [coordinate.terms.get(exponent, _zero_like(coordinate.center)) for coordinate in coordinates]
        for exponent in union_support
    ]
    if torch is not None and isinstance(centers[0], torch.Tensor):
        center_vector = torch.stack(centers)
        if matrix:
            return center_vector, torch.stack([torch.stack(row) for row in matrix])
        return center_vector, torch.empty((0, len(centers)), dtype=center_vector.dtype, device=center_vector.device)
    return centers, matrix


def _weighted_dot(left: Sequence[Any], right: Sequence[Any], weights: Sequence[float]):
    result = _zero_like(left[0])
    for lhs, rhs, weight in zip(left, right, weights):
        result = _add_coeff(result, _mul_coeff(_mul_coeff(lhs, rhs), weight))
    return result


def integrate_pz_twojet_squared(
    jet: PZTwoJet, cell: PZIntegrationCell, integrand_kind: TwoJetIntegrandKind
):
    """Directly integrate a squared two-jet over a supported affine cell.

    Unsupported densities and Hessian layouts deliberately use the explicit
    squared-integrand reference pipeline.
    """

    def explicit_fallback():
        from .pz_norms import (
            pz_twojet_l2_integrand,
            pz_twojet_w12_integrand,
            pz_twojet_w22_integrand,
        )

        constructor = {
            "l2": pz_twojet_l2_integrand,
            "w12": pz_twojet_w12_integrand,
            "w22": pz_twojet_w22_integrand,
        }.get(integrand_kind)
        if constructor is None:
            raise ValueError("integrand_kind must be 'l2', 'w12', or 'w22'.")
        return integrate_over_cell(constructor(jet), cell, output="interval")

    density = cell.jacobian_density
    if not isinstance(density, Real) or not isfinite(float(density)) or float(density) < 0.0:
        return explicit_fallback()

    num_noise, noise_kinds = _validate_twojet_metadata(jet)
    try:
        coordinates, weights = _twojet_weighted_coordinates(jet, integrand_kind)
    except ValueError as error:
        if "Hessian" in str(error):
            return explicit_fallback()
        raise
    if not coordinates:
        return explicit_fallback()

    domain_indices = tuple(int(index) for index in cell.domain_noise_indices)
    if len(set(domain_indices)) != len(domain_indices) or any(index < 0 or index >= num_noise for index in domain_indices):
        raise ValueError("cell domain noise index out of range or duplicated.")
    domain_set = set(domain_indices)
    retained_indices = tuple(index for index in range(num_noise) if index not in domain_set)
    retained_kinds = tuple(noise_kinds[index] for index in retained_indices)
    pointwise_indices = tuple(index for index, kind in enumerate(noise_kinds) if kind in POINTWISE_RESIDUAL_KINDS)
    measure = float(2 ** len(domain_indices))
    scale = float(density)

    support = sorted(set().union(*(coordinate.terms for coordinate in coordinates)))
    centers, matrix = _coefficient_matrix(coordinates, support)
    if torch is not None and isinstance(centers, torch.Tensor):
        weight_vector = torch.tensor(weights, dtype=centers.dtype, device=centers.device)
        weighted_matrix = matrix * weight_vector.unsqueeze(0)
        center_cross = weighted_matrix @ centers
        gram = weighted_matrix @ matrix.T
        center_square = torch.dot(centers * weight_vector, centers)
    else:
        center_cross = [_weighted_dot(row, centers, weights) for row in matrix]
        gram = [[_weighted_dot(left, right, weights) for right in matrix] for left in matrix]
        center_square = _weighted_dot(centers, centers, weights)

    retained: dict[Exponent, Any] = {}
    pointwise: dict[Exponent, Any] = {}
    zero_retained = (0,) * len(retained_indices)

    def accumulate(target: dict[Exponent, Any], exponent: Exponent, coefficient: Any) -> None:
        target[exponent] = _add_coeff(target[exponent], coefficient) if exponent in target else coefficient

    def route(exponent: Exponent, coefficient: Any) -> None:
        scaled = _mul_coeff(coefficient, scale)
        if any(exponent[index] for index in pointwise_indices):
            accumulate(pointwise, exponent, scaled)
            return
        moment = box_monomial_moment(tuple(exponent[index] for index in domain_indices))
        if moment == 0.0:
            return
        retained_exponent = tuple(exponent[index] for index in retained_indices)
        accumulate(retained, retained_exponent, _mul_coeff(scaled, moment))

    route((0,) * num_noise, center_square)
    for index, exponent in enumerate(support):
        route(exponent, _mul_coeff(center_cross[index], 2.0))
        for other_index in range(index, len(support)):
            pair_exponent = tuple(a + b for a, b in zip(exponent, support[other_index]))
            factor = 1.0 if index == other_index else 2.0
            route(pair_exponent, _mul_coeff(gram[index][other_index], factor))

    center = retained.pop(zero_retained, _zero_like(centers[0]))
    radius = _zero_like(center)
    for coefficient in pointwise.values():
        radius = _add_coeff(radius, _mul_coeff(_abs_coeff(coefficient), measure))
    result = IntegratedPZResult(
        polynomial=PolynomialZonotope(center, retained, num_noise=len(retained_indices), noise_kinds=retained_kinds),
        interval_radius=radius,
        measure=measure,
        metadata={"mode": "pointwise_interval", "direct_twojet_squared": True, "integrand_kind": integrand_kind},
    )
    return result.interval_enclosure()


def _require_interval_tensor_domain(domain: Any):
    from .pytorch import IntervalTensor as RuntimeIntervalTensor

    if not isinstance(domain, RuntimeIntervalTensor):
        raise TypeError("PZ adaptive integration requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("PZ adaptive integration currently supports flat input boxes only.")
    return RuntimeIntervalTensor


def _require_l2_output(output: str) -> None:
    if output not in {"interval", "pz"}:
        raise ValueError("output must be either 'interval' or 'pz'.")


def _require_adaptive_parameters(iterations: int, theta: float, chebyshev_degree: int, residual_subdivisions: int) -> None:
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not isfinite(float(theta)) or float(theta) <= 0.0 or float(theta) > 1.0:
        raise ValueError("theta must be a finite real number in the interval (0, 1].")
    if chebyshev_degree < 0:
        raise ValueError("chebyshev_degree must be non-negative.")
    if residual_subdivisions < 1:
        raise ValueError("residual_subdivisions must be positive.")


def _sqrt_interval_nonnegative(value: Interval) -> Interval:
    lower = max(0.0, float(value.lower))
    upper = max(0.0, float(value.upper))
    return Interval.from_bounds(nextafter(sqrt(lower), -inf), nextafter(sqrt(upper), inf))


def _interval_width(value: Interval) -> float:
    return max(0.0, float(value.upper) - float(value.lower))


def _interval_add(left: Interval, right: Interval) -> Interval:
    return left + right


def _dorfler_marking(indicators: list[float], theta: float) -> list[int]:
    """Return a minimal Dörfler marked set, matching ``pytorch._dorfler_marking``."""

    if not indicators:
        raise ValueError("Indicators must be non-empty for Dörfler marking.")
    total = sum(indicators)
    if total <= 0.0:
        return [max(range(len(indicators)), key=lambda idx: indicators[idx])]
    threshold = theta * total
    ranked_indices = sorted(range(len(indicators)), key=lambda idx: indicators[idx], reverse=True)
    marked: list[int] = []
    accumulated = 0.0
    for idx in ranked_indices:
        marked.append(idx)
        accumulated += indicators[idx]
        if accumulated >= threshold:
            break
    return marked


def _split_box(box: "IntervalTensor", split_dim: int | None = None) -> tuple["IntervalTensor", "IntervalTensor"]:
    """Bisect an interval box, matching ``pytorch._split_box`` behavior."""

    if split_dim is None:
        widths = [float(upper - lower) for lower, upper in zip(box.lower, box.upper)]
        split_dim = max(range(len(widths)), key=lambda idx: widths[idx])
    midpoint = 0.5 * (box.lower[split_dim] + box.upper[split_dim])
    lower_left = list(box.lower)
    upper_left = list(box.upper)
    lower_right = list(box.lower)
    upper_right = list(box.upper)
    upper_left[split_dim] = midpoint
    lower_right[split_dim] = midpoint
    from .pytorch import IntervalTensor as RuntimeIntervalTensor

    return RuntimeIntervalTensor.from_bounds(lower_left, upper_left), RuntimeIntervalTensor.from_bounds(lower_right, upper_right)


def _choose_split_dim_from_jacobian(box: "IntervalTensor", jacobian: Interval | None) -> int:
    widths = [float(upper - lower) for lower, upper in zip(box.lower, box.upper)]
    if jacobian is None or len(widths) <= 1:
        return max(range(len(widths)), key=lambda idx: widths[idx])
    try:
        if len(jacobian.shape) != 2:
            raise ValueError
        output_dim = len(jacobian.lower)
        input_dim = len(jacobian.lower[0]) if output_dim > 0 else 0
        scores = [0.0] * input_dim
        for row_idx in range(output_dim):
            for col_idx in range(input_dim):
                lower = float(jacobian.lower[row_idx][col_idx])
                upper = float(jacobian.upper[row_idx][col_idx])
                scores[col_idx] += max(abs(lower), abs(upper))
        weighted = [width * score for width, score in zip(widths, scores)]
        if any(score > 0.0 for score in weighted):
            return max(range(len(weighted)), key=lambda idx: weighted[idx])
    except Exception:
        pass
    return max(range(len(widths)), key=lambda idx: widths[idx])


def _eval_pz_twojet(model, domain: PolynomialZonotope, *, chebyshev_degree: int = 5, residual_subdivisions: int = 128):
    if hasattr(model, "eval_pz_twojet"):
        return model.eval_pz_twojet(domain, chebyshev_degree=chebyshev_degree, residual_subdivisions=residual_subdivisions)
    from .pytorch import pz_twojet_forward

    return pz_twojet_forward(model, domain, chebyshev_degree=chebyshev_degree, residual_subdivisions=residual_subdivisions)


@dataclass(frozen=True)
class _CachedSquaredContribution:
    """Cached adaptive-quadrature data for one active PZ integration cell."""

    box: "IntervalTensor"
    contribution: Interval
    jacobian: Any
    split_dim: int


def _squared_twojet_integrand(jet: Any, integrand_kind: Literal["l2", "w12", "w22"]) -> PolynomialZonotope:
    from .pz_norms import pz_twojet_l2_integrand, pz_twojet_w12_integrand, pz_twojet_w22_integrand

    integrands = {
        "l2": pz_twojet_l2_integrand,
        "w12": pz_twojet_w12_integrand,
        "w22": pz_twojet_w22_integrand,
    }
    try:
        return integrands[integrand_kind](jet)
    except KeyError as exc:  # pragma: no cover - guarded by Literal/internal callers
        raise ValueError("integrand_kind must be one of 'l2', 'w12', or 'w22'.") from exc


def _evaluate_squared_contribution_cache(
    model,
    box: "IntervalTensor",
    *,
    integrand_kind: Literal["l2", "w12", "w22"],
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
) -> _CachedSquaredContribution:
    """Evaluate and cache all expensive data needed for one active cell.

    The affine PZ integration cell, two-jet enclosure, squared integrand,
    integrated interval contribution, Jacobian enclosure, and preferred split
    dimension are computed exactly once for the cell lifetime. Refinement
    discards only marked parent cells and computes fresh cache entries for
    their children.
    """

    cell = PZIntegrationCell.from_affine_box(box)
    jet = _eval_pz_twojet(
        model,
        cell.domain,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
    )
    integrand = _squared_twojet_integrand(jet, integrand_kind)
    contribution = integrate_over_cell(integrand, cell, output="interval")
    jacobian = jet.J.interval_enclosure()
    return _CachedSquaredContribution(
        box=box,
        contribution=contribution,
        jacobian=jacobian,
        split_dim=_choose_split_dim_from_jacobian(box, jacobian),
    )


def _integrated_squared_contribution(
    model,
    box: "IntervalTensor",
    *,
    integrand_kind: Literal["l2", "w12", "w22"],
    chebyshev_degree: int,
    residual_subdivisions: int,
) -> tuple[Interval, Interval | None]:
    cached = _evaluate_squared_contribution_cache(
        model,
        box,
        integrand_kind=integrand_kind,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
    )
    return cached.contribution, cached.jacobian


def _pz_adaptive_squared_integral(
    model,
    domain: "IntervalTensor",
    *,
    integrand_kind: Literal["l2", "w12", "w22"],
    iterations: int,
    theta: float,
    chebyshev_degree: int,
    residual_subdivisions: int,
) -> Interval:
    active_cells = [
        _evaluate_squared_contribution_cache(
            model,
            domain,
            integrand_kind=integrand_kind,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
    ]
    for _ in range(iterations):
        indicators = [_interval_width(cell.contribution) for cell in active_cells]
        marked_indices = set(_dorfler_marking(indicators, theta))
        refined_cells: list[_CachedSquaredContribution] = []
        for idx, cell in enumerate(active_cells):
            if idx not in marked_indices:
                refined_cells.append(cell)
                continue
            for child_box in _split_box(cell.box, split_dim=cell.split_dim):
                refined_cells.append(
                    _evaluate_squared_contribution_cache(
                        model,
                        child_box,
                        integrand_kind=integrand_kind,
                        chebyshev_degree=chebyshev_degree,
                        residual_subdivisions=residual_subdivisions,
                    )
                )
        active_cells = refined_cells

    integral = Interval.point(0.0)
    for cell in active_cells:
        integral = _interval_add(integral, cell.contribution)
    return integral


def pz_l2norm_bounds(
    model,
    domain: "IntervalTensor",
    iterations: int = 0,
    theta: float = 0.5,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    output: IntegrationOutput = "interval",
) -> Interval:
    """Adaptive PZ two-jet enclosure of the L2 norm over an interval domain."""

    _require_interval_tensor_domain(domain)
    _require_l2_output(output)
    _require_adaptive_parameters(iterations, theta, chebyshev_degree, residual_subdivisions)
    squared = _pz_adaptive_squared_integral(
        model,
        domain,
        integrand_kind="l2",
        iterations=iterations,
        theta=theta,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
    )
    return _sqrt_interval_nonnegative(squared)


def pz_sobolev_norm_bounds(
    model,
    domain: "IntervalTensor",
    order: Literal[1, 2] = 1,
    iterations: int = 0,
    theta: float = 0.5,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    output: IntegrationOutput = "interval",
) -> Interval:
    """Adaptive PZ two-jet enclosure of W^{order,2} Sobolev norms."""

    _require_interval_tensor_domain(domain)
    _require_l2_output(output)
    _require_adaptive_parameters(iterations, theta, chebyshev_degree, residual_subdivisions)
    if order not in {1, 2}:
        raise ValueError("order must be either 1 or 2.")
    squared = _pz_adaptive_squared_integral(
        model,
        domain,
        integrand_kind="w12" if order == 1 else "w22",
        iterations=iterations,
        theta=theta,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
    )
    return _sqrt_interval_nonnegative(squared)
