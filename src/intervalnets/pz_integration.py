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

try:  # pragma: no cover - optional acceleration
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

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


def _is_real_scalar_coefficient(value: Any) -> bool:
    if isinstance(value, Real):
        return True
    return bool(
        torch is not None
        and isinstance(value, torch.Tensor)
        and value.numel() == 1
        and not value.is_complex()
    )


def _real_scalar_value(value: Any) -> float:
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _all_real_scalar_coefficients(centers: Sequence[Any], matrix: Sequence[Sequence[Any]]) -> bool:
    """Return whether a possibly mixed coefficient table can use NumPy."""

    return all(_is_real_scalar_coefficient(value) for value in centers) and all(
        _is_real_scalar_coefficient(value) for row in matrix for value in row
    )


def _integrate_numpy_twojet_square(
    *,
    support: Sequence[Exponent],
    num_noise: int,
    center_square: float,
    center_cross: Any,
    gram: Any,
    scale: float,
    measure: float,
    domain_indices: tuple[int, ...],
    retained_indices: tuple[int, ...],
    pointwise_indices: tuple[int, ...],
):
    """Canonicalize and integrate the dense-real contraction in vectorized batches."""

    support_size = len(support)
    support_array = np.asarray(support, dtype=np.int64).reshape(support_size, num_noise)

    # The affine activation-enclosure pipeline keeps the value PZ affine:
    # every support exponent is one distinct unit vector.  In this important
    # case, all canonical squared terms are known a priori and their box
    # moments can be contracted directly.  Avoid constructing and sorting the
    # O(m^2 * num_noise) pair-exponent table, which is especially expensive for
    # 100-dimensional inputs with many approximation symbols.
    affine_support = bool(
        support_size
        and num_noise
        and np.all((support_array == 0) | (support_array == 1))
        and np.all(support_array.sum(axis=1) == 1)
    )
    if affine_support:
        support_noise_indices = np.argmax(support_array, axis=1)
        if len(np.unique(support_noise_indices)) == support_size:
            domain_lookup = np.zeros(num_noise, dtype=bool)
            domain_lookup[np.asarray(domain_indices, dtype=np.int64)] = True
            pointwise_lookup = np.zeros(num_noise, dtype=bool)
            pointwise_lookup[np.asarray(pointwise_indices, dtype=np.int64)] = True

            support_domain = domain_lookup[support_noise_indices]
            support_pointwise = pointwise_lookup[support_noise_indices]
            support_symbolic = ~(support_domain | support_pointwise)

            diagonal = np.diag(gram)
            integrated_center = float(
                scale
                * measure
                * (
                    center_square
                    + diagonal[support_domain].sum() / 3.0
                )
            )

            pointwise_radius_unscaled = float(
                np.abs(2.0 * np.asarray(center_cross)[support_pointwise]).sum()
                + np.abs(diagonal[support_pointwise]).sum()
            )
            symbolic_radius_unscaled = float(
                np.abs(2.0 * np.asarray(center_cross)[support_symbolic]).sum()
                + np.abs(diagonal[support_symbolic]).sum()
            )

            row_indices, column_indices = np.triu_indices(support_size, k=1)
            off_diagonal = 2.0 * gram[row_indices, column_indices]
            pair_pointwise = (
                support_pointwise[row_indices]
                | support_pointwise[column_indices]
            )
            pair_symbolic = (
                support_symbolic[row_indices]
                & support_symbolic[column_indices]
            )
            pointwise_radius_unscaled += float(
                np.abs(off_diagonal[pair_pointwise]).sum()
            )
            symbolic_radius_unscaled += float(
                np.abs(off_diagonal[pair_symbolic]).sum()
            )

            pointwise_radius = float(
                scale * measure * pointwise_radius_unscaled
            )
            symbolic_radius = float(
                scale * measure * symbolic_radius_unscaled
            )
            base = Interval.from_bounds(
                nextafter(integrated_center - symbolic_radius, -inf),
                nextafter(integrated_center + symbolic_radius, inf),
            )
            return base + Interval.from_bounds(
                -pointwise_radius,
                pointwise_radius,
            )

    row_indices, column_indices = np.triu_indices(support_size)
    pair_coefficients = gram[row_indices, column_indices].copy()
    pair_coefficients[row_indices != column_indices] *= 2.0

    all_coefficients = np.concatenate(
        (
            np.asarray((center_square,), dtype=float),
            2.0 * np.asarray(center_cross, dtype=float),
            pair_coefficients,
        )
    )

    # Canonicalize before any absolute value, exactly as required for
    # pointwise approximation residuals. Encode exponent rows as collision-free
    # mixed-radix int64 keys whenever possible. Pair exponents then correspond
    # exactly to adding their keys, and NumPy sorts eight-byte integers instead
    # of repeatedly comparing full exponent rows.
    maximum_pair_exponents = (
        2 * support_array.max(axis=0)
        if support_size
        else np.zeros(num_noise, dtype=np.int64)
    )
    bases = maximum_pair_exponents + 1
    strides_list: list[int] = []
    capacity = 1
    for base in bases:
        strides_list.append(capacity)
        capacity *= int(base)
    if capacity <= np.iinfo(np.int64).max:
        strides = np.asarray(strides_list, dtype=np.int64)
        support_codes = support_array @ strides
        pair_codes = support_codes[row_indices] + support_codes[column_indices]
        all_codes = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                support_codes,
                pair_codes,
            )
        )
        canonical_codes, inverse = np.unique(all_codes, return_inverse=True)
        canonical_exponents = (
            (canonical_codes[:, np.newaxis] // strides[np.newaxis, :])
            % bases[np.newaxis, :]
        )
    else:
        # Extremely wide supports may exceed a signed 64-bit mixed-radix key.
        # Keep an exact row-based path for those cases.
        pair_exponents = support_array[row_indices] + support_array[column_indices]
        all_exponents = np.concatenate(
            (
                np.zeros((1, num_noise), dtype=np.int64),
                support_array,
                pair_exponents,
            ),
            axis=0,
        )
        canonical_exponents, inverse = np.unique(
            all_exponents,
            axis=0,
            return_inverse=True,
        )
    canonical_coefficients = np.bincount(
        inverse,
        weights=all_coefficients,
        minlength=len(canonical_exponents),
    )

    if pointwise_indices:
        pointwise_mask = np.any(canonical_exponents[:, pointwise_indices] != 0, axis=1)
    else:
        pointwise_mask = np.zeros(len(canonical_exponents), dtype=bool)
    radius = float(scale * measure * np.abs(canonical_coefficients[pointwise_mask]).sum())

    exact_exponents = canonical_exponents[~pointwise_mask]
    exact_coefficients = canonical_coefficients[~pointwise_mask]
    if domain_indices:
        domain_exponents = exact_exponents[:, domain_indices]
        even_mask = np.all(domain_exponents % 2 == 0, axis=1)
        exact_exponents = exact_exponents[even_mask]
        exact_coefficients = exact_coefficients[even_mask]
        domain_exponents = domain_exponents[even_mask]
        moments = np.prod(2.0 / (domain_exponents + 1.0), axis=1)
    else:
        moments = np.ones(len(exact_coefficients), dtype=float)
    integrated_coefficients = scale * moments * exact_coefficients

    if retained_indices:
        retained_exponents = exact_exponents[:, retained_indices]
        retained_bases = bases[np.asarray(retained_indices)]
        retained_strides_list: list[int] = []
        retained_capacity = 1
        for base in retained_bases:
            retained_strides_list.append(retained_capacity)
            retained_capacity *= int(base)
        encoded_retained = retained_capacity <= np.iinfo(np.int64).max
        if encoded_retained:
            retained_strides = np.asarray(retained_strides_list, dtype=np.int64)
            retained_codes = retained_exponents @ retained_strides
            canonical_retained_codes, retained_inverse = np.unique(
                retained_codes,
                return_inverse=True,
            )
            retained_group_count = len(canonical_retained_codes)
        else:
            canonical_retained, retained_inverse = np.unique(
                retained_exponents,
                axis=0,
                return_inverse=True,
            )
            retained_group_count = len(canonical_retained)
        canonical_integrated = np.bincount(
            retained_inverse,
            weights=integrated_coefficients,
            minlength=retained_group_count,
        )
        zero_mask = (
            canonical_retained_codes == 0
            if encoded_retained
            else np.all(canonical_retained == 0, axis=1)
        )
        center = float(canonical_integrated[zero_mask].sum())
        symbolic_radius = float(np.abs(canonical_integrated[~zero_mask]).sum())
    else:
        center = float(integrated_coefficients.sum())
        symbolic_radius = 0.0

    # Match ``IntegratedPZResult.interval_enclosure`` without materializing a
    # retained PolynomialZonotope containing tens of thousands of terms.
    base = Interval.from_bounds(
        nextafter(center - symbolic_radius, -inf),
        nextafter(center + symbolic_radius, inf),
    )
    return base + Interval.from_bounds(-radius, radius)


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
    gram = None
    if torch is not None and isinstance(centers, torch.Tensor):
        weight_vector = torch.tensor(weights, dtype=centers.dtype, device=centers.device)
        weighted_matrix = matrix * weight_vector.unsqueeze(0)
        center_cross = weighted_matrix @ centers
        gram = weighted_matrix @ matrix.T
        center_square = torch.dot(centers * weight_vector, centers)
        if (
            np is not None
            and not centers.is_complex()
            and not matrix.is_complex()
        ):
            return _integrate_numpy_twojet_square(
                support=support,
                num_noise=num_noise,
                center_square=float(center_square.detach().cpu().item()),
                center_cross=center_cross.detach().cpu().numpy(),
                gram=gram.detach().cpu().numpy(),
                scale=scale,
                measure=measure,
                domain_indices=domain_indices,
                retained_indices=retained_indices,
                pointwise_indices=pointwise_indices,
            )
    elif np is not None and _all_real_scalar_coefficients(centers, matrix):
        # Affine PZ propagation may produce a mixture of lightweight floats and
        # zero-dimensional torch tensors. Normalize them once, then use one
        # BLAS contraction instead of millions of Python scalar operations.
        center_vector = np.fromiter(
            (_real_scalar_value(value) for value in centers),
            dtype=float,
            count=len(centers),
        )
        coefficient_matrix = np.fromiter(
            (_real_scalar_value(value) for row in matrix for value in row),
            dtype=float,
            count=len(matrix) * len(centers),
        ).reshape(len(matrix), len(centers))
        weight_vector = np.asarray(weights, dtype=float)
        weighted_matrix = coefficient_matrix * weight_vector[np.newaxis, :]
        center_cross = weighted_matrix @ center_vector
        gram = weighted_matrix @ coefficient_matrix.T
        center_square = float(np.dot(center_vector * weight_vector, center_vector))
        return _integrate_numpy_twojet_square(
            support=support,
            num_noise=num_noise,
            center_square=center_square,
            center_cross=center_cross,
            gram=gram,
            scale=scale,
            measure=measure,
            domain_indices=domain_indices,
            retained_indices=retained_indices,
            pointwise_indices=pointwise_indices,
        )
    else:
        center_cross = [_weighted_dot(row, centers, weights) for row in matrix]
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
        center_coefficient = center_cross[index]
        if np is not None and isinstance(center_coefficient, np.generic):
            center_coefficient = float(center_coefficient)
        route(exponent, _mul_coeff(center_coefficient, 2.0))
        for other_index in range(index, len(support)):
            pair_exponent = tuple(a + b for a, b in zip(exponent, support[other_index]))
            factor = 1.0 if index == other_index else 2.0
            if gram is None:
                gram_coefficient = _weighted_dot(matrix[index], matrix[other_index], weights)
            else:
                gram_coefficient = gram[index][other_index]
                if np is not None and isinstance(gram_coefficient, np.generic):
                    gram_coefficient = float(gram_coefficient)
            route(pair_exponent, _mul_coeff(gram_coefficient, factor))

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


def integrate_pz_value_squared(
    value: PolynomialZonotope,
    cell: PZIntegrationCell,
):
    """Directly integrate the squared Euclidean norm of a value-only PZ.

    The lightweight zero components adapt the existing weighted-coordinate
    contraction without allocating input-dimensional Jacobian or Hessian
    tensors.
    """

    zero = PolynomialZonotope.constant(
        0.0,
        num_noise=value.num_noise,
        noise_kinds=value.noise_kinds,
    )
    return integrate_pz_twojet_squared(
        PZTwoJet(Y=value, J=zero, H=zero),
        cell,
        "l2",
    )


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


def _eval_pz_value(
    model,
    domain: PolynomialZonotope,
    *,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
):
    if hasattr(model, "eval_pz_value"):
        return model.eval_pz_value(
            domain,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
    from .pytorch import pz_value_forward

    return pz_value_forward(
        model,
        domain,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
    )


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

    The affine PZ integration cell, two-jet enclosure, directly integrated
    squared contribution, Jacobian enclosure, and preferred split dimension
    are computed exactly once for the cell lifetime. Refinement discards only
    marked parent cells and computes fresh cache entries for their children.
    """

    cell = PZIntegrationCell.from_affine_box(box)
    if integrand_kind == "l2":
        value = _eval_pz_value(
            model,
            cell.domain,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
        contribution = integrate_pz_value_squared(value, cell)
        jacobian = None
    else:
        jet = _eval_pz_twojet(
            model,
            cell.domain,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
        contribution = integrate_pz_twojet_squared(jet, cell, integrand_kind)
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
    """Adaptive value-only PZ enclosure of the L2 norm over an interval domain."""

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
