"""Fast uncompressed hybrid one-jets for shallow scalar tanh networks.

This module specializes the architecture ``Linear -> Tanh -> Linear`` with a
scalar output.  It contracts the activation-derivative enclosure with the
output layer before expanding the physical-input Jacobian (reverse mode),
retains every domain monomial and every per-neuron approximation-noise
generator, and integrates the resulting squared Jacobian without constructing
its explicit polynomial square.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter, tanh
from time import perf_counter
from typing import Any, Literal

from .interval import Interval
from .polynomial_zonotope import PZOneJet, PolynomialZonotope
from .pz_integration import PZIntegrationCell
from .pz_tanh import (
    affine_tanh_enclosure,
    affine_tanh_prime_enclosure,
    quadratic_tanh_prime_enclosure,
)

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class ShallowHybridOneJetResult:
    """Uncompressed reverse one-jet and its factored integration data."""

    final: PZOneJet
    preactivation_lower: Any
    preactivation_upper: Any
    derivative_approximation_radii: Any
    affine_derivative_approximation_radii: Any
    derivative_degrees: Any
    derivative_relative_slopes: Any
    preactivation_center: Any
    preactivation_coefficients: Any
    value_approximation_radii: Any
    value_center: Any
    value_domain_coefficients: Any
    value_error_generators: Any
    domain_center: Any
    domain_coefficients: Any
    derivative_error_generators: Any
    quadratic_vectors: Any
    quadratic_slopes: Any
    timings: dict[str, float]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for shallow hybrid certification.")


def _shallow_layers(module: Any):
    children = list(module.children()) if isinstance(module, nn.Sequential) else []
    if (
        len(children) != 3
        or not isinstance(children[0], nn.Linear)
        or not isinstance(children[1], nn.Tanh)
        or not isinstance(children[2], nn.Linear)
        or children[2].out_features != 1
        or children[0].out_features != children[2].in_features
    ):
        raise ValueError(
            "The reverse shallow hybrid path requires Linear -> Tanh -> "
            "Linear with one scalar output."
        )
    return children[0], children[2]


def _affine_domain_coefficients(x: PolynomialZonotope) -> tuple[list[tuple[int, ...]], Any]:
    if len(x.shape) != 1:
        raise ValueError("The shallow reverse path requires a flat input PZ.")
    support = sorted(x.terms)
    if not support:
        return support, torch.empty((0, x.shape[0]), dtype=x.center.dtype, device=x.center.device)
    for exponent in support:
        active = [index for index, power in enumerate(exponent) if power]
        if len(active) != 1 or exponent[active[0]] != 1:
            raise ValueError("The shallow reverse path requires an affine input PZ.")
        if x.noise_kinds[active[0]] != "domain":
            raise ValueError("Every active input symbol must be a domain noise symbol.")
    return support, torch.stack([x.terms[exponent] for exponent in support])


def _hybrid_activation_coefficients(
    lower: Any,
    upper: Any,
    *,
    flatness_threshold: float,
    certificate_subdivisions: int,
) -> tuple[Any, ...]:
    value_slopes: list[float] = []
    value_intercepts: list[float] = []
    value_radii: list[float] = []
    constants: list[float] = []
    linears: list[float] = []
    quadratics: list[float] = []
    radii: list[float] = []
    affine_radii: list[float] = []
    degrees: list[int] = []
    relative_slopes: list[float] = []
    for lo, hi in zip(lower.detach().cpu().tolist(), upper.detach().cpu().tolist()):
        interval = Interval(float(lo), float(hi))
        value_affine = affine_tanh_enclosure(interval)
        affine = affine_tanh_prime_enclosure(interval)
        half_width = 0.5 * (float(hi) - float(lo))
        endpoint_lower = 1.0 - tanh(float(lo)) ** 2
        endpoint_upper = 1.0 - tanh(float(hi)) ** 2
        maximum = 1.0 if lo <= 0.0 <= hi else max(endpoint_lower, endpoint_upper)
        interval_radius = 0.5 * (maximum - min(endpoint_lower, endpoint_upper))
        relative_slope = (
            abs(affine.p) * half_width / interval_radius
            if interval_radius > 0.0
            else 0.0
        )
        use_quadratic = lo <= 0.0 <= hi and relative_slope <= flatness_threshold
        quadratic = (
            quadratic_tanh_prime_enclosure(
                Interval(float(lo), float(hi)),
                certificate_subdivisions=certificate_subdivisions,
            )
            if use_quadratic
            else None
        )
        if quadratic is not None and quadratic.delta < affine.delta:
            constant, linear, quadratic_coefficient = quadratic.coeffs
            radius = quadratic.delta
            degree = 2
        else:
            constant, linear, quadratic_coefficient = affine.q, affine.p, 0.0
            radius = affine.delta
            degree = 1
        constants.append(constant)
        linears.append(linear)
        quadratics.append(quadratic_coefficient)
        radii.append(radius)
        affine_radii.append(affine.delta)
        degrees.append(degree)
        relative_slopes.append(relative_slope)
        value_slopes.append(value_affine.p)
        value_intercepts.append(value_affine.q)
        value_radii.append(value_affine.delta)

    options = {"dtype": lower.dtype, "device": lower.device}
    return (
        torch.tensor(value_slopes, **options),
        torch.tensor(value_intercepts, **options),
        torch.tensor(value_radii, **options),
        torch.tensor(constants, **options),
        torch.tensor(linears, **options),
        torch.tensor(quadratics, **options),
        torch.tensor(radii, **options),
        torch.tensor(affine_radii, **options),
        torch.tensor(degrees, dtype=torch.int64, device=lower.device),
        torch.tensor(relative_slopes, **options),
    )


def shallow_scalar_hybrid_onejet_reverse(
    module: Any,
    x: PolynomialZonotope,
    *,
    derivative_flatness_threshold: float = 0.01,
    quadratic_certificate_subdivisions: int = 64,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
) -> ShallowHybridOneJetResult:
    """Build an uncompressed hybrid one-jet by scalar-output reverse mode.

    No monomial support reduction or coefficient boxing is performed.  The
    derivative approximation error of each hidden neuron remains one shared
    pointwise approximation-noise generator across all input derivatives.
    """

    _require_torch()
    if derivative_flatness_threshold < 0.0:
        raise ValueError("derivative_flatness_threshold must be non-negative.")
    first, output = _shallow_layers(module)
    parameter = first.weight
    if not isinstance(x.center, torch.Tensor):
        x = PolynomialZonotope(
            torch.as_tensor(x.center, dtype=parameter.dtype, device=parameter.device),
            {
                exponent: torch.as_tensor(
                    coefficient, dtype=parameter.dtype, device=parameter.device
                )
                for exponent, coefficient in x.terms.items()
            },
            num_noise=x.num_noise,
            noise_kinds=x.noise_kinds,
        )
    support, input_coefficients = _affine_domain_coefficients(x)
    input_dim = x.shape[0]
    if first.in_features != input_dim:
        raise ValueError("Network input dimension does not match the input PZ.")
    if len(support) != input_dim:
        raise ValueError(
            "The shallow reverse path currently requires one affine domain "
            "generator per physical input coordinate."
        )

    # The value and reverse-derivative paths use the same dense hidden
    # preactivation representation and interval hull.  Keep it once rather
    # than invoking the generic value forward and rebuilding W*x+b below.
    del chebyshev_degree, residual_subdivisions
    preparation_start = perf_counter()
    weight_in = first.weight.detach().to(dtype=x.center.dtype, device=x.center.device)
    bias_in = first.bias.detach().to(dtype=x.center.dtype, device=x.center.device)
    weight_out = output.weight.detach()[0].to(dtype=x.center.dtype, device=x.center.device)
    bias_out = output.bias.detach()[0].to(dtype=x.center.dtype, device=x.center.device)

    z_center = weight_in @ x.center + bias_in
    # One row per hidden neuron and one column per retained domain monomial.
    z_coefficients = weight_in @ input_coefficients.T
    z_radius = torch.sum(torch.abs(z_coefficients), dim=1)
    lower = torch.nextafter(z_center - z_radius, torch.full_like(z_center, -torch.inf))
    upper = torch.nextafter(z_center + z_radius, torch.full_like(z_center, torch.inf))
    preparation_seconds = perf_counter() - preparation_start

    certification_start = perf_counter()
    (
        value_slopes,
        value_intercepts,
        value_radii,
        constants,
        linears,
        quadratics,
        radii,
        affine_radii,
        degrees,
        relative_slopes,
    ) = _hybrid_activation_coefficients(
        lower,
        upper,
        flatness_threshold=derivative_flatness_threshold,
        certificate_subdivisions=quadratic_certificate_subdivisions,
    )
    certification_seconds = perf_counter() - certification_start

    value_start = perf_counter()
    hidden_value_center = value_intercepts + value_slopes * z_center
    value_center = torch.dot(weight_out, hidden_value_center) + bias_out
    value_domain_coefficients = z_coefficients.T @ (weight_out * value_slopes)
    value_error_generators = weight_out * value_radii
    value_seconds = perf_counter() - value_start

    construction_start = perf_counter()
    derivative_center = constants + linears * z_center + quadratics * z_center.square()
    derivative_linear = (linears + 2.0 * quadratics * z_center).unsqueeze(1) * z_coefficients
    reverse_vectors = weight_out.unsqueeze(1) * weight_in
    gradient_center = derivative_center @ reverse_vectors
    gradient_linear = derivative_linear.T @ reverse_vectors

    pair_rows, pair_columns = torch.triu_indices(
        len(support), len(support), device=x.center.device
    )
    pair_factor = torch.where(
        pair_rows == pair_columns,
        torch.ones_like(pair_rows, dtype=x.center.dtype),
        torch.full_like(pair_rows, 2.0, dtype=x.center.dtype),
    )
    selected = torch.nonzero(quadratics != 0.0, as_tuple=False).reshape(-1)
    if selected.numel():
        quadratic_slopes = z_coefficients[selected]
        quadratic_vectors = quadratics[selected].unsqueeze(1) * reverse_vectors[selected]
        pair_hidden_coefficients = (
            quadratic_slopes[:, pair_rows]
            * quadratic_slopes[:, pair_columns]
            * pair_factor.unsqueeze(0)
        )
        gradient_quadratic = (
            (pair_hidden_coefficients * quadratics[selected].unsqueeze(1)).T
            @ reverse_vectors[selected]
        )
    else:
        quadratic_slopes = z_coefficients.new_empty((0, len(support)))
        quadratic_vectors = z_coefficients.new_empty((0, input_dim))
        gradient_quadratic = z_coefficients.new_empty((len(pair_rows), input_dim))

    derivative_error_generators = (weight_out * radii).unsqueeze(1) * weight_in
    domain_coefficients = torch.cat((gradient_linear, gradient_quadratic), dim=0)

    total_noise = x.num_noise + 2 * first.out_features
    noise_kinds = x.noise_kinds + ("approximation_pointwise",) * (2 * first.out_features)
    value_terms: dict[tuple[int, ...], Any] = {}
    terms: dict[tuple[int, ...], Any] = {}
    padding = (0,) * (total_noise - x.num_noise)
    for exponent, value_coefficient, gradient_coefficient in zip(
        support, value_domain_coefficients, gradient_linear
    ):
        value_terms[exponent + padding] = value_coefficient.unsqueeze(0)
        terms[exponent + padding] = gradient_coefficient
    for row, column, coefficient in zip(pair_rows.tolist(), pair_columns.tolist(), gradient_quadratic):
        exponent = tuple(a + b for a, b in zip(support[row], support[column]))
        terms[exponent + padding] = coefficient
    for neuron, coefficient in enumerate(value_error_generators):
        exponent = [0] * total_noise
        exponent[x.num_noise + neuron] = 1
        value_terms[tuple(exponent)] = coefficient.unsqueeze(0)
    for neuron, coefficient in enumerate(derivative_error_generators):
        exponent = [0] * total_noise
        exponent[x.num_noise + first.out_features + neuron] = 1
        terms[tuple(exponent)] = coefficient

    value = PolynomialZonotope(
        value_center.unsqueeze(0),
        value_terms,
        num_noise=total_noise,
        noise_kinds=noise_kinds,
    )
    jacobian = PolynomialZonotope(
        gradient_center.unsqueeze(0),
        {exponent: coefficient.unsqueeze(0) for exponent, coefficient in terms.items()},
        num_noise=total_noise,
        noise_kinds=noise_kinds,
    )
    construction_seconds = perf_counter() - construction_start
    return ShallowHybridOneJetResult(
        final=PZOneJet(Y=value, J=jacobian),
        preactivation_lower=lower,
        preactivation_upper=upper,
        derivative_approximation_radii=radii,
        affine_derivative_approximation_radii=affine_radii,
        derivative_degrees=degrees,
        derivative_relative_slopes=relative_slopes,
        preactivation_center=z_center,
        preactivation_coefficients=z_coefficients,
        value_approximation_radii=value_radii,
        value_center=value_center,
        value_domain_coefficients=value_domain_coefficients,
        value_error_generators=value_error_generators,
        domain_center=gradient_center,
        domain_coefficients=domain_coefficients,
        derivative_error_generators=derivative_error_generators,
        quadratic_vectors=quadratic_vectors,
        quadratic_slopes=quadratic_slopes,
        timings={
            "preactivation_preparation": preparation_seconds,
            "activation_certification": certification_seconds,
            "value_construction": value_seconds,
            "reverse_jacobian_construction": construction_seconds,
            "onejet_construction": (
                preparation_seconds
                + certification_seconds
                + value_seconds
                + construction_seconds
            ),
        },
    )


def _value_squared_components(result: ShallowHybridOneJetResult) -> tuple[Any, Any]:
    """Return normalized center/radius for the scalar value square."""

    center = result.value_center
    linear = result.value_domain_coefficients
    errors = result.value_error_generators
    normalized_center = center.square() + torch.sum(linear.square()) / 3.0

    domain_with_center = torch.cat((center.reshape(1), linear))
    domain_error_cross = 2.0 * domain_with_center[:, None] * errors[None, :]
    error_gram = errors[:, None] * errors[None, :]
    off_rows, off_columns = torch.triu_indices(
        len(errors), len(errors), offset=1, device=errors.device
    )
    normalized_radius = (
        torch.sum(torch.abs(domain_error_cross))
        + torch.sum(torch.abs(torch.diagonal(error_gram)))
        + 2.0 * torch.sum(torch.abs(error_gram[off_rows, off_columns]))
    )
    return normalized_center, normalized_radius


def _gradient_squared_components(
    result: ShallowHybridOneJetResult,
) -> tuple[Any, Any]:
    """Return normalized center/radius for the squared gradient."""

    center = result.domain_center
    coefficients = result.domain_coefficients
    errors = result.derivative_error_generators
    input_dim = center.numel()
    linear = coefficients[:input_dim]
    quadratic_vectors = result.quadratic_vectors
    slopes = result.quadratic_slopes

    normalized_center = torch.dot(center, center) + torch.sum(linear.square()) / 3.0
    if len(quadratic_vectors):
        slope_norms = torch.sum(slopes.square(), dim=1)
        normalized_center = normalized_center + (2.0 / 3.0) * torch.sum(
            (quadratic_vectors @ center) * slope_norms
        )
        slope_gram = slopes @ slopes.T
        squared_coordinate_overlap = slopes.square() @ slopes.square().T
        fourth_moments = (
            (slope_norms[:, None] * slope_norms[None, :] + 2.0 * slope_gram.square()) / 9.0
            - (2.0 / 15.0) * squared_coordinate_overlap
        )
        normalized_center = normalized_center + torch.sum(
            (quadratic_vectors @ quadratic_vectors.T) * fourth_moments
        )

    # Canonicalize each alpha^beta eta_i coefficient by one dense contraction
    # before taking absolute values.  The center is the beta=0 row.
    domain_with_center = torch.cat((center.unsqueeze(0), coefficients), dim=0)
    domain_error_cross = 2.0 * (domain_with_center @ errors.T)
    error_gram = errors @ errors.T
    diagonal = torch.diagonal(error_gram)
    off_rows, off_columns = torch.triu_indices(
        len(errors), len(errors), offset=1, device=errors.device
    )
    normalized_radius = (
        torch.sum(torch.abs(domain_error_cross))
        + torch.sum(torch.abs(diagonal))
        + 2.0 * torch.sum(torch.abs(error_gram[off_rows, off_columns]))
    )
    return normalized_center, normalized_radius


def integrate_shallow_hybrid_onejet_squared(
    result: ShallowHybridOneJetResult,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"] = "interval",
):
    """Direct symbolic integral of ``|Y|^2 + |J|_F^2``.

    The domain-only quadratic Jacobian core is integrated using exact second
    and fourth moments of independent ``U[-1,1]`` domain symbols.  Terms that
    contain derivative approximation noise are canonicalized in batched Gram
    products and use the established pointwise-noise integration semantics.
    """

    if output not in {"interval", "pz"}:
        raise ValueError("output must be either 'interval' or 'pz'.")
    if not isinstance(cell.volume, (int, float)):
        raise NotImplementedError("The shallow direct integral requires a scalar cell volume.")
    value_center, value_radius = _value_squared_components(result)
    gradient_center, gradient_radius = _gradient_squared_components(result)

    # Value and derivative approximation errors use disjoint symbol blocks.
    # Their canonical pointwise coefficients therefore cannot collide, while
    # all domain-only contributions can be summed before the single final
    # intervalization.  This is the joint W12 specialization of the direct
    # integrated-square algorithm.
    normalized_center = value_center + gradient_center
    normalized_radius = value_radius + gradient_radius
    integrated_center = float(cell.volume) * float(
        normalized_center.detach().cpu().item()
    )
    integrated_radius = float(cell.volume) * float(
        normalized_radius.detach().cpu().item()
    )
    total = Interval.from_bounds(
        nextafter(integrated_center - integrated_radius, -inf),
        nextafter(integrated_center + integrated_radius, inf),
    )
    if output == "interval":
        return total
    center = total.midpoint
    radius = total.radius
    return PolynomialZonotope.constant(center).add_independent_error(
        radius, kind="global_symbolic_residual"
    )
