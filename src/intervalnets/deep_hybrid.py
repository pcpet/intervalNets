"""Fast factored hybrid one-jets for deep scalar tanh networks.

The expanded polynomial support of a deep Jacobian grows combinatorially even
when every activation enclosure is only affine.  This module therefore keeps
the same polynomial exactly as an arithmetic circuit: cached affine
preactivations feed affine-or-quadratic derivative factors, and the factors are
contracted in reverse order.  No monomial or approximation-noise symbol is
discarded.

For one hidden layer the public dispatcher deliberately calls the specialized
expanded implementation in :mod:`intervalnets.shallow_hybrid`.  The shallow
certificate and its direct integral are therefore recovered exactly.  For two
or more hidden layers the final Jacobian remains factored.  Its norm routine
integrates a retained affine domain core exactly and encloses the unexpanded
higher-order circuit by a pointwise residual.  An independent, certified
operator-norm cap is intersected with that enclosure.  This last integration
step is sound but can be wider than fully expanding and canonicalizing every
deep monomial; it is what makes the depth-generic path scalable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter, sqrt
from time import perf_counter
from typing import Any, Literal

from .interval import Interval
from .polynomial_zonotope import PolynomialZonotope
from .pz_integration import PZIntegrationCell
from .shallow_hybrid import (
    ShallowHybridOneJetResult,
    _affine_domain_coefficients,
    _hybrid_activation_coefficients,
    integrate_shallow_hybrid_onejet_squared,
    shallow_scalar_hybrid_onejet_reverse,
)

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class HybridDerivativeFactor:
    """One unexpanded componentwise derivative PZ factor.

    With ``xi`` denoting the domain and preceding value-approximation symbols,
    and ``eta`` the fresh derivative-approximation symbols, the represented
    factor is

    ``center + linear @ xi + quadratic * (preactivation @ xi)**2 + error * eta``.
    """

    preactivation_center: Any
    preactivation_coefficients: Any
    preactivation_lower: Any
    preactivation_upper: Any
    value_slopes: Any
    value_intercepts: Any
    value_approximation_radii: Any
    center: Any
    linear_coefficients: Any
    quadratic_coefficients: Any
    approximation_radii: Any
    affine_approximation_radii: Any
    approximation_degrees: Any
    relative_slopes: Any
    derivative_lower: Any
    derivative_upper: Any
    active_value_noise: int
    derivative_noise_offset: int


@dataclass(frozen=True)
class FactoredPolynomialJacobian:
    """Exact arithmetic-circuit representation of a scalar-output Jacobian."""

    input_weight: Any
    hidden_weights: tuple[Any, ...]
    output_weight: Any
    factors: tuple[HybridDerivativeFactor, ...]
    num_domain_noise: int
    num_value_noise: int
    num_derivative_noise: int
    interval_lower: Any
    interval_upper: Any

    @property
    def num_noise(self) -> int:
        return self.num_value_noise + self.num_derivative_noise

    @property
    def max_degree(self) -> int:
        return sum(int(factor.approximation_degrees.max().item()) for factor in self.factors)

    def evaluate(self, noise: Any) -> Any:
        """Evaluate the retained polynomial circuit at one or more noise vectors."""

        if torch is None:
            raise ImportError("PyTorch is required for factored Jacobian evaluation.")
        template = self.input_weight
        values = torch.as_tensor(noise, dtype=template.dtype, device=template.device)
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.num_noise:
            raise ValueError(
                f"Expected {self.num_noise} noise coordinates, got {values.shape[-1]}."
            )

        adjoint = self.output_weight.unsqueeze(0).expand(values.shape[0], -1)
        derivative_base = self.num_value_noise
        for layer_index in range(len(self.factors) - 1, -1, -1):
            factor = self.factors[layer_index]
            xi = values[:, : factor.active_value_noise]
            affine_argument = xi @ factor.preactivation_coefficients.T
            derivative = (
                factor.center.unsqueeze(0)
                + xi @ factor.linear_coefficients.T
                + factor.quadratic_coefficients.unsqueeze(0) * affine_argument.square()
            )
            width = factor.center.numel()
            eta = values[
                :,
                derivative_base
                + factor.derivative_noise_offset : derivative_base
                + factor.derivative_noise_offset
                + width,
            ]
            derivative = derivative + factor.approximation_radii.unsqueeze(0) * eta
            adjoint = adjoint * derivative
            if layer_index:
                adjoint = adjoint @ self.hidden_weights[layer_index - 1]
        gradient = adjoint @ self.input_weight
        return gradient.squeeze(0) if squeeze else gradient

    def interval_enclosure(self) -> Interval:
        lower = self.interval_lower.detach().cpu().tolist()
        upper = self.interval_upper.detach().cpu().tolist()
        return Interval.from_bounds(lower, upper)


@dataclass(frozen=True)
class DeepHybridOneJetResult:
    """Depth-generic uncompressed factored one-jet certificate."""

    value: PolynomialZonotope
    jacobian: FactoredPolynomialJacobian
    factors: tuple[HybridDerivativeFactor, ...]
    value_center: Any
    value_domain_coefficients: Any
    value_error_generators: Any
    gradient_affine_center: Any
    gradient_affine_domain_coefficients: Any
    gradient_pointwise_remainder: Any
    gradient_spectral_bound: float
    timings: dict[str, float]


HybridOneJetResult = ShallowHybridOneJetResult | DeepHybridOneJetResult


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for deep hybrid certification.")


def _tanh_scalar_network(module: Any) -> tuple[list[Any], list[Any], Any]:
    """Return alternating affine/activation layers for a scalar tanh MLP."""

    children = list(module.children()) if isinstance(module, nn.Sequential) else []
    if len(children) < 3 or len(children) % 2 != 1:
        raise ValueError(
            "The fast hybrid path requires alternating Linear/Tanh layers and "
            "one final scalar Linear layer."
        )
    linears: list[Any] = []
    activations: list[Any] = []
    for index, child in enumerate(children[:-1]):
        expected = nn.Linear if index % 2 == 0 else nn.Tanh
        if not isinstance(child, expected):
            raise ValueError(
                "The fast hybrid path requires Linear -> Tanh repetitions "
                "followed by one scalar Linear layer."
            )
        (linears if index % 2 == 0 else activations).append(child)
    output = children[-1]
    if not isinstance(output, nn.Linear) or output.out_features != 1:
        raise ValueError("The fast hybrid path requires one scalar Linear output.")
    if len(linears) != len(activations):
        raise ValueError("Every hidden Linear layer must be followed by Tanh.")
    previous = linears[0].out_features
    for layer in linears[1:]:
        if layer.in_features != previous:
            raise ValueError("Adjacent hidden layer dimensions do not match.")
        previous = layer.out_features
    if output.in_features != previous:
        raise ValueError("The output layer dimension does not match the final hidden layer.")
    return linears, activations, output


def _as_torch_pz(module: Any, x: PolynomialZonotope) -> PolynomialZonotope:
    parameter = next(module.parameters())
    if isinstance(x.center, torch.Tensor):
        return x
    return PolynomialZonotope(
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


def _linear_interval_row(lower: Any, upper: Any, weight: Any) -> tuple[Any, Any]:
    positive = torch.clamp(weight, min=0.0)
    negative = torch.clamp(weight, max=0.0)
    return lower @ positive + upper @ negative, upper @ positive + lower @ negative


def _multiply_intervals(
    left_lower: Any,
    left_upper: Any,
    right_lower: Any,
    right_upper: Any,
) -> tuple[Any, Any]:
    candidates = torch.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        )
    )
    return candidates.amin(dim=0), candidates.amax(dim=0)


def _reverse_affine_core_and_interval(
    linears: list[Any],
    output: Any,
    factors: list[HybridDerivativeFactor],
    input_dim: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Retain the constant/linear domain core and enclose the exact circuit."""

    dtype = linears[0].weight.dtype
    device = linears[0].weight.device
    center = output.weight.detach()[0].to(dtype=dtype, device=device)
    domain_linear = torch.zeros(
        (center.numel(), input_dim), dtype=dtype, device=device
    )
    lower = center.clone()
    upper = center.clone()

    for layer_index in range(len(factors) - 1, -1, -1):
        factor = factors[layer_index]
        lower, upper = _multiply_intervals(
            lower,
            upper,
            factor.derivative_lower,
            factor.derivative_upper,
        )
        old_center = center
        old_linear = domain_linear
        derivative_linear = factor.linear_coefficients[:, :input_dim]
        center = old_center * factor.center
        domain_linear = (
            old_center.unsqueeze(1) * derivative_linear
            + factor.center.unsqueeze(1) * old_linear
        )
        if layer_index:
            weight = linears[layer_index].weight.detach().to(dtype=dtype, device=device)
            lower, upper = _linear_interval_row(lower, upper, weight)
            center = center @ weight
            domain_linear = weight.T @ domain_linear

    input_weight = linears[0].weight.detach().to(dtype=dtype, device=device)
    lower, upper = _linear_interval_row(lower, upper, input_weight)
    center = center @ input_weight
    domain_linear = input_weight.T @ domain_linear

    affine_radius = torch.sum(torch.abs(domain_linear), dim=1)
    remainder = torch.maximum(
        torch.abs(lower - (center + affine_radius)),
        torch.abs(upper - (center - affine_radius)),
    )
    remainder = torch.nextafter(remainder, torch.full_like(remainder, torch.inf))
    return center, domain_linear, remainder, lower, upper


def _padded_spectral_norm(matrix: Any) -> float:
    """Return a posteriori padded upper bound for the spectral norm.

    The raw leading singular value is not used on its own.  We reconstruct the
    SVD, bound the reconstruction residual by its Frobenius norm, and bound
    the possible non-orthogonality of the computed singular vectors through
    their Gram residuals.  A deliberately conservative floating-point term is
    added to the three dense contractions.
    """

    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    reconstructed = (u * singular_values.unsqueeze(0)) @ vh
    residual = float(torch.linalg.vector_norm(matrix - reconstructed).item())
    identity = torch.eye(
        singular_values.numel(), dtype=matrix.dtype, device=matrix.device
    )
    u_orthogonality = float(torch.linalg.vector_norm(u.T @ u - identity).item())
    v_orthogonality = float(torch.linalg.vector_norm(vh @ vh.T - identity).item())
    u_bound = sqrt(1.0 + u_orthogonality)
    v_bound = sqrt(1.0 + v_orthogonality)
    leading = float(singular_values[0].item()) if singular_values.numel() else 0.0
    frobenius = float(torch.linalg.vector_norm(matrix).item())
    dimension = max(matrix.shape, default=1)
    eps = torch.finfo(matrix.dtype).eps
    rounding = 1024.0 * eps * dimension * dimension * max(1.0, frobenius)
    return nextafter(u_bound * leading * v_bound + residual + rounding, inf)


def _gradient_spectral_bound(
    linears: list[Any], output: Any, factors: list[HybridDerivativeFactor]
) -> float:
    """Independent global cap using certified per-neuron derivative maxima."""

    output_weight = output.weight.detach()
    bound = _padded_spectral_norm(
        output_weight * factors[-1].derivative_upper.unsqueeze(0)
    )
    for layer_index in range(len(linears) - 1, 0, -1):
        weight = linears[layer_index].weight.detach()
        bound *= _padded_spectral_norm(
            weight * factors[layer_index - 1].derivative_upper.unsqueeze(0)
        )
    bound *= _padded_spectral_norm(linears[0].weight.detach())
    return nextafter(bound, inf)


def deep_scalar_hybrid_onejet_reverse(
    module: Any,
    x: PolynomialZonotope,
    *,
    derivative_flatness_threshold: float = 0.01,
    quadratic_certificate_subdivisions: int = 64,
) -> DeepHybridOneJetResult:
    """Build a fast uncompressed factored hybrid one-jet for a deep MLP."""

    _require_torch()
    if derivative_flatness_threshold < 0.0:
        raise ValueError("derivative_flatness_threshold must be non-negative.")
    linears, _, output = _tanh_scalar_network(module)
    if len(linears) < 2:
        raise ValueError(
            "Use scalar_hybrid_onejet_reverse for the depth-generic dispatcher; "
            "deep_scalar_hybrid_onejet_reverse requires at least two hidden layers."
        )
    x = _as_torch_pz(module, x)
    support, input_coefficients = _affine_domain_coefficients(x)
    input_dim = x.shape[0]
    if x.num_noise != input_dim or len(support) != input_dim:
        raise ValueError(
            "The fast deep path requires one affine domain symbol per input coordinate."
        )
    if linears[0].in_features != input_dim:
        raise ValueError("Network input dimension does not match the input PZ.")

    total_start = perf_counter()
    forward_start = perf_counter()
    value_center = x.center
    value_coefficients = input_coefficients.T
    raw_factors: list[dict[str, Any]] = []
    value_widths: list[int] = []

    for layer in linears:
        weight = layer.weight.detach().to(dtype=x.center.dtype, device=x.center.device)
        bias = layer.bias.detach().to(dtype=x.center.dtype, device=x.center.device)
        z_center = weight @ value_center + bias
        z_coefficients = weight @ value_coefficients
        z_radius = torch.sum(torch.abs(z_coefficients), dim=1)
        lower = torch.nextafter(
            z_center - z_radius, torch.full_like(z_center, -torch.inf)
        )
        upper = torch.nextafter(
            z_center + z_radius, torch.full_like(z_center, torch.inf)
        )
        (
            value_slopes,
            value_intercepts,
            value_radii,
            constants,
            derivative_linears,
            quadratics,
            derivative_radii,
            affine_radii,
            degrees,
            relative_slopes,
        ) = _hybrid_activation_coefficients(
            lower,
            upper,
            flatness_threshold=derivative_flatness_threshold,
            certificate_subdivisions=quadratic_certificate_subdivisions,
        )
        derivative_center = (
            constants
            + derivative_linears * z_center
            + quadratics * z_center.square()
        )
        derivative_linear_coefficients = (
            derivative_linears + 2.0 * quadratics * z_center
        ).unsqueeze(1) * z_coefficients
        endpoint_lower = 1.0 - torch.tanh(lower).square()
        endpoint_upper = 1.0 - torch.tanh(upper).square()
        derivative_lower = torch.minimum(endpoint_lower, endpoint_upper)
        crosses_zero = (lower <= 0.0) & (upper >= 0.0)
        derivative_upper = torch.where(
            crosses_zero,
            torch.ones_like(lower),
            torch.maximum(endpoint_lower, endpoint_upper),
        )
        derivative_lower = torch.nextafter(
            derivative_lower, torch.full_like(derivative_lower, -torch.inf)
        )
        derivative_upper = torch.nextafter(
            derivative_upper, torch.full_like(derivative_upper, torch.inf)
        )
        raw_factors.append(
            {
                "preactivation_center": z_center,
                "preactivation_coefficients": z_coefficients,
                "preactivation_lower": lower,
                "preactivation_upper": upper,
                "value_slopes": value_slopes,
                "value_intercepts": value_intercepts,
                "value_approximation_radii": value_radii,
                "center": derivative_center,
                "linear_coefficients": derivative_linear_coefficients,
                "quadratic_coefficients": quadratics,
                "approximation_radii": derivative_radii,
                "affine_approximation_radii": affine_radii,
                "approximation_degrees": degrees,
                "relative_slopes": relative_slopes,
                "derivative_lower": derivative_lower,
                "derivative_upper": derivative_upper,
                "active_value_noise": value_coefficients.shape[1],
            }
        )
        value_center = value_intercepts + value_slopes * z_center
        value_coefficients = torch.cat(
            (
                value_slopes.unsqueeze(1) * z_coefficients,
                torch.diag(value_radii),
            ),
            dim=1,
        )
        value_widths.append(layer.out_features)

    weight_out = output.weight.detach()[0].to(
        dtype=x.center.dtype, device=x.center.device
    )
    bias_out = output.bias.detach()[0].to(dtype=x.center.dtype, device=x.center.device)
    output_center = torch.dot(weight_out, value_center) + bias_out
    output_coefficients = value_coefficients.T @ weight_out
    forward_seconds = perf_counter() - forward_start

    factors: list[HybridDerivativeFactor] = []
    derivative_offset = 0
    for width, raw in zip(value_widths, raw_factors):
        factors.append(
            HybridDerivativeFactor(
                **raw,
                derivative_noise_offset=derivative_offset,
            )
        )
        derivative_offset += width

    value_noise_count = value_coefficients.shape[1]
    derivative_noise_count = sum(value_widths)
    total_noise = value_noise_count + derivative_noise_count
    noise_kinds = (
        x.noise_kinds
        + ("approximation_pointwise",) * (value_noise_count - x.num_noise)
        + ("approximation_pointwise",) * derivative_noise_count
    )
    padding = (0,) * (total_noise - x.num_noise)
    value_terms: dict[tuple[int, ...], Any] = {}
    for exponent, coefficient in zip(support, output_coefficients[:input_dim]):
        value_terms[exponent + padding] = coefficient.unsqueeze(0)
    for index, coefficient in enumerate(output_coefficients[input_dim:]):
        exponent = [0] * total_noise
        exponent[input_dim + index] = 1
        value_terms[tuple(exponent)] = coefficient.unsqueeze(0)
    value = PolynomialZonotope(
        output_center.unsqueeze(0),
        value_terms,
        num_noise=total_noise,
        noise_kinds=noise_kinds,
    )

    reverse_start = perf_counter()
    gradient_center, gradient_linear, gradient_remainder, j_lower, j_upper = (
        _reverse_affine_core_and_interval(linears, output, factors, input_dim)
    )
    spectral_bound = _gradient_spectral_bound(linears, output, factors)
    reverse_seconds = perf_counter() - reverse_start

    hidden_weights = tuple(
        layer.weight.detach().to(dtype=x.center.dtype, device=x.center.device)
        for layer in linears[1:]
    )
    factored = FactoredPolynomialJacobian(
        input_weight=linears[0].weight.detach().to(
            dtype=x.center.dtype, device=x.center.device
        ),
        hidden_weights=hidden_weights,
        output_weight=weight_out,
        factors=tuple(factors),
        num_domain_noise=input_dim,
        num_value_noise=value_noise_count,
        num_derivative_noise=derivative_noise_count,
        interval_lower=j_lower,
        interval_upper=j_upper,
    )
    total_seconds = perf_counter() - total_start
    return DeepHybridOneJetResult(
        value=value,
        jacobian=factored,
        factors=tuple(factors),
        value_center=output_center,
        value_domain_coefficients=output_coefficients[:input_dim],
        value_error_generators=output_coefficients[input_dim:],
        gradient_affine_center=gradient_center,
        gradient_affine_domain_coefficients=gradient_linear,
        gradient_pointwise_remainder=gradient_remainder,
        gradient_spectral_bound=spectral_bound,
        timings={
            "forward_cache_and_value": forward_seconds,
            "reverse_jacobian_certificate": reverse_seconds,
            "onejet_construction": total_seconds,
        },
    )


def scalar_hybrid_onejet_reverse(
    module: Any,
    x: PolynomialZonotope,
    **kwargs: Any,
) -> HybridOneJetResult:
    """Depth-generic dispatcher that exactly recovers the shallow fast path."""

    _require_torch()
    linears, _, _ = _tanh_scalar_network(module)
    if len(linears) == 1:
        return shallow_scalar_hybrid_onejet_reverse(module, x, **kwargs)
    kwargs.pop("chebyshev_degree", None)
    kwargs.pop("residual_subdivisions", None)
    return deep_scalar_hybrid_onejet_reverse(module, x, **kwargs)


def _deep_value_squared_components(result: DeepHybridOneJetResult) -> tuple[Any, Any]:
    center = result.value_center
    domain = result.value_domain_coefficients
    errors = result.value_error_generators
    diagonal = errors.square()
    normalized_center = (
        center.square()
        + torch.sum(domain.square()) / 3.0
        + 0.5 * torch.sum(diagonal)
    )
    center_cross = 2.0 * center * errors
    # The 2*alpha_j*eta_i coefficients receive E|alpha_j| = 1/2.
    domain_cross = domain[:, None] * errors[None, :]
    gram = errors[:, None] * errors[None, :]
    rows, columns = torch.triu_indices(
        len(errors), len(errors), offset=1, device=errors.device
    )
    normalized_radius = (
        torch.sum(torch.abs(center_cross))
        + torch.sum(torch.abs(domain_cross))
        + 0.5 * torch.sum(torch.abs(torch.diagonal(gram)))
        + 2.0 * torch.sum(torch.abs(gram[rows, columns]))
    )
    return normalized_center, normalized_radius


def _deep_interval_from_normalized_bounds(
    lower: float,
    upper: float,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"],
):
    if output not in {"interval", "pz"}:
        raise ValueError("output must be either 'interval' or 'pz'.")
    if not isinstance(cell.volume, (int, float)):
        raise NotImplementedError("The deep direct integral requires a scalar cell volume.")
    volume = float(cell.volume)
    total = Interval.from_bounds(
        nextafter(volume * lower, -inf),
        nextafter(volume * upper, inf),
    )
    if output == "interval":
        return total
    return PolynomialZonotope.constant(total.midpoint).add_independent_error(
        total.radius, kind="global_symbolic_residual"
    )


def integrate_deep_hybrid_value_squared(
    result: DeepHybridOneJetResult,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"] = "interval",
):
    """Integrate the deep affine value PZ with refined pointwise moments."""

    center, radius = _deep_value_squared_components(result)
    lower = max(0.0, float((center - radius).detach().cpu().item()))
    upper = max(0.0, float((center + radius).detach().cpu().item()))
    return _deep_interval_from_normalized_bounds(lower, upper, cell, output=output)


def integrate_deep_hybrid_onejet_squared(
    result: DeepHybridOneJetResult,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"] = "interval",
):
    """Integrate the deep factored ``|Y|^2 + |J|_F^2`` enclosure.

    The affine domain core is integrated using exact uniform-box moments.  The
    higher-order factored circuit is retained without support reduction and is
    enclosed as one pointwise residual for this integration functional.  The
    Jacobian upper bound is intersected with an independent spectral bound.
    """

    value_center, value_radius = _deep_value_squared_components(result)
    gradient_center = result.gradient_affine_center
    gradient_linear = result.gradient_affine_domain_coefficients
    gradient_remainder = result.gradient_pointwise_remainder
    gradient_moment_center = (
        torch.dot(gradient_center, gradient_center)
        + torch.sum(gradient_linear.square()) / 3.0
        + 0.5 * torch.sum(gradient_remainder.square())
    )
    # Apply the same coefficientwise functional after collapsing the exact
    # factored higher-order circuit to the certified pointwise remainder R.
    # Constant*R uses moment 1, alpha_j*R uses E|alpha_j|=1/2, and R**2 is
    # one-sided because it is non-negative.
    gradient_radius = torch.sum(
        2.0 * torch.abs(gradient_center) * gradient_remainder
        + torch.sum(torch.abs(gradient_linear), dim=1) * gradient_remainder
        + 0.5 * gradient_remainder.square()
    )
    gradient_lower = max(
        0.0,
        float((gradient_moment_center - gradient_radius).detach().cpu().item()),
    )
    envelope_upper = max(
        0.0,
        float((gradient_moment_center + gradient_radius).detach().cpu().item()),
    )
    spectral_upper = result.gradient_spectral_bound**2
    gradient_upper = min(envelope_upper, spectral_upper)
    value_lower = max(
        0.0, float((value_center - value_radius).detach().cpu().item())
    )
    value_upper = max(
        0.0, float((value_center + value_radius).detach().cpu().item())
    )
    return _deep_interval_from_normalized_bounds(
        value_lower + gradient_lower,
        value_upper + gradient_upper,
        cell,
        output=output,
    )


def integrate_hybrid_onejet_squared(
    result: HybridOneJetResult,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"] = "interval",
):
    """Depth-generic dispatcher for the optimized hybrid integral."""

    if isinstance(result, ShallowHybridOneJetResult):
        return integrate_shallow_hybrid_onejet_squared(result, cell, output=output)
    return integrate_deep_hybrid_onejet_squared(result, cell, output=output)


def integrate_hybrid_value_squared(
    result: HybridOneJetResult,
    cell: PZIntegrationCell,
    *,
    output: Literal["interval", "pz"] = "interval",
):
    """Depth-generic dispatcher for the optimized hybrid L2 integral."""

    if isinstance(result, ShallowHybridOneJetResult):
        from .shallow_hybrid import integrate_shallow_hybrid_value_squared

        return integrate_shallow_hybrid_value_squared(result, cell, output=output)
    return integrate_deep_hybrid_value_squared(result, cell, output=output)
