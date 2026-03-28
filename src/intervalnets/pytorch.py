from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, isfinite, log, nextafter, tanh
from typing import Any

from .interval import Interval

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    nn = None


@dataclass(frozen=True)
class IntervalTensor(Interval):
    """Tensor-shaped interval wrapper for PyTorch interop."""

    @classmethod
    def point(cls, value: Any) -> "IntervalTensor":
        if torch is not None and isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        base = Interval.point(value)
        return cls(base.lower, base.upper)

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "IntervalTensor":
        if torch is not None and isinstance(lower, torch.Tensor):
            lower = lower.detach().cpu().tolist()
        if torch is not None and isinstance(upper, torch.Tensor):
            upper = upper.detach().cpu().tolist()
        base = Interval.from_bounds(lower, upper)
        return cls(base.lower, base.upper)

    def to_torch(self, dtype=None):
        if torch is None:
            raise ImportError("PyTorch is required for IntervalTensor.to_torch().")
        dtype = dtype or torch.float64
        return torch.tensor(self.lower, dtype=dtype), torch.tensor(self.upper, dtype=dtype)


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for interval neural network evaluation.")


def _pad_outward(value: float, direction: float, steps: int = 1, include_float32: bool = False) -> float:
    """Move a scalar bound outward by a small, controlled amount."""
    out = float(value)
    for _ in range(max(1, steps)):
        out = nextafter(out, direction)

    if include_float32 and torch is not None:
        base32 = torch.tensor(value, dtype=torch.float32)
        target32 = torch.tensor(float("-inf") if direction < 0 else float("inf"), dtype=torch.float32)
        neighbor32 = float(torch.nextafter(base32, target32).item())
        if direction < 0 and neighbor32 < out:
            out = neighbor32
        if direction > 0 and neighbor32 > out:
            out = neighbor32

    return out


def _pad_outward_tensor(values, direction: float, steps: int = 1, include_float32: bool = False):
    """Vectorized companion to _pad_outward for tensor inputs."""
    out = values.to(dtype=torch.float64)
    direction_tensor = torch.full_like(out, float("-inf") if direction < 0 else float("inf"))
    for _ in range(max(1, steps)):
        out = torch.nextafter(out, direction_tensor)

    if include_float32:
        values32 = values.to(dtype=torch.float32)
        direction32 = torch.full_like(values32, float("-inf") if direction < 0 else float("inf"))
        neighbor32 = torch.nextafter(values32, direction32).to(dtype=torch.float64)
        if direction < 0:
            out = torch.minimum(out, neighbor32)
        else:
            out = torch.maximum(out, neighbor32)

    return out


def _widen_weight_bounds(weight):
    """Return outward-rounded [lower, upper] coefficient bounds and whether widening occurred."""
    coefficients = weight.detach().cpu().to(dtype=torch.float64)
    lower = coefficients.clone()
    upper = coefficients.clone()
    widened = False

    if weight.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        finite = torch.isfinite(coefficients)
        nonzero = coefficients != 0.0
        non_integer = coefficients != torch.trunc(coefficients)
        mask = finite & nonzero & non_integer
        if torch.any(mask):
            lower[mask] = _pad_outward_tensor(coefficients[mask], -inf, include_float32=True)
            upper[mask] = _pad_outward_tensor(coefficients[mask], inf, include_float32=True)
            widened = True

    return lower, upper, widened




def _needs_float32_padding_scalar(value: float) -> bool:
    return isfinite(value) and value != 0.0 and not float(value).is_integer()


def _float32_dot_roundoff_bound(sum_abs_products: float, operation_count: int) -> float:
    """Conservative absolute roundoff bound for float32 multiply/add chains."""
    if operation_count <= 0 or not isfinite(sum_abs_products) or sum_abs_products <= 0.0:
        return 0.0
    u = 2.0**-24  # float32 unit roundoff
    gamma = (operation_count * u) / max(1.0 - operation_count * u, 1e-12)
    return gamma * sum_abs_products

def _weight_needs_widening(weight) -> bool:
    if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    coefficients = weight.detach().cpu().to(dtype=torch.float64)
    finite = torch.isfinite(coefficients)
    nonzero = coefficients != 0.0
    non_integer = coefficients != torch.trunc(coefficients)
    return bool(torch.any(finite & nonzero & non_integer))


def _scalar_interval_from_weight(weight: Any, value: Interval, widen_float32: bool = False) -> Interval:
    if torch is not None and isinstance(weight, torch.Tensor):
        scalar = weight.detach().cpu()
        if scalar.numel() != 1:
            raise ValueError("Expected a scalar weight tensor.")
        coefficient = float(scalar.item())
        if (
            widen_float32
            and scalar.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and coefficient != 0.0
            and not coefficient.is_integer()
        ):
            lower = _pad_outward(coefficient, -inf, include_float32=True)
            upper = _pad_outward(coefficient, inf, include_float32=True)
            return Interval.from_bounds(lower, upper) * value
        return Interval.point(coefficient) * value
    return Interval.point(weight) * value


def _apply_monotone_bounds(x: IntervalTensor, func) -> IntervalTensor:
    # For monotone activations f, interval images satisfy
    # f([l, u]) = [f(l), f(u)].
    # Therefore, evaluating only endpoints is sound and complete.
    include_float32 = len(x.lower) == 1
    lower = tuple(_pad_outward(func(bound), -inf, include_float32=include_float32) for bound in x.lower)
    upper = tuple(_pad_outward(func(bound), inf, include_float32=include_float32) for bound in x.upper)
    return IntervalTensor(lower, upper)


def _apply_monotone_bounds_vectorized(x: IntervalTensor, func) -> IntervalTensor:
    lower_tensor = torch.tensor(x.lower, dtype=torch.float64)
    upper_tensor = torch.tensor(x.upper, dtype=torch.float64)
    lower_eval = func(lower_tensor)
    upper_eval = func(upper_tensor)
    include_float32 = len(x.lower) == 1
    lower_out = _pad_outward_tensor(lower_eval, -inf, include_float32=include_float32)
    upper_out = _pad_outward_tensor(upper_eval, inf, include_float32=include_float32)
    return IntervalTensor.from_bounds(lower_out.tolist(), upper_out.tolist())


def _relu_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds_vectorized(x, lambda values: torch.clamp(values, min=0.0))


def _sigmoid_scalar(value: float) -> float:
    """Compute sigmoid(value) for scalar intervals.

    Sigmoid is strictly increasing on R, so interval propagation can evaluate
    the lower endpoint for the lower bound and the upper endpoint for the
    upper bound before outward rounding is applied.
    """
    return float(torch.sigmoid(torch.tensor(value, dtype=torch.float64)).item())


def _sigmoid_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds_vectorized(x, torch.sigmoid)


def _tanh_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds_vectorized(x, torch.tanh)


def _softplus_forward(layer, x: IntervalTensor) -> IntervalTensor:
    beta = float(layer.beta)
    threshold = float(layer.threshold)

    return _apply_monotone_bounds_vectorized(
        x,
        lambda values: torch.nn.functional.softplus(values, beta=beta, threshold=threshold),
    )


def _leaky_relu_forward(layer, x: IntervalTensor) -> IntervalTensor:
    slope = float(layer.negative_slope)
    return _apply_monotone_bounds_vectorized(
        x,
        lambda values: torch.where(values >= 0.0, values, slope * values),
    )


class IntervalAdd(nn.Module):
    """Add outputs of two branches receiving the same input."""

    def __init__(self, left: nn.Module, right: nn.Module) -> None:
        super().__init__()
        self.left = left
        self.right = right

    def forward(self, x):
        return self.left(x) + self.right(x)


class IntervalCat(nn.Module):
    """Concatenate outputs of multiple branches receiving the same input."""

    def __init__(self, *branches: nn.Module, dim: int = -1) -> None:
        super().__init__()
        if not branches:
            raise ValueError("IntervalCat requires at least one branch.")
        self.branches = nn.ModuleList(branches)
        self.dim = dim

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=self.dim)


def _logsumexp(values: tuple[float, ...]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    pivot = max(values)
    if pivot == -inf:
        return -inf
    return pivot + log(sum(exp(value - pivot) for value in values))


def _softmax_component_bounds(index: int, lower: tuple[float, ...], upper: tuple[float, ...]) -> tuple[float, float]:
    # Exact box extrema for component i:
    # min uses x_i=lower_i and x_j=upper_j (j != i),
    # max uses x_i=upper_i and x_j=lower_j (j != i).
    lower_terms = tuple(lower[index] if j == index else upper[j] for j in range(len(lower)))
    upper_terms = tuple(upper[index] if j == index else lower[j] for j in range(len(lower)))

    lower_log_ratio = lower[index] - _logsumexp(lower_terms)
    upper_log_ratio = upper[index] - _logsumexp(upper_terms)

    lower_value = exp(lower_log_ratio)
    upper_value = exp(upper_log_ratio)

    # Apply two outward steps to absorb log/exp rounding, then clamp to
    # the probabilistic range [0, 1].
    lower_out = nextafter(nextafter(lower_value, -inf), -inf)
    upper_out = nextafter(nextafter(upper_value, inf), inf)
    return max(0.0, lower_out), min(1.0, upper_out)


def _softmax_forward(layer, x: IntervalTensor) -> IntervalTensor:
    if len(x.shape) != 1:
        raise NotImplementedError("Interval Softmax currently supports 1D vectors only.")

    dim = layer.dim
    n = len(x.lower)
    # Older PyTorch models may carry dim=None; for 1D vectors this is equivalent
    # to applying softmax over the single axis.
    normalized_dim = -1 if dim is None else dim
    if normalized_dim not in (-1, 0):
        raise NotImplementedError(
            f"Interval Softmax currently supports dim=-1/0 for 1D vectors only; got dim={dim}."
        )
    if n == 0:
        raise ValueError("Softmax input interval must be non-empty.")

    lower = tuple(float(v) for v in x.lower)
    upper = tuple(float(v) for v in x.upper)

    lower_out: list[float] = []
    upper_out: list[float] = []
    for index in range(n):
        component_lower, component_upper = _softmax_component_bounds(index, lower, upper)
        lower_out.append(component_lower)
        upper_out.append(component_upper)

    return IntervalTensor(tuple(lower_out), tuple(upper_out))


def _interval_add(left: IntervalTensor, right: IntervalTensor) -> IntervalTensor:
    if left.shape != right.shape:
        raise ValueError(f"IntervalAdd requires matching shapes, got {left.shape} and {right.shape}.")
    lower = tuple(l + r for l, r in zip(left.lower, right.lower))
    upper = tuple(l + r for l, r in zip(left.upper, right.upper))
    return IntervalTensor.from_bounds(lower, upper)


def _interval_cat(intervals: list[IntervalTensor], dim: int) -> IntervalTensor:
    if not intervals:
        raise ValueError("IntervalCat requires at least one interval input.")
    if any(len(item.shape) != 1 for item in intervals):
        raise NotImplementedError("IntervalCat currently supports flat vectors only.")

    normalized_dim = dim if dim >= 0 else dim + 1
    if normalized_dim != 0:
        raise NotImplementedError(f"IntervalCat currently supports dim=0/-1 for 1D vectors only; got dim={dim}.")

    lower: list[float] = []
    upper: list[float] = []
    for item in intervals:
        lower.extend(float(value) for value in item.lower)
        upper.extend(float(value) for value in item.upper)
    return IntervalTensor.from_bounds(lower, upper)


def _linear_forward(layer, x: IntervalTensor) -> IntervalTensor:
    # Perform outward-rounded accumulation per term to preserve enclosure
    # soundness, while avoiding Interval object churn in inner loops.
    weight = layer.weight.detach().cpu()
    # Coefficients stored in model parameters are exact constants for the
    # interval problem; do not widen them into uncertainty intervals.
    weight_lower = None
    weight_upper = None
    bias = layer.bias.detach().cpu() if layer.bias is not None else None

    x_lower = [float(value) for value in x.lower]
    x_upper = [float(value) for value in x.upper]

    out_lower: list[float] = []
    out_upper: list[float] = []
    for row_idx in range(weight.shape[0]):
        acc_lower = 0.0
        acc_upper = 0.0
        sum_abs_products = 0.0
        float32_guard = False

        for col_idx in range(weight.shape[1]):
            coefficient = float(weight[row_idx, col_idx])
            coef_lower = float(weight_lower[row_idx, col_idx]) if weight_lower is not None else coefficient
            coef_upper = float(weight_upper[row_idx, col_idx]) if weight_upper is not None else coefficient
            left = x_lower[col_idx]
            right = x_upper[col_idx]

            float32_guard = float32_guard or any(
                _needs_float32_padding_scalar(value) for value in (coef_lower, coef_upper)
            )
            candidates = (
                coef_lower * left,
                coef_lower * right,
                coef_upper * left,
                coef_upper * right,
            )
            product_lower = nextafter(min(candidates), -inf)
            product_upper = nextafter(max(candidates), inf)
            sum_abs_products += max(abs(candidate) for candidate in candidates)
            acc_lower = nextafter(acc_lower + product_lower, -inf)
            acc_upper = nextafter(acc_upper + product_upper, inf)

        if bias is not None:
            bias_value = float(bias[row_idx])
            float32_guard = float32_guard or _needs_float32_padding_scalar(bias_value)
            sum_abs_products += abs(bias_value)
            acc_lower = nextafter(acc_lower + bias_value, -inf)
            acc_upper = nextafter(acc_upper + bias_value, inf)

        if float32_guard:
            operation_count = 2 * weight.shape[1] + (1 if bias is not None else 0)
            roundoff = _float32_dot_roundoff_bound(sum_abs_products, operation_count)
            acc_lower = nextafter(acc_lower - roundoff, -inf)
            acc_upper = nextafter(acc_upper + roundoff, inf)

        out_lower.append(_pad_outward(acc_lower, -inf, include_float32=False))
        out_upper.append(_pad_outward(acc_upper, inf, include_float32=False))

    return IntervalTensor.from_bounds(out_lower, out_upper)


def _interval_abs_bounds(value: Interval) -> Interval:
    if isinstance(value.lower, tuple) or isinstance(value.upper, tuple):
        raise ValueError("Expected a scalar interval.")
    lower = float(value.lower)
    upper = float(value.upper)
    if lower <= 0.0 <= upper:
        return Interval.from_bounds(0.0, max(abs(lower), abs(upper)))
    candidates = (abs(lower), abs(upper))
    return Interval.from_bounds(min(candidates), max(candidates))


def _interval_pow_scalar(value: Interval, exponent: float) -> Interval:
    if isinstance(value.lower, tuple) or isinstance(value.upper, tuple):
        raise ValueError("Expected a scalar interval.")
    # Intervals produced by outward rounding can dip slightly below zero
    # (e.g. lower = nextafter(0, -inf)) even when the exact quantity is
    # mathematically non-negative. Clamp these artifacts to zero before
    # exponentiation.
    lower_bound = max(0.0, float(value.lower))
    upper_bound = max(0.0, float(value.upper))
    lower = lower_bound ** exponent
    upper = upper_bound ** exponent
    return Interval.from_bounds(lower, upper)


def _box_volume(box: IntervalTensor) -> float:
    if len(box.shape) != 1:
        raise NotImplementedError("Lp integration currently supports 1D boxes only.")
    volume = 1.0
    for lower, upper in zip(box.lower, box.upper):
        width = float(upper - lower)
        if width < 0.0:
            raise ValueError("Box widths must be non-negative.")
        volume *= width
    return volume


def _lp_pointwise_power_bounds(
    model,
    box: IntervalTensor,
    p: float,
) -> Interval:
    output = model.eval(box)
    components = [Interval(lb, ub) for lb, ub in zip(output.lower, output.upper)]
    total = Interval.point(0.0)
    for component in components:
        absolute = _interval_abs_bounds(component)
        total = total + _interval_pow_scalar(absolute, p)
    if len(components) == 1 and len(box.lower) > 1:
        certified_lower = _scalar_output_certified_lower_power(model, box, p, input_lipschitz_weights)
        total = Interval.from_bounds(max(float(total.lower), certified_lower), float(total.upper))
    return total


def _activation_lipschitz_upper(layer) -> float:
    if isinstance(layer, (nn.ReLU, nn.Identity, nn.Flatten, nn.Softplus)):
        return 1.0
    if isinstance(layer, nn.LeakyReLU):
        return max(1.0, abs(float(layer.negative_slope)))
    if isinstance(layer, nn.Sigmoid):
        return 0.25
    if isinstance(layer, nn.Tanh):
        return 1.0
    if isinstance(layer, nn.Softmax):
        # Conservative elementwise upper bound in this context.
        return 1.0
    raise NotImplementedError


def _precompute_split_weights(model, domain: IntervalTensor) -> tuple[float, ...] | None:
    if len(domain.shape) != 1 or len(domain.lower) <= 1:
        return None

    layers = list(model) if isinstance(model, nn.Sequential) else [model]
    sensitivity = None

    try:
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                weight_abs = layer.weight.detach().cpu().to(dtype=torch.float64).abs()
                if sensitivity is None:
                    sensitivity = torch.ones(weight_abs.shape[0], dtype=torch.float64)
                sensitivity = weight_abs.t().matmul(sensitivity)
            else:
                slope_upper = _activation_lipschitz_upper(layer)
                if sensitivity is None:
                    continue
                sensitivity = sensitivity * slope_upper
    except NotImplementedError:
        return None

    if sensitivity is None:
        return None
    if sensitivity.numel() != len(domain.lower):
        return None
    return tuple(float(max(value.item(), 0.0)) for value in sensitivity)


def _split_box(box: IntervalTensor, split_weights: tuple[float, ...] | None = None) -> tuple[IntervalTensor, IntervalTensor]:
    widths = [upper - lower for lower, upper in zip(box.lower, box.upper)]
    if split_weights is not None and len(widths) > 1 and len(split_weights) == len(widths):
        scored_widths = [width * max(split_weights[idx], 0.0) for idx, width in enumerate(widths)]
        weighted_best = max(scored_widths)
        if weighted_best > 0.0:
            split_dim = max(range(len(scored_widths)), key=lambda idx: scored_widths[idx])
        else:
            split_dim = max(range(len(widths)), key=lambda idx: widths[idx])
    else:
        split_dim = max(range(len(widths)), key=lambda idx: widths[idx])
    midpoint = 0.5 * (box.lower[split_dim] + box.upper[split_dim])

    lower_left = list(box.lower)
    upper_left = list(box.upper)
    lower_right = list(box.lower)
    upper_right = list(box.upper)

    upper_left[split_dim] = midpoint
    lower_right[split_dim] = midpoint

    return (
        IntervalTensor.from_bounds(lower_left, upper_left),
        IntervalTensor.from_bounds(lower_right, upper_right),
    )


def _validate_dorfler_theta(theta: float) -> None:
    if not isfinite(theta) or theta <= 0.0 or theta > 1.0:
        raise ValueError("theta must be a finite real number in the interval (0, 1].")


def _dorfler_marking(indicators: list[float], theta: float) -> list[int]:
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


def _lpnorm_bounds(model, domain: IntervalTensor, p: float, iterations: int, theta: float) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.lpnorm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Lp integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    _validate_dorfler_theta(theta)
    effective_theta = theta
    boxes = [domain]
    split_weights = _precompute_split_weights(model, domain) if len(domain.lower) > 1 else None
    integrand_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], Interval] = {}
    volume_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}

    def _box_key(box: IntervalTensor) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            tuple(float(value) for value in box.lower),
            tuple(float(value) for value in box.upper),
        )

    def _cached_integrand(box: IntervalTensor) -> Interval:
        key = _box_key(box)
        cached = integrand_cache.get(key)
        if cached is None:
            cached = _lp_pointwise_power_bounds(model, box, p)
            integrand_cache[key] = cached
        return cached

    def _cached_volume(box: IntervalTensor) -> float:
        key = _box_key(box)
        cached = volume_cache.get(key)
        if cached is None:
            cached = _box_volume(box)
            volume_cache[key] = cached
        return cached

    for _ in range(iterations):
        indicators: list[float] = []
        for box in boxes:
            integrand_bounds = _cached_integrand(box)
            width = float(integrand_bounds.upper) - float(integrand_bounds.lower)
            indicators.append(width * _cached_volume(box))

        marked_indices = set(_dorfler_marking(indicators, effective_theta))
        refined_boxes: list[IntervalTensor] = []
        for idx, box in enumerate(boxes):
            if idx in marked_indices:
                left, right = _split_box(box, split_weights=split_weights)
                refined_boxes.extend([left, right])
            else:
                refined_boxes.append(box)
        boxes = refined_boxes

    integral = Interval.point(0.0)
    for box in boxes:
        integrand_bounds = _cached_integrand(box)
        box_volume = _cached_volume(box)
        weighted = Interval.from_bounds(
            float(integrand_bounds.lower) * box_volume,
            float(integrand_bounds.upper) * box_volume,
        )
        integral = integral + weighted

    non_negative = Interval.from_bounds(max(0.0, float(integral.lower)), max(0.0, float(integral.upper)))
    exponent = 1.0 / p
    return _interval_pow_scalar(non_negative, exponent)


def _identity_jacobian(size: int) -> list[list[Interval]]:
    rows: list[list[Interval]] = []
    for row_idx in range(size):
        row: list[Interval] = []
        for col_idx in range(size):
            row.append(Interval.point(1.0 if row_idx == col_idx else 0.0))
        rows.append(row)
    return rows


def _interval_derivative_bounds_relu(value: Interval) -> Interval:
    lower = float(value.lower)
    upper = float(value.upper)
    if upper <= 0.0:
        return Interval.point(0.0)
    if lower >= 0.0:
        return Interval.point(1.0)
    return Interval.from_bounds(0.0, 1.0)


def _interval_derivative_bounds_sigmoid(value: Interval) -> Interval:
    sigmoid_bounds = _apply_monotone_bounds(IntervalTensor((value.lower,), (value.upper,)), _sigmoid_scalar)
    sigma_lower = float(sigmoid_bounds.lower[0])
    sigma_upper = float(sigmoid_bounds.upper[0])
    candidate_values = [sigma_lower * (1.0 - sigma_lower), sigma_upper * (1.0 - sigma_upper)]
    maximum = max(candidate_values)
    if sigma_lower <= 0.5 <= sigma_upper:
        maximum = max(maximum, 0.25)
    minimum = min(candidate_values)
    return Interval.from_bounds(minimum, maximum)


def _interval_derivative_bounds_tanh(value: Interval) -> Interval:
    lower = float(value.lower)
    upper = float(value.upper)
    tanh_lower = tanh(lower)
    tanh_upper = tanh(upper)

    derivative_lower_endpoint = 1.0 - tanh_lower * tanh_lower
    derivative_upper_endpoint = 1.0 - tanh_upper * tanh_upper

    maximum = max(derivative_lower_endpoint, derivative_upper_endpoint)
    if lower <= 0.0 <= upper:
        maximum = 1.0
    minimum = min(derivative_lower_endpoint, derivative_upper_endpoint)
    lower_out = _pad_outward(minimum, -inf, include_float32=True)
    upper_out = _pad_outward(maximum, inf, include_float32=True)
    return Interval.from_bounds(lower_out, upper_out)


def _matrix_multiply(left: list[list[Interval]], right: list[list[Interval]]) -> list[list[Interval]]:
    if not left or not right:
        return []
    left_width = len(left[0])
    right_height = len(right)
    if left_width != right_height:
        raise ValueError("Jacobian dimensions are incompatible for multiplication.")

    right_width = len(right[0])
    output: list[list[Interval]] = []
    for row in left:
        output_row: list[Interval] = []
        for col_idx in range(right_width):
            accumulator = Interval.point(0.0)
            for shared_idx in range(left_width):
                accumulator = accumulator + row[shared_idx] * right[shared_idx][col_idx]
            output_row.append(accumulator)
        output.append(output_row)
    return output


def _jacobian_for_layer(layer, pre_activation: IntervalTensor) -> list[list[Interval]]:
    if isinstance(layer, nn.Linear):
        weight = layer.weight.detach().cpu()
        return [
            [_scalar_interval_from_weight(weight[row_idx, col_idx], Interval.point(1.0)) for col_idx in range(weight.shape[1])]
            for row_idx in range(weight.shape[0])
        ]
    if isinstance(layer, nn.ReLU):
        derivatives = [
            _interval_derivative_bounds_relu(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(len(pre_activation.lower))
        ]
        size = len(derivatives)
        return [[derivatives[row_idx] if row_idx == col_idx else Interval.point(0.0) for col_idx in range(size)] for row_idx in range(size)]
    if isinstance(layer, nn.Sigmoid):
        derivatives = [
            _interval_derivative_bounds_sigmoid(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(len(pre_activation.lower))
        ]
        size = len(derivatives)
        return [[derivatives[row_idx] if row_idx == col_idx else Interval.point(0.0) for col_idx in range(size)] for row_idx in range(size)]
    if isinstance(layer, nn.Tanh):
        derivatives = [
            _interval_derivative_bounds_tanh(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(len(pre_activation.lower))
        ]
        size = len(derivatives)
        return [[derivatives[row_idx] if row_idx == col_idx else Interval.point(0.0) for col_idx in range(size)] for row_idx in range(size)]
    if isinstance(layer, nn.Softmax):
        softmax_bounds = _softmax_forward(layer, pre_activation)
        size = len(softmax_bounds.lower)
        matrix: list[list[Interval]] = []
        for row_idx in range(size):
            row: list[Interval] = []
            s_i = Interval(softmax_bounds.lower[row_idx], softmax_bounds.upper[row_idx])
            for col_idx in range(size):
                s_j = Interval(softmax_bounds.lower[col_idx], softmax_bounds.upper[col_idx])
                if row_idx == col_idx:
                    row.append(s_i * (Interval.point(1.0) - s_j))
                else:
                    row.append(-(s_i * s_j))
            matrix.append(row)
        return matrix
    if isinstance(layer, nn.Flatten):
        if len(pre_activation.shape) != 1:
            raise NotImplementedError("Interval Jacobians currently support flat vectors only.")
        return _identity_jacobian(len(pre_activation.lower))
    raise NotImplementedError(
        f"Interval Jacobian currently supports nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax, and nn.Flatten; got {type(layer).__name__}."
    )


def _eval_jacobian_bounds(model, domain: IntervalTensor) -> IntervalTensor:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.eval_jacobian(domain) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Interval Jacobian evaluation currently supports flat input boxes only.")

    if isinstance(model, nn.Sequential):
        current_interval = domain
        current_jacobian = _identity_jacobian(len(domain.lower))
        for child in model:
            local_jacobian = _jacobian_for_layer(child, current_interval)
            current_jacobian = _matrix_multiply(local_jacobian, current_jacobian)
            current_interval = interval_forward(child, current_interval)
    else:
        local_jacobian = _jacobian_for_layer(model, domain)
        current_jacobian = local_jacobian

    lower = tuple(tuple(entry.lower for entry in row) for row in current_jacobian)
    upper = tuple(tuple(entry.upper for entry in row) for row in current_jacobian)
    return IntervalTensor.from_bounds(lower, upper)


def _sobolev_pointwise_power_bounds(
    model,
    box: IntervalTensor,
    p: float,
) -> Interval:
    output = model.eval(box)
    jacobian = model.eval_jacobian(box)
    if len(box.lower) <= 1:
        output = model.eval(box)
    else:
        output = _mean_value_output_bounds(model, box, jacobian=jacobian)
    total = Interval.point(0.0)

    for lower, upper in zip(output.lower, output.upper):
        component = Interval(lower, upper)
        total = total + _interval_pow_scalar(_interval_abs_bounds(component), p)

    for row_lower, row_upper in zip(jacobian.lower, jacobian.upper):
        for entry_lower, entry_upper in zip(row_lower, row_upper):
            derivative_component = Interval(entry_lower, entry_upper)
            total = total + _interval_pow_scalar(_interval_abs_bounds(derivative_component), p)

    if len(output.lower) == 1 and len(box.lower) > 1:
        certified_lower = _scalar_output_certified_lower_power(model, box, p, input_lipschitz_weights)
        total = Interval.from_bounds(max(float(total.lower), certified_lower), float(total.upper))

    return total


def _sobolev_norm_bounds(model, domain: IntervalTensor, p: float, iterations: int, theta: float) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.sobolev_norm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Sobolev integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    _validate_dorfler_theta(theta)
    effective_theta = theta
    boxes = [domain]
    split_weights = _precompute_split_weights(model, domain) if len(domain.lower) > 1 else None
    integrand_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], Interval] = {}
    volume_cache: dict[tuple[tuple[float, ...], tuple[float, ...]], float] = {}

    def _box_key(box: IntervalTensor) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            tuple(float(value) for value in box.lower),
            tuple(float(value) for value in box.upper),
        )

    def _cached_integrand(box: IntervalTensor) -> Interval:
        key = _box_key(box)
        cached = integrand_cache.get(key)
        if cached is None:
            cached = _sobolev_pointwise_power_bounds(model, box, p)
            integrand_cache[key] = cached
        return cached

    def _cached_volume(box: IntervalTensor) -> float:
        key = _box_key(box)
        cached = volume_cache.get(key)
        if cached is None:
            cached = _box_volume(box)
            volume_cache[key] = cached
        return cached

    for _ in range(iterations):
        indicators: list[float] = []
        for box in boxes:
            integrand_bounds = _cached_integrand(box)
            width = float(integrand_bounds.upper) - float(integrand_bounds.lower)
            indicators.append(width * _cached_volume(box))

        marked_indices = set(_dorfler_marking(indicators, effective_theta))
        refined_boxes: list[IntervalTensor] = []
        for idx, box in enumerate(boxes):
            if idx in marked_indices:
                left, right = _split_box(box, split_weights=split_weights)
                refined_boxes.extend([left, right])
            else:
                refined_boxes.append(box)
        boxes = refined_boxes

    integral = Interval.point(0.0)
    for box in boxes:
        integrand_bounds = _cached_integrand(box)
        box_volume = _cached_volume(box)
        weighted = Interval.from_bounds(
            float(integrand_bounds.lower) * box_volume,
            float(integrand_bounds.upper) * box_volume,
        )
        integral = integral + weighted

    non_negative = Interval.from_bounds(max(0.0, float(integral.lower)), max(0.0, float(integral.upper)))
    exponent = 1.0 / p
    return _interval_pow_scalar(non_negative, exponent)


def interval_forward(module, x: IntervalTensor) -> IntervalTensor:
    _require_torch()
    if isinstance(module, nn.Sequential):
        result = x
        for child in module:
            result = interval_forward(child, result)
        return result
    if isinstance(module, nn.Flatten):
        return IntervalTensor(tuple(x.lower), tuple(x.upper))
    if isinstance(module, nn.Linear):
        return _linear_forward(module, x)
    if isinstance(module, nn.ReLU):
        return _relu_forward(module, x)
    if isinstance(module, nn.Sigmoid):
        return _sigmoid_forward(module, x)
    if isinstance(module, nn.Tanh):
        return _tanh_forward(module, x)
    if isinstance(module, nn.Softplus):
        return _softplus_forward(module, x)
    if isinstance(module, nn.LeakyReLU):
        return _leaky_relu_forward(module, x)
    if isinstance(module, nn.Softmax):
        return _softmax_forward(module, x)
    if isinstance(module, nn.Identity):
        return IntervalTensor(tuple(x.lower), tuple(x.upper))
    if isinstance(module, IntervalAdd):
        left = interval_forward(module.left, x)
        right = interval_forward(module.right, x)
        return _interval_add(left, right)
    if isinstance(module, IntervalCat):
        parts = [interval_forward(branch, x) for branch in module.branches]
        return _interval_cat(parts, module.dim)
    raise NotImplementedError(
        f"Interval forward currently supports nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.LeakyReLU, nn.Softmax, nn.Identity, IntervalAdd, and IntervalCat only; got {type(module).__name__}."
    )


_ORIGINAL_EVAL = getattr(nn.Module, "eval", None) if nn is not None else None
_PATCHED = False


def enable_interval_eval() -> None:
    _require_torch()
    global _PATCHED
    if _PATCHED:
        return

    def eval_with_interval(self, interval: IntervalTensor | None = None):
        result = _ORIGINAL_EVAL(self)
        if interval is None:
            return result
        if not isinstance(interval, IntervalTensor):
            raise TypeError("model.eval(interval) requires an IntervalTensor input.")
        return interval_forward(self, interval)

    def lpnorm_with_interval(self, domain: IntervalTensor, p: float, iterations: int = 0, theta: float = 0.5):
        _ORIGINAL_EVAL(self)
        return _lpnorm_bounds(self, domain, p, iterations, theta)

    def eval_jacobian_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _eval_jacobian_bounds(self, domain)

    def sobolev_norm_with_interval(self, domain: IntervalTensor, p: float, iterations: int = 0, theta: float = 0.5):
        _ORIGINAL_EVAL(self)
        return _sobolev_norm_bounds(self, domain, p, iterations, theta)

    nn.Module.eval = eval_with_interval
    nn.Module.lpnorm = lpnorm_with_interval
    nn.Module.eval_jacobian = eval_jacobian_with_interval
    nn.Module.sobolev_norm = sobolev_norm_with_interval
    _PATCHED = True
