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


def _scalar_interval_from_weight(weight: Any, value: Interval, widen_float32: bool = True) -> Interval:
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
    lower = tuple(_pad_outward(func(bound), -inf, include_float32=True) for bound in x.lower)
    upper = tuple(_pad_outward(func(bound), inf, include_float32=True) for bound in x.upper)
    return IntervalTensor(lower, upper)


def _relu_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds(x, lambda value: max(0.0, value))


def _sigmoid_scalar(value: float) -> float:
    """Compute sigmoid(value) for scalar intervals.

    Sigmoid is strictly increasing on R, so interval propagation can evaluate
    the lower endpoint for the lower bound and the upper endpoint for the
    upper bound before outward rounding is applied.
    """
    return float(torch.sigmoid(torch.tensor(value, dtype=torch.float64)).item())


def _sigmoid_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds(x, _sigmoid_scalar)


def _tanh_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds(x, lambda value: float(torch.tanh(torch.tensor(value, dtype=torch.float64)).item()))


def _softplus_forward(layer, x: IntervalTensor) -> IntervalTensor:
    beta = float(layer.beta)
    threshold = float(layer.threshold)

    def _softplus_scalar(value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64)
        return float(torch.nn.functional.softplus(tensor, beta=beta, threshold=threshold).item())

    return _apply_monotone_bounds(x, _softplus_scalar)


def _leaky_relu_forward(layer, x: IntervalTensor) -> IntervalTensor:
    slope = float(layer.negative_slope)
    return _apply_monotone_bounds(x, lambda value: value if value >= 0.0 else slope * value)


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
    weight = layer.weight.detach().cpu()
    bias = layer.bias.detach().cpu() if layer.bias is not None else None
    x_lower = list(x.lower)
    x_upper = list(x.upper)
    input_intervals = [Interval(lb, ub) for lb, ub in zip(x_lower, x_upper)]

    outputs: list[Interval] = []
    for row_index, row in enumerate(weight):
        accumulator = Interval.point(0.0)
        for coefficient, input_interval in zip(row, input_intervals):
            accumulator = accumulator + _scalar_interval_from_weight(coefficient, input_interval)
        if bias is not None:
            accumulator = accumulator + _scalar_interval_from_weight(bias[row_index], Interval.point(1.0), widen_float32=True)
        outputs.append(accumulator)

    lower = tuple(item.lower for item in outputs)
    upper = tuple(item.upper for item in outputs)
    return IntervalTensor(lower, upper)


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


def _abs_power_bounds_from_scalar_bounds(lower: float, upper: float, p: float) -> Interval:
    absolute = _interval_abs_bounds(Interval(lower, upper))
    return _interval_pow_scalar(absolute, p)


def _lp_pointwise_power_bounds(model, box: IntervalTensor, p: float) -> Interval:
    output = model.eval(box)
    total = Interval.point(0.0)
    for lower, upper in zip(output.lower, output.upper):
        total = total + _abs_power_bounds_from_scalar_bounds(float(lower), float(upper), p)
    return total


def _split_box_along_dimension(box: IntervalTensor, split_dim: int) -> tuple[IntervalTensor, IntervalTensor]:
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


def _split_box(box: IntervalTensor) -> tuple[IntervalTensor, IntervalTensor]:
    widths = [upper - lower for lower, upper in zip(box.lower, box.upper)]
    split_dim = max(range(len(widths)), key=lambda idx: widths[idx])
    return _split_box_along_dimension(box, split_dim)


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




@dataclass(frozen=True)
class _AdaptiveBoxData:
    box: IntervalTensor
    volume: float
    integrand_bounds: Interval
    indicator: float


def _build_box_data(
    box: IntervalTensor,
    integrand_evaluator,
    bounds_cache: dict[IntervalTensor, Interval],
) -> _AdaptiveBoxData:
    volume = _box_volume(box)
    integrand_bounds = bounds_cache.get(box)
    if integrand_bounds is None:
        integrand_bounds = integrand_evaluator(box)
        bounds_cache[box] = integrand_bounds
    indicator = (float(integrand_bounds.upper) - float(integrand_bounds.lower)) * volume
    return _AdaptiveBoxData(
        box=box,
        volume=volume,
        integrand_bounds=integrand_bounds,
        indicator=indicator,
    )


def _build_box_data_batch(
    boxes: list[IntervalTensor],
    integrand_evaluator,
    bounds_cache: dict[IntervalTensor, Interval],
    batch_size: int,
) -> list[_AdaptiveBoxData]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    data: list[_AdaptiveBoxData] = []
    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start : start + batch_size]
        data.extend(_build_box_data(box, integrand_evaluator, bounds_cache) for box in chunk)
    return data


def _adaptive_integral_bounds(
    domain: IntervalTensor,
    iterations: int,
    theta: float,
    integrand_evaluator,
    split_topk: int,
    batch_size: int,
) -> Interval:
    del split_topk
    bounds_cache: dict[IntervalTensor, Interval] = {}
    boxes = _build_box_data_batch([domain], integrand_evaluator, bounds_cache, batch_size)
    for _ in range(iterations):
        indicators = [data.indicator for data in boxes]

        marked_indices = set(_dorfler_marking(indicators, theta))
        refined_boxes: list[_AdaptiveBoxData] = []
        for idx, data in enumerate(boxes):
            if idx in marked_indices:
                left, right = _split_box(data.box)
                left_data, right_data = _build_box_data_batch([left, right], integrand_evaluator, bounds_cache, batch_size)
                refined_boxes.append(left_data)
                refined_boxes.append(right_data)
            else:
                refined_boxes.append(data)
        boxes = refined_boxes

    integral = Interval.point(0.0)
    for data in boxes:
        weighted = Interval.from_bounds(
            float(data.integrand_bounds.lower) * data.volume,
            float(data.integrand_bounds.upper) * data.volume,
        )
        integral = integral + weighted
    return integral

def _lpnorm_bounds(
    model,
    domain: IntervalTensor,
    p: float,
    iterations: int,
    theta: float,
    split_topk: int,
    batch_size: int,
) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.lpnorm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Lp integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    _validate_dorfler_theta(theta)
    if split_topk <= 0:
        raise ValueError("split_topk must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    integral = _adaptive_integral_bounds(
        domain,
        iterations,
        theta,
        lambda box: _lp_pointwise_power_bounds(model, box, p),
        split_topk,
        batch_size,
    )

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


def _eval_jacobian_bounds_batched(model, boxes: list[IntervalTensor], batch_size: int) -> list[IntervalTensor]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    jacobians: list[IntervalTensor] = []
    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start : start + batch_size]
        jacobians.extend(_eval_jacobian_bounds(model, box) for box in chunk)
    return jacobians


def _sobolev_pointwise_power_bounds(model, box: IntervalTensor, p: float) -> Interval:
    output = model.eval(box)
    jacobian = model.eval_jacobian(box)
    total = Interval.point(0.0)

    for lower, upper in zip(output.lower, output.upper):
        total = total + _abs_power_bounds_from_scalar_bounds(float(lower), float(upper), p)

    for row_lower, row_upper in zip(jacobian.lower, jacobian.upper):
        for entry_lower, entry_upper in zip(row_lower, row_upper):
            total = total + _abs_power_bounds_from_scalar_bounds(float(entry_lower), float(entry_upper), p)

    return total


def _sobolev_pointwise_power_bounds_batched(model, boxes: list[IntervalTensor], p: float, batch_size: int) -> list[Interval]:
    outputs = [model.eval(box) for box in boxes]
    jacobians = _eval_jacobian_bounds_batched(model, boxes, batch_size)
    results: list[Interval] = []
    for output, jacobian in zip(outputs, jacobians):
        total = Interval.point(0.0)
        for lower, upper in zip(output.lower, output.upper):
            total = total + _abs_power_bounds_from_scalar_bounds(float(lower), float(upper), p)
        for row_lower, row_upper in zip(jacobian.lower, jacobian.upper):
            for entry_lower, entry_upper in zip(row_lower, row_upper):
                total = total + _abs_power_bounds_from_scalar_bounds(float(entry_lower), float(entry_upper), p)
        results.append(total)
    return results


def _sobolev_norm_bounds(
    model,
    domain: IntervalTensor,
    p: float,
    iterations: int,
    theta: float,
    split_topk: int,
    batch_size: int,
) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.sobolev_norm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Sobolev integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    _validate_dorfler_theta(theta)
    if split_topk <= 0:
        raise ValueError("split_topk must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    integral = _adaptive_integral_bounds(
        domain,
        iterations,
        theta,
        lambda box: _sobolev_pointwise_power_bounds(model, box, p),
        split_topk,
        batch_size,
    )

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

    def lpnorm_with_interval(
        self,
        domain: IntervalTensor,
        p: float,
        iterations: int = 0,
        theta: float = 0.5,
        split_topk: int = 2,
        batch_size: int = 64,
    ):
        _ORIGINAL_EVAL(self)
        return _lpnorm_bounds(self, domain, p, iterations, theta, split_topk, batch_size)

    def eval_jacobian_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _eval_jacobian_bounds(self, domain)

    def sobolev_norm_with_interval(
        self,
        domain: IntervalTensor,
        p: float,
        iterations: int = 0,
        theta: float = 0.5,
        split_topk: int = 2,
        batch_size: int = 64,
    ):
        _ORIGINAL_EVAL(self)
        return _sobolev_norm_bounds(self, domain, p, iterations, theta, split_topk, batch_size)

    nn.Module.eval = eval_with_interval
    nn.Module.lpnorm = lpnorm_with_interval
    nn.Module.eval_jacobian = eval_jacobian_with_interval
    nn.Module.sobolev_norm = sobolev_norm_with_interval
    _PATCHED = True
