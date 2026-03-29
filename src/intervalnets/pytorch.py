from __future__ import annotations

from math import exp, inf, isfinite, log, nextafter, tanh
from typing import Any

from .interval import Interval

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - environment dependent
    torch = None
    nn = None


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
    lower_out: list[float] = []
    upper_out: list[float] = []
    for lower, upper in zip(x.lower, x.upper):
        lo = float(lower)
        hi = float(upper)
        if hi <= 0.0:
            # ReLU([l, u]) is exactly [0, 0] on non-positive inputs.
            lower_out.append(0.0)
            upper_out.append(0.0)
            continue
        if lo >= 0.0:
            lower_out.append(_pad_outward(lo, -inf, include_float32=True))
            upper_out.append(_pad_outward(hi, inf, include_float32=True))
            continue
        # Crossing zero: lower bound is exactly 0, upper needs outward padding.
        lower_out.append(0.0)
        upper_out.append(_pad_outward(max(0.0, hi), inf, include_float32=True))
    return IntervalTensor.from_bounds(tuple(lower_out), tuple(upper_out))


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
    weight = layer.weight.detach().cpu().to(torch.float64)
    bias = layer.bias.detach().cpu().to(torch.float64) if layer.bias is not None else None
    x_mid = torch.tensor(x.midpoint, dtype=torch.float64)
    x_rad = torch.tensor(x.radius, dtype=torch.float64)

    # Inflate the incoming radius so [x_mid - x_rad, x_mid + x_rad] is a
    # directed-rounded enclosure before the linear map.
    x_upper = torch.nextafter(x_mid + x_rad, torch.full_like(x_mid, float("inf")))
    x_lower = torch.nextafter(x_mid - x_rad, torch.full_like(x_mid, float("-inf")))
    x_rad_enclosed = torch.maximum(x_upper - x_mid, x_mid - x_lower)

    output_mid_tensor = weight.matmul(x_mid)
    if bias is not None:
        output_mid_tensor = output_mid_tensor + bias
    output_rad_tensor = weight.abs().matmul(x_rad_enclosed)

    lower_tensor = torch.nextafter(output_mid_tensor - output_rad_tensor, torch.full_like(output_mid_tensor, float("-inf")))
    upper_tensor = torch.nextafter(output_mid_tensor + output_rad_tensor, torch.full_like(output_mid_tensor, float("inf")))
    return IntervalTensor.from_bounds(tuple(float(value) for value in lower_tensor.tolist()), tuple(float(value) for value in upper_tensor.tolist()))


def _concretize_affine_bounds(
    lower_matrix: torch.Tensor,
    lower_bias: torch.Tensor,
    upper_matrix: torch.Tensor,
    upper_bias: torch.Tensor,
    input_box: IntervalTensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_mid = torch.tensor(input_box.midpoint, dtype=torch.float64)
    x_rad = torch.tensor(input_box.radius, dtype=torch.float64)

    lower_center = lower_matrix.matmul(x_mid) + lower_bias
    lower_radius = lower_matrix.abs().matmul(x_rad)
    upper_center = upper_matrix.matmul(x_mid) + upper_bias
    upper_radius = upper_matrix.abs().matmul(x_rad)

    lower = torch.nextafter(lower_center - lower_radius, torch.full_like(lower_center, float("-inf")))
    upper = torch.nextafter(upper_center + upper_radius, torch.full_like(upper_center, float("inf")))
    return lower, upper


def _linear_relaxation_step(
    weight: torch.Tensor,
    bias: torch.Tensor,
    lower_matrix: torch.Tensor,
    lower_bias: torch.Tensor,
    upper_matrix: torch.Tensor,
    upper_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    positive = (weight >= 0).to(torch.float64)
    negative = (weight < 0).to(torch.float64)

    next_lower_matrix = (positive * weight).matmul(lower_matrix) + (negative * weight).matmul(upper_matrix)
    next_upper_matrix = (positive * weight).matmul(upper_matrix) + (negative * weight).matmul(lower_matrix)

    next_lower_bias = (positive * weight).matmul(lower_bias) + (negative * weight).matmul(upper_bias) + bias
    next_upper_bias = (positive * weight).matmul(upper_bias) + (negative * weight).matmul(lower_bias) + bias
    return next_lower_matrix, next_lower_bias, next_upper_matrix, next_upper_bias


def _relu_relaxation_step(
    lower_matrix: torch.Tensor,
    lower_bias: torch.Tensor,
    upper_matrix: torch.Tensor,
    upper_bias: torch.Tensor,
    input_box: IntervalTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_lower, pre_upper = _concretize_affine_bounds(lower_matrix, lower_bias, upper_matrix, upper_bias, input_box)
    size = pre_lower.numel()
    next_lower_matrix = torch.zeros_like(lower_matrix)
    next_lower_bias = torch.zeros_like(lower_bias)
    next_upper_matrix = torch.zeros_like(upper_matrix)
    next_upper_bias = torch.zeros_like(upper_bias)

    for idx in range(size):
        lower = float(pre_lower[idx].item())
        upper = float(pre_upper[idx].item())

        if upper <= 0.0:
            continue

        if lower >= 0.0:
            next_lower_matrix[idx] = lower_matrix[idx]
            next_lower_bias[idx] = lower_bias[idx]
            next_upper_matrix[idx] = upper_matrix[idx]
            next_upper_bias[idx] = upper_bias[idx]
            continue

        slope = upper / (upper - lower)
        slope = min(1.0, max(0.0, slope))
        intercept = -slope * lower
        intercept = _pad_outward(intercept, inf, include_float32=True)

        next_upper_matrix[idx] = slope * upper_matrix[idx]
        next_upper_bias[idx] = slope * upper_bias[idx] + intercept
        # Crossing neurons use the constant lower bound 0.0, which is a
        # globally valid lower enclosure for ReLU.

    return next_lower_matrix, next_lower_bias, next_upper_matrix, next_upper_bias


def _sequential_linear_relu_relaxation(module: nn.Sequential, x: IntervalTensor) -> IntervalTensor:
    if len(x.shape) != 1:
        raise NotImplementedError("Slope-aware enclosure currently supports flat vectors only.")

    input_dim = len(x.lower)
    lower_matrix = torch.eye(input_dim, dtype=torch.float64)
    upper_matrix = torch.eye(input_dim, dtype=torch.float64)
    lower_bias = torch.zeros(input_dim, dtype=torch.float64)
    upper_bias = torch.zeros(input_dim, dtype=torch.float64)

    children = list(module.children())
    for child_idx, child in enumerate(children):
        if isinstance(child, nn.Linear):
            weight = child.weight.detach().cpu().to(torch.float64)
            bias = child.bias.detach().cpu().to(torch.float64) if child.bias is not None else torch.zeros(weight.shape[0], dtype=torch.float64)
            lower_matrix, lower_bias, upper_matrix, upper_bias = _linear_relaxation_step(
                weight,
                bias,
                lower_matrix,
                lower_bias,
                upper_matrix,
                upper_bias,
            )
            continue

        if isinstance(child, nn.ReLU):
            lower_matrix, lower_bias, upper_matrix, upper_bias = _relu_relaxation_step(
                lower_matrix,
                lower_bias,
                upper_matrix,
                upper_bias,
                x,
            )
            continue

        lower, upper = _concretize_affine_bounds(lower_matrix, lower_bias, upper_matrix, upper_bias, x)
        concretized = IntervalTensor.from_bounds(
            tuple(float(value) for value in lower.tolist()),
            tuple(float(value) for value in upper.tolist()),
        )
        remainder = nn.Sequential(*children[child_idx:])
        return interval_forward(remainder, concretized, enclosure_mode="box")

    lower, upper = _concretize_affine_bounds(lower_matrix, lower_bias, upper_matrix, upper_bias, x)
    return IntervalTensor.from_bounds(
        tuple(float(value) for value in lower.tolist()),
        tuple(float(value) for value in upper.tolist()),
    )


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


def _lp_pointwise_power_bounds(model, box: IntervalTensor, p: float) -> Interval:
    output = model.eval(box)
    components = [Interval(lb, ub) for lb, ub in zip(output.lower, output.upper)]
    total = Interval.point(0.0)
    for component in components:
        absolute = _interval_abs_bounds(component)
        total = total + _interval_pow_scalar(absolute, p)
    return total


def _jacobian_dimension_scores(jacobian: IntervalTensor) -> list[float]:
    if len(jacobian.shape) != 2:
        raise ValueError("Jacobian interval must be matrix-shaped.")
    output_dim = len(jacobian.lower)
    input_dim = len(jacobian.lower[0]) if output_dim > 0 else 0
    scores = [0.0] * input_dim
    for row_idx in range(output_dim):
        for col_idx in range(input_dim):
            lower = float(jacobian.lower[row_idx][col_idx])
            upper = float(jacobian.upper[row_idx][col_idx])
            scores[col_idx] += max(abs(lower), abs(upper))
    return scores


def _choose_split_dim(box: IntervalTensor, jacobian: IntervalTensor | None = None) -> int:
    widths = [float(upper - lower) for lower, upper in zip(box.lower, box.upper)]
    if jacobian is None or len(widths) <= 1:
        return max(range(len(widths)), key=lambda idx: widths[idx])
    scores = _jacobian_dimension_scores(jacobian)
    weighted = [width * score for width, score in zip(widths, scores)]
    if all(score <= 0.0 for score in weighted):
        return max(range(len(widths)), key=lambda idx: widths[idx])
    return max(range(len(weighted)), key=lambda idx: weighted[idx])


def _split_box(box: IntervalTensor, split_dim: int | None = None) -> tuple[IntervalTensor, IntervalTensor]:
    if split_dim is None:
        split_dim = _choose_split_dim(box, jacobian=None)
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

    boxes = [domain]
    use_jacobian_splitting = len(domain.lower) > 1
    for _ in range(iterations):
        indicators: list[float] = []
        split_dims: list[int] = []
        for box in boxes:
            affine_on_box = False
            try:
                affine_on_box = _is_affine_on_box(model, box)
            except (NotImplementedError, TypeError):
                affine_on_box = False

            integrand_bounds = _lp_pointwise_power_bounds(model, box, p)
            if affine_on_box:
                # On a certified affine ReLU piece we currently keep the box unsplit.
                # This preserves validity while prioritizing speed until an exact affine-piece
                # integral path is introduced.
                indicators.append(0.0)
            else:
                width = float(integrand_bounds.upper) - float(integrand_bounds.lower)
                indicators.append(width * _box_volume(box))

            if use_jacobian_splitting and not affine_on_box:
                jacobian = model.eval_jacobian(box)
                split_dims.append(_choose_split_dim(box, jacobian))
            else:
                split_dims.append(_choose_split_dim(box, None))

        if all(indicator <= 0.0 for indicator in indicators):
            break
        marked_indices = set(_dorfler_marking(indicators, theta))
        refined_boxes: list[IntervalTensor] = []
        for idx, box in enumerate(boxes):
            if idx in marked_indices:
                left, right = _split_box(box, split_dim=split_dims[idx])
                refined_boxes.extend([left, right])
            else:
                refined_boxes.append(box)
        boxes = refined_boxes

    integral = Interval.point(0.0)
    for box in boxes:
        integrand_bounds = _lp_pointwise_power_bounds(model, box, p)
        weighted = Interval.from_bounds(
            float(integrand_bounds.lower) * _box_volume(box),
            float(integrand_bounds.upper) * _box_volume(box),
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

    left_mid = torch.tensor([[float(entry.midpoint) for entry in row] for row in left], dtype=torch.float64)
    left_rad = torch.tensor([[float(entry.radius) for entry in row] for row in left], dtype=torch.float64)
    right_mid = torch.tensor([[float(entry.midpoint) for entry in row] for row in right], dtype=torch.float64)
    right_rad = torch.tensor([[float(entry.radius) for entry in row] for row in right], dtype=torch.float64)

    output_mid = left_mid.matmul(right_mid)
    output_rad = (
        left_mid.abs().matmul(right_rad)
        + left_rad.matmul(right_mid.abs())
        + left_rad.matmul(right_rad)
    )

    lower = torch.nextafter(output_mid - output_rad, torch.full_like(output_mid, float("-inf")))
    upper = torch.nextafter(output_mid + output_rad, torch.full_like(output_mid, float("inf")))
    lower_rows = lower.tolist()
    upper_rows = upper.tolist()
    return [
        [Interval.from_bounds(lower_rows[row_idx][col_idx], upper_rows[row_idx][col_idx]) for col_idx in range(len(lower_rows[row_idx]))]
        for row_idx in range(len(lower_rows))
    ]


def _jacobian_for_layer(layer, pre_activation: IntervalTensor) -> list[list[Interval]]:
    if isinstance(layer, nn.Linear):
        weight = layer.weight.detach().cpu()
        matrix = weight.to(torch.float64).tolist()
        return [[Interval.point(value) for value in row] for row in matrix]
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
    output: IntervalTensor | None = None,
    jacobian: IntervalTensor | None = None,
) -> Interval:
    output = output if output is not None else model.eval(box)
    jacobian = jacobian if jacobian is not None else model.eval_jacobian(box)
    total = Interval.point(0.0)

    for lower, upper in zip(output.lower, output.upper):
        component = Interval(lower, upper)
        total = total + _interval_pow_scalar(_interval_abs_bounds(component), p)

    for row_lower, row_upper in zip(jacobian.lower, jacobian.upper):
        for entry_lower, entry_upper in zip(row_lower, row_upper):
            derivative_component = Interval(entry_lower, entry_upper)
            total = total + _interval_pow_scalar(_interval_abs_bounds(derivative_component), p)

    return total


def _interval_tensor_is_exact_constant(interval: IntervalTensor) -> bool:
    def _all_equal(lower, upper) -> bool:
        if isinstance(lower, tuple) and isinstance(upper, tuple):
            return len(lower) == len(upper) and all(_all_equal(lo, hi) for lo, hi in zip(lower, upper))
        if isinstance(lower, tuple) or isinstance(upper, tuple):
            return False
        return float(lower) == float(upper)

    return _all_equal(interval.lower, interval.upper)


def _jacobian_is_exact_zero(jacobian: IntervalTensor) -> bool:
    return all(
        float(entry_lower) == 0.0 and float(entry_upper) == 0.0
        for row_lower, row_upper in zip(jacobian.lower, jacobian.upper)
        for entry_lower, entry_upper in zip(row_lower, row_upper)
    )


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

    boxes = [domain]
    use_jacobian_splitting = len(domain.lower) > 1
    for _ in range(iterations):
        indicators: list[float] = []
        split_dims: list[int] = []
        for box in boxes:
            affine_on_box = False
            try:
                affine_on_box = _is_affine_on_box(model, box)
            except (NotImplementedError, TypeError):
                affine_on_box = False

            output = model.eval(box)
            jacobian = model.eval_jacobian(box)
            integrand_bounds = _sobolev_pointwise_power_bounds(model, box, p, output=output, jacobian=jacobian)
            if affine_on_box:
                # On a certified affine ReLU piece we currently keep the box unsplit.
                # This preserves validity while prioritizing speed until an exact affine-piece
                # integral path is introduced.
                indicators.append(0.0)
            elif _interval_tensor_is_exact_constant(output) and _jacobian_is_exact_zero(jacobian):
                # A rigorously constant box has zero Sobolev seminorm contribution,
                # so further refinement is unnecessary for the derivative part.
                indicators.append(0.0)
            else:
                width = float(integrand_bounds.upper) - float(integrand_bounds.lower)
                indicators.append(width * _box_volume(box))
            if use_jacobian_splitting:
                split_dims.append(_choose_split_dim(box, jacobian))
            else:
                split_dims.append(_choose_split_dim(box, None))

        if all(indicator <= 0.0 for indicator in indicators):
            break
        marked_indices = set(_dorfler_marking(indicators, theta))
        refined_boxes: list[IntervalTensor] = []
        for idx, box in enumerate(boxes):
            if idx in marked_indices:
                left, right = _split_box(box, split_dim=split_dims[idx])
                refined_boxes.extend([left, right])
            else:
                refined_boxes.append(box)
        boxes = refined_boxes

    integral = Interval.point(0.0)
    for box in boxes:
        integrand_bounds = _sobolev_pointwise_power_bounds(model, box, p)
        weighted = Interval.from_bounds(
            float(integrand_bounds.lower) * _box_volume(box),
            float(integrand_bounds.upper) * _box_volume(box),
        )
        integral = integral + weighted

    non_negative = Interval.from_bounds(max(0.0, float(integral.lower)), max(0.0, float(integral.upper)))
    exponent = 1.0 / p
    return _interval_pow_scalar(non_negative, exponent)


def _is_affine_on_box(module, domain: IntervalTensor) -> bool:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.is_affine_on(domain) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Affine-piece checks currently support flat input boxes only.")

    if not isinstance(module, nn.Sequential):
        raise NotImplementedError("Affine-piece checks currently support nn.Sequential models only.")

    current = domain
    for child in module:
        if isinstance(child, nn.ReLU):
            for lower, upper in zip(current.lower, current.upper):
                if float(lower) < 0.0 < float(upper):
                    return False
            current = interval_forward(child, current, enclosure_mode="box")
            continue
        current = interval_forward(child, current, enclosure_mode="box")
    return True


def interval_forward(module, x: IntervalTensor, enclosure_mode: str = "box") -> IntervalTensor:
    _require_torch()
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    if isinstance(module, nn.Sequential):
        if enclosure_mode == "slope":
            return _sequential_linear_relu_relaxation(module, x)
        result = x
        for child in module:
            result = interval_forward(child, result, enclosure_mode=enclosure_mode)
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
        left = interval_forward(module.left, x, enclosure_mode=enclosure_mode)
        right = interval_forward(module.right, x, enclosure_mode=enclosure_mode)
        return _interval_add(left, right)
    if isinstance(module, IntervalCat):
        parts = [interval_forward(branch, x, enclosure_mode=enclosure_mode) for branch in module.branches]
        return _interval_cat(parts, module.dim)
    raise NotImplementedError(
        f"Interval forward currently supports nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.LeakyReLU, nn.Softmax, nn.Identity, IntervalAdd, and IntervalCat only; got {type(module).__name__}."
    )


_ORIGINAL_EVAL = getattr(nn.Module, "eval", None) if nn is not None else None
_PATCHED = False


def enable_interval_eval(enclosure_mode: str = "box") -> None:
    _require_torch()
    global _PATCHED
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    if _PATCHED:
        return

    def eval_with_interval(self, interval: IntervalTensor | None = None):
        result = _ORIGINAL_EVAL(self)
        if interval is None:
            return result
        if not isinstance(interval, IntervalTensor):
            raise TypeError("model.eval(interval) requires an IntervalTensor input.")
        return interval_forward(self, interval, enclosure_mode=enclosure_mode)

    def lpnorm_with_interval(self, domain: IntervalTensor, p: float, iterations: int = 0, theta: float = 0.5):
        _ORIGINAL_EVAL(self)
        return _lpnorm_bounds(self, domain, p, iterations, theta)

    def eval_jacobian_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _eval_jacobian_bounds(self, domain)

    def sobolev_norm_with_interval(self, domain: IntervalTensor, p: float, iterations: int = 0, theta: float = 0.5):
        _ORIGINAL_EVAL(self)
        return _sobolev_norm_bounds(self, domain, p, iterations, theta)

    def is_affine_on_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _is_affine_on_box(self, domain)

    nn.Module.eval = eval_with_interval
    nn.Module.lpnorm = lpnorm_with_interval
    nn.Module.eval_jacobian = eval_jacobian_with_interval
    nn.Module.sobolev_norm = sobolev_norm_with_interval
    nn.Module.is_affine_on = is_affine_on_with_interval
    _PATCHED = True
