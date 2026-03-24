from __future__ import annotations

from dataclasses import dataclass
from math import exp, inf, log, nextafter
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


def _scalar_interval_from_weight(weight: Any, value: Interval) -> Interval:
    if torch is not None and isinstance(weight, torch.Tensor):
        scalar = weight.detach().cpu()
        if scalar.numel() != 1:
            raise ValueError("Expected a scalar weight tensor.")
        coefficient = float(scalar.item())
        if scalar.dtype in {torch.float16, torch.bfloat16, torch.float32}:
            negative_inf = torch.tensor(float("-inf"), dtype=scalar.dtype)
            positive_inf = torch.tensor(float("inf"), dtype=scalar.dtype)
            lower = float(torch.nextafter(scalar, negative_inf).item())
            upper = float(torch.nextafter(scalar, positive_inf).item())
            return Interval.from_bounds(lower, upper) * value
        return Interval.point(coefficient) * value
    return Interval.point(weight) * value


def _apply_monotone_bounds(x: IntervalTensor, func) -> IntervalTensor:
    # For monotone activations f, interval images satisfy
    # f([l, u]) = [f(l), f(u)].
    # Therefore, evaluating only endpoints is sound and complete.
    lower = tuple(nextafter(func(bound), -inf) for bound in x.lower)
    upper = tuple(nextafter(func(bound), inf) for bound in x.upper)
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
    return nextafter(lower_value, -inf), nextafter(upper_value, inf)


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
            accumulator = accumulator + _scalar_interval_from_weight(bias[row_index], Interval.point(1.0))
        outputs.append(accumulator)

    lower = tuple(item.lower for item in outputs)
    upper = tuple(item.upper for item in outputs)
    return IntervalTensor(lower, upper)


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
    if isinstance(module, nn.Softmax):
        return _softmax_forward(module, x)
    raise NotImplementedError(
        f"Interval forward currently supports nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, nn.Sigmoid, and nn.Softmax only; got {type(module).__name__}."
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

    nn.Module.eval = eval_with_interval
    _PATCHED = True
