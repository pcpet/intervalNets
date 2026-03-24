from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter
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


def _scalar_interval_from_weight(weight: float, value: Interval) -> Interval:
    return Interval.point(weight) * value


def _apply_monotone_bounds(x: IntervalTensor, func) -> IntervalTensor:
    lower = tuple(nextafter(func(bound), -inf) for bound in x.lower)
    upper = tuple(nextafter(func(bound), inf) for bound in x.upper)
    return IntervalTensor(lower, upper)


def _relu_forward(layer, x: IntervalTensor) -> IntervalTensor:
    return _apply_monotone_bounds(x, lambda value: max(0.0, value))


def _softmax_component_bounds(index: int, lower: tuple[float, ...], upper: tuple[float, ...]) -> tuple[float, float]:
    lower_num = float(torch.exp(torch.tensor(lower[index], dtype=torch.float64)).item())
    lower_den = lower_num + sum(
        float(torch.exp(torch.tensor(upper[j], dtype=torch.float64)).item())
        for j in range(len(lower))
        if j != index
    )

    upper_num = float(torch.exp(torch.tensor(upper[index], dtype=torch.float64)).item())
    upper_den = upper_num + sum(
        float(torch.exp(torch.tensor(lower[j], dtype=torch.float64)).item())
        for j in range(len(lower))
        if j != index
    )

    return nextafter(lower_num / lower_den, -inf), nextafter(upper_num / upper_den, inf)


def _softmax_forward(layer, x: IntervalTensor) -> IntervalTensor:
    if len(x.shape) != 1:
        raise NotImplementedError("Interval Softmax currently supports 1D vectors only.")

    dim = layer.dim
    n = len(x.lower)
    if dim not in (-1, 0):
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
    weight = layer.weight.detach().cpu().tolist()
    bias = layer.bias.detach().cpu().tolist() if layer.bias is not None else None
    x_lower = list(x.lower)
    x_upper = list(x.upper)
    input_intervals = [Interval(lb, ub) for lb, ub in zip(x_lower, x_upper)]

    outputs: list[Interval] = []
    for row_index, row in enumerate(weight):
        accumulator = Interval.point(0.0)
        for coefficient, input_interval in zip(row, input_intervals):
            accumulator = accumulator + _scalar_interval_from_weight(float(coefficient), input_interval)
        if bias is not None:
            accumulator = accumulator + Interval.point(float(bias[row_index]))
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
    if isinstance(module, nn.Softmax):
        return _softmax_forward(module, x)
    raise NotImplementedError(
        f"Interval forward currently supports nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, and nn.Softmax only; got {type(module).__name__}."
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
