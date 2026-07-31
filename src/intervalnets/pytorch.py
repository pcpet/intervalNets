from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, inf, isfinite, log, nextafter, tanh
from typing import Any

from .interval import Interval
from .polynomial_zonotope import PZTwoJet, PolynomialZonotope
from .pz_tanh import (
    affine_tanh_double_prime_enclosure,
    affine_tanh_enclosure,
    affine_tanh_prime_enclosure,
)
from .pz_integration import PZIntegrationCell, pz_l2norm_bounds, pz_sobolev_norm_bounds
from .pz_norms import pz_twojet_l2_norm, pz_twojet_w12_norm, pz_twojet_w22_norm


@dataclass(frozen=True)
class PZTwoJetTraceRecord:
    """One opt-in trace snapshot from polynomial-zonotope two-jet propagation."""

    layer_index: int
    layer_name: str
    layer_type: str
    jet: PZTwoJet
    summary: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PZTwoJetTraceResult:
    """Final two-jet plus per-layer trace snapshots."""

    final: PZTwoJet
    records: list[PZTwoJetTraceRecord]


@dataclass(frozen=True)
class PZValueTraceRecord:
    """One opt-in trace snapshot from value-only PZ propagation."""

    layer_index: int
    layer_name: str
    layer_type: str
    value: PolynomialZonotope
    summary: dict[str, Any]


@dataclass(frozen=True)
class PZValueTraceResult:
    """Final value enclosure plus per-layer trace snapshots."""

    final: PolynomialZonotope
    records: list[PZValueTraceRecord]


def _pz_summary(zonotope: PolynomialZonotope) -> dict[str, Any]:
    return {
        "shape": zonotope.shape,
        "num_noise": zonotope.num_noise,
        "noise_kinds": zonotope.noise_kinds,
        "term_count": len(zonotope.terms),
        "max_degree": max((sum(exp) for exp in zonotope.terms), default=0),
    }


def _pz_twojet_trace_record(layer_index: int, layer_name: str, layer_type: str, jet: PZTwoJet) -> PZTwoJetTraceRecord:
    return PZTwoJetTraceRecord(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_type=layer_type,
        jet=jet,
        summary={"Y": _pz_summary(jet.Y), "J": _pz_summary(jet.J), "H": _pz_summary(jet.H)},
    )


def _pz_value_trace_record(
    layer_index: int,
    layer_name: str,
    layer_type: str,
    value: PolynomialZonotope,
) -> PZValueTraceRecord:
    return PZValueTraceRecord(
        layer_index=layer_index,
        layer_name=layer_name,
        layer_type=layer_type,
        value=value,
        summary=_pz_summary(value),
    )

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


def _pz_twojet_linear_forward(layer: nn.Linear, jet: PZTwoJet) -> PZTwoJet:
    """Propagate a polynomial-zonotope two-jet through ``nn.Linear`` exactly."""

    _require_torch()
    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None
    is_torch_backend = torch is not None and isinstance(jet.Y.center, torch.Tensor)
    if is_torch_backend:
        weight = weight.to(dtype=jet.Y.center.dtype, device=jet.Y.center.device)
        if bias is not None:
            bias = bias.to(dtype=jet.Y.center.dtype, device=jet.Y.center.device)
    else:
        weight = weight.cpu().tolist()
        bias = bias.cpu().tolist() if bias is not None else None

    return PZTwoJet(
        Y=jet.Y.linear_map(weight, bias),
        J=jet.J.linear_map(weight, bias=None),
        H=jet.H.linear_map(weight, bias=None),
    )


def _pz_value_linear_forward(
    layer: nn.Linear,
    value: PolynomialZonotope,
) -> PolynomialZonotope:
    """Propagate a value-only polynomial zonotope through ``nn.Linear``."""

    _require_torch()
    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None
    if isinstance(value.center, torch.Tensor):
        weight = weight.to(dtype=value.center.dtype, device=value.center.device)
        if bias is not None:
            bias = bias.to(dtype=value.center.dtype, device=value.center.device)
    else:
        weight = weight.cpu().tolist()
        bias = bias.cpu().tolist() if bias is not None else None
    return value.linear_map(weight, bias)


def _pz_scalar_interval(zonotope: PolynomialZonotope) -> Interval:
    """Return the scalar interval enclosure of a scalar polynomial zonotope."""

    enclosure = zonotope.interval_enclosure()
    lower = enclosure.lower
    upper = enclosure.upper
    if torch is not None and isinstance(lower, torch.Tensor):
        if lower.numel() != 1 or upper.numel() != 1:
            raise ValueError("Expected a scalar polynomial-zonotope interval enclosure.")
        return Interval(float(lower.reshape(()).item()), float(upper.reshape(()).item()))
    if isinstance(lower, tuple) or isinstance(upper, tuple):
        raise ValueError("Expected a scalar polynomial-zonotope interval enclosure.")
    return Interval(float(lower), float(upper))


def _affine_enclosure_pz(
    Z_i: PolynomialZonotope,
    *,
    slope: float,
    intercept: float,
    radius: float,
) -> PolynomialZonotope:
    """Build ``slope * Z_i + intercept + radius * eta`` with pointwise eta."""

    return (slope * Z_i + intercept).add_independent_error(
        radius, kind="approximation_pointwise"
    )


def _pz_twojet_tanh_forward(jet: PZTwoJet, chebyshev_degree: int, residual_subdivisions: int) -> PZTwoJet:
    """Propagate a polynomial-zonotope two-jet through componentwise ``tanh``.

    For each scalar preactivation ``Z_i``, compute its interval enclosure and
    use certified affine-plus-pointwise-residual enclosures for ``tanh``,
    ``tanh'``, and ``tanh''``. The resulting scalar enclosures are propagated
    by the componentwise two-jet chain rule without silently replacing existing
    polynomial dependencies by intervals.
    """

    _require_torch()
    if jet.Y.shape == ():
        components = 1
    elif len(jet.Y.shape) == 1:
        components = jet.Y.shape[0]
    else:
        raise ValueError("_pz_twojet_tanh_forward expects a scalar or 1-D value zonotope.")

    y_items: list[PolynomialZonotope] = []
    j_items: list[PolynomialZonotope] = []
    h_items: list[PolynomialZonotope] = []

    current_noise = jet.Y.num_noise
    current_noise_kinds = jet.Y.noise_kinds
    for i in range(components):
        Z_i = jet.Y if jet.Y.shape == () else jet.Y[i]
        Z_i = Z_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds)
        interval_i = _pz_scalar_interval(Z_i)

        tanh_i = affine_tanh_enclosure(interval_i)
        tanh_prime_i = affine_tanh_prime_enclosure(interval_i)
        tanh_double_prime_i = affine_tanh_double_prime_enclosure(interval_i)

        Y_i = _affine_enclosure_pz(
            Z_i, slope=tanh_i.p, intercept=tanh_i.q, radius=tanh_i.delta
        )
        current_noise = Y_i.num_noise
        current_noise_kinds = Y_i.noise_kinds

        D1_i = _affine_enclosure_pz(
            Z_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds),
            slope=tanh_prime_i.p,
            intercept=tanh_prime_i.q,
            radius=tanh_prime_i.delta,
        )
        current_noise = D1_i.num_noise
        current_noise_kinds = D1_i.noise_kinds

        D2_i = _affine_enclosure_pz(
            Z_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds),
            slope=tanh_double_prime_i.p,
            intercept=tanh_double_prime_i.q,
            radius=tanh_double_prime_i.delta,
        )
        current_noise = D2_i.num_noise
        current_noise_kinds = D2_i.noise_kinds

        J_i = jet.J if components == 1 and jet.J.shape[:1] != (components,) else jet.J[i, :]
        H_i = jet.H if components == 1 and jet.H.shape[:1] != (components,) else jet.H[i, :, :]
        J_i = J_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds)
        H_i = H_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds)

        y_items.append(Y_i.with_num_noise(current_noise).with_noise_kinds(current_noise_kinds))
        j_items.append(D1_i * J_i)
        h_items.append(D2_i * J_i.tensor_product(J_i) + D1_i * H_i)

    if jet.Y.shape == ():
        return PZTwoJet(Y=y_items[0], J=j_items[0], H=h_items[0])
    return PZTwoJet(
        Y=PolynomialZonotope.stack(y_items, dim=0),
        J=PolynomialZonotope.stack(j_items, dim=0),
        H=PolynomialZonotope.stack(h_items, dim=0),
    )


def _pz_value_tanh_forward(
    value: PolynomialZonotope,
    chebyshev_degree: int,
    residual_subdivisions: int,
) -> PolynomialZonotope:
    """Propagate only function values through componentwise ``tanh``.

    The current activation enclosure is affine.  All neuron slopes,
    intercepts, and certified residual radii are therefore applied in one
    tensor operation, followed by one independent residual symbol per neuron.
    ``chebyshev_degree`` and ``residual_subdivisions`` remain accepted for API
    compatibility with the two-jet path.
    """

    del chebyshev_degree, residual_subdivisions
    _require_torch()
    if value.shape == ():
        components = 1
    elif len(value.shape) == 1:
        components = value.shape[0]
    else:
        raise ValueError("_pz_value_tanh_forward expects a scalar or 1-D value zonotope.")

    enclosure = value.interval_enclosure()
    lower = enclosure.lower
    upper = enclosure.upper
    if isinstance(lower, torch.Tensor):
        lower_values = lower.reshape(-1).detach().cpu().tolist()
        upper_values = upper.reshape(-1).detach().cpu().tolist()
    else:
        lower_values = [lower] if value.shape == () else list(lower)
        upper_values = [upper] if value.shape == () else list(upper)

    approximations = [
        affine_tanh_enclosure(Interval(float(lo), float(hi)))
        for lo, hi in zip(lower_values, upper_values)
    ]
    if len(approximations) != components:
        raise RuntimeError("Tanh enclosure component count does not match the PZ shape.")

    if isinstance(value.center, torch.Tensor):
        target_shape = value.center.shape
        slopes = torch.tensor(
            [item.p for item in approximations],
            dtype=value.center.dtype,
            device=value.center.device,
        ).reshape(target_shape)
        intercepts = torch.tensor(
            [item.q for item in approximations],
            dtype=value.center.dtype,
            device=value.center.device,
        ).reshape(target_shape)
        radii = torch.tensor(
            [item.delta for item in approximations],
            dtype=value.center.dtype,
            device=value.center.device,
        ).reshape(target_shape)
        affine = PolynomialZonotope(
            slopes * value.center + intercepts,
            {
                exponent: slopes * coefficient
                for exponent, coefficient in value.terms.items()
            },
            num_noise=value.num_noise,
            noise_kinds=value.noise_kinds,
        )
        return affine.add_independent_errors(
            radii,
            kind="approximation_pointwise",
        )

    items = []
    for index, approximation in enumerate(approximations):
        component = value if value.shape == () else value[index]
        items.append(
            _affine_enclosure_pz(
                component,
                slope=approximation.p,
                intercept=approximation.q,
                radius=approximation.delta,
            )
        )
    return items[0] if value.shape == () else PolynomialZonotope.stack(items, dim=0)


def _pz_value_forward_from_value(
    module,
    value: PolynomialZonotope,
    *,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    reduce: bool = False,
    return_trace: bool = False,
) -> PolynomialZonotope | PZValueTraceResult:
    """Propagate a function-value PZ without allocating derivative tensors."""

    _require_torch()
    if reduce:
        raise NotImplementedError("PZ value reduction is not implemented yet.")
    if isinstance(module, nn.Sequential):
        result = value
        records = (
            [_pz_value_trace_record(-1, "input", "Input", result)]
            if return_trace
            else []
        )
        for index, (name, child) in enumerate(module.named_children()):
            result = _pz_value_forward_from_value(
                child,
                result,
                chebyshev_degree=chebyshev_degree,
                residual_subdivisions=residual_subdivisions,
                reduce=reduce,
                return_trace=False,
            )
            if return_trace:
                records.append(
                    _pz_value_trace_record(
                        index,
                        name,
                        type(child).__name__,
                        result,
                    )
                )
        return PZValueTraceResult(final=result, records=records) if return_trace else result
    if isinstance(module, nn.Linear):
        result = _pz_value_linear_forward(module, value)
    elif isinstance(module, nn.Tanh):
        result = _pz_value_tanh_forward(
            value,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
    elif isinstance(module, nn.Identity):
        result = value
    elif isinstance(module, nn.Flatten):
        if len(value.shape) > 1:
            raise NotImplementedError(
                "PZ value Flatten currently supports already-flat vectors only."
            )
        result = value
    else:
        raise NotImplementedError(
            "PZ value forward currently supports nn.Sequential, nn.Linear, "
            "nn.Tanh, nn.Identity, and flat-vector nn.Flatten only; got "
            f"{type(module).__name__}."
        )
    if return_trace:
        return PZValueTraceResult(
            final=result,
            records=[
                _pz_value_trace_record(-1, "input", "Input", value),
                _pz_value_trace_record(0, "0", type(module).__name__, result),
            ],
        )
    return result


def _pz_twojet_forward_from_jet(
    module,
    jet: PZTwoJet,
    *,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    reduce: bool = False,
    return_trace: bool = False,
) -> PZTwoJet | PZTwoJetTraceResult:
    """Propagate an initialized two-jet through supported PyTorch modules."""

    _require_torch()
    if reduce:
        raise NotImplementedError("PZ two-jet reduction is not implemented yet.")
    if isinstance(module, nn.Sequential):
        result = jet
        records = [_pz_twojet_trace_record(-1, "input", "Input", result)] if return_trace else []
        for index, (name, child) in enumerate(module.named_children()):
            result = _pz_twojet_forward_from_jet(
                child,
                result,
                chebyshev_degree=chebyshev_degree,
                residual_subdivisions=residual_subdivisions,
                reduce=reduce,
                return_trace=False,
            )
            if return_trace:
                records.append(_pz_twojet_trace_record(index, name, type(child).__name__, result))
        return PZTwoJetTraceResult(final=result, records=records) if return_trace else result
    if isinstance(module, nn.Linear):
        result = _pz_twojet_linear_forward(module, jet)
    elif isinstance(module, nn.Tanh):
        result = _pz_twojet_tanh_forward(
            jet,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
        )
    elif isinstance(module, nn.Identity):
        result = jet
    elif isinstance(module, nn.Flatten):
        if len(jet.Y.shape) > 1:
            raise NotImplementedError("PZ two-jet Flatten currently supports already-flat vectors only.")
        result = jet
    else:
        raise NotImplementedError(
            f"PZ two-jet forward currently supports nn.Sequential, nn.Linear, nn.Tanh, nn.Identity, and flat-vector nn.Flatten only; got {type(module).__name__}."
        )
    if return_trace:
        records = [
            _pz_twojet_trace_record(-1, "input", "Input", jet),
            _pz_twojet_trace_record(0, "0", type(module).__name__, result),
        ]
        return PZTwoJetTraceResult(final=result, records=records)
    return result


def pz_twojet_forward(
    module,
    x: PolynomialZonotope,
    *,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    reduce: bool = False,
    input_dim: int | None = None,
    return_trace: bool = False,
) -> PZTwoJet | PZTwoJetTraceResult:
    """Evaluate a supported PyTorch module on a polynomial-zonotope two-jet.

    ``x`` must be a flat scalar/vector polynomial zonotope.  The returned
    two-jet contains polynomial-zonotope enclosures for the value, Jacobian,
    and Hessian with respect to the physical input coordinates.
    """

    _require_torch()
    if not isinstance(x, PolynomialZonotope):
        raise TypeError("pz_twojet_forward(module, x) requires x to be a PolynomialZonotope.")
    if len(x.shape) > 1:
        raise NotImplementedError("PZ two-jet forward currently supports scalar or flat-vector inputs only.")
    inferred_dim = 1 if x.shape == () else x.shape[0]
    dim = inferred_dim if input_dim is None else int(input_dim)
    if dim != inferred_dim:
        raise ValueError(f"input_dim={dim} does not match polynomial-zonotope input dimension {inferred_dim}.")
    jet = PZTwoJet.from_input(x, input_dim=dim)
    return _pz_twojet_forward_from_jet(
        module,
        jet,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
        reduce=reduce,
        return_trace=return_trace,
    )


def pz_value_forward(
    module,
    x: PolynomialZonotope,
    *,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    reduce: bool = False,
    return_trace: bool = False,
) -> PolynomialZonotope | PZValueTraceResult:
    """Evaluate only a network's function-value PZ enclosure.

    Unlike :func:`pz_twojet_forward`, this path never initializes or
    propagates Jacobian and Hessian coefficient tensors.  It is the intended
    forward routine for certified PZ ``L^2`` computation.
    """

    _require_torch()
    if not isinstance(x, PolynomialZonotope):
        raise TypeError("pz_value_forward(module, x) requires x to be a PolynomialZonotope.")
    if len(x.shape) > 1:
        raise NotImplementedError(
            "PZ value forward currently supports scalar or flat-vector inputs only."
        )
    if not isinstance(x.center, torch.Tensor):
        parameter = next(module.parameters(), None)
        dtype = (
            parameter.dtype
            if parameter is not None and parameter.is_floating_point()
            else torch.float64
        )
        device = parameter.device if parameter is not None else None
        x = PolynomialZonotope(
            torch.as_tensor(x.center, dtype=dtype, device=device),
            {
                exponent: torch.as_tensor(
                    coefficient,
                    dtype=dtype,
                    device=device,
                )
                for exponent, coefficient in x.terms.items()
            },
            num_noise=x.num_noise,
            noise_kinds=x.noise_kinds,
        )
    return _pz_value_forward_from_value(
        module,
        x,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
        reduce=reduce,
        return_trace=return_trace,
    )


def pz_l2norm(
    module,
    domain: IntervalTensor,
    p: float = 2.0,
    *,
    iterations: int = 0,
    theta: float = 0.5,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    output: str = "interval",
) -> Interval:
    """Return a value-only PZ enclosure of a module's L2 norm over ``domain``."""

    _require_torch()
    if not isfinite(float(p)) or float(p) != 2.0:
        raise NotImplementedError("Polynomial-zonotope norm helpers currently support only p=2.0.")
    if not isinstance(domain, IntervalTensor):
        raise TypeError("pz_l2norm(module, domain) requires an IntervalTensor domain.")
    return pz_l2norm_bounds(
        module,
        domain,
        iterations=iterations,
        theta=theta,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
        output=output,
    )


def pz_sobolev_norm(
    module,
    domain: IntervalTensor,
    p: float = 2.0,
    order: int = 1,
    *,
    iterations: int = 0,
    theta: float = 0.5,
    chebyshev_degree: int = 5,
    residual_subdivisions: int = 128,
    output: str = "interval",
) -> Interval:
    """Return a PZ two-jet enclosure of a module's W^{order,2} norm."""

    _require_torch()
    if not isfinite(float(p)) or float(p) != 2.0:
        raise NotImplementedError("Polynomial-zonotope norm helpers currently support only p=2.0.")
    if not isinstance(domain, IntervalTensor):
        raise TypeError("pz_sobolev_norm(module, domain) requires an IntervalTensor domain.")
    if order not in {1, 2}:
        raise ValueError("order must be either 1 or 2.")
    return pz_sobolev_norm_bounds(
        module,
        domain,
        order=order,
        iterations=iterations,
        theta=theta,
        chebyshev_degree=chebyshev_degree,
        residual_subdivisions=residual_subdivisions,
        output=output,
    )

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


def _subdivide_box(box: IntervalTensor, splits_per_dim: int, max_cells: int) -> list[IntervalTensor]:
    if len(box.shape) != 1:
        raise NotImplementedError("Refinement currently supports flat vectors only.")
    if splits_per_dim < 1:
        raise ValueError("splits_per_dim must be at least 1.")

    dim = len(box.lower)
    total_cells = splits_per_dim ** dim
    if total_cells > max_cells:
        raise ValueError(
            f"Refinement would create {total_cells} cells which exceeds max_cells={max_cells}. "
            "Reduce splits_per_dim, input dimension, or increase max_cells."
        )
    if splits_per_dim == 1:
        return [box]

    per_dim_edges = [
        [float(box.lower[d] + (box.upper[d] - box.lower[d]) * idx / splits_per_dim) for idx in range(splits_per_dim + 1)]
        for d in range(dim)
    ]

    cells: list[IntervalTensor] = []
    for cell_index in product(range(splits_per_dim), repeat=dim):
        cell_lower = [per_dim_edges[d][cell_index[d]] for d in range(dim)]
        cell_upper = [per_dim_edges[d][cell_index[d] + 1] for d in range(dim)]
        cells.append(IntervalTensor.from_bounds(cell_lower, cell_upper))
    return cells


def _hull_intervals(intervals: list[Interval]) -> Interval:
    if not intervals:
        raise ValueError("Cannot hull an empty interval list.")
    lower = min(float(item.lower) for item in intervals)
    upper = max(float(item.upper) for item in intervals)
    return Interval.from_bounds(lower, upper)


def _lp_pointwise_power_bounds_refined(
    model,
    box: IntervalTensor,
    p: float,
    forward_refine_splits: int,
    forward_refine_max_cells: int,
) -> Interval:
    if forward_refine_splits <= 1:
        return _lp_pointwise_power_bounds(model, box, p)
    cells = _subdivide_box(box, splits_per_dim=forward_refine_splits, max_cells=forward_refine_max_cells)
    return _hull_intervals([_lp_pointwise_power_bounds(model, cell, p) for cell in cells])


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


def _lpnorm_bounds(
    model,
    domain: IntervalTensor,
    p: float,
    iterations: int,
    theta: float,
    enclosure_mode: str = "slope",
    forward_refine_splits: int = 1,
    forward_refine_max_cells: int = 256,
) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.lpnorm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Lp integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    _validate_dorfler_theta(theta)
    if forward_refine_splits < 1:
        raise ValueError("forward_refine_splits must be at least 1.")

    boxes = [domain]
    use_jacobian_splitting = len(domain.lower) > 1
    for _ in range(iterations):
        indicators: list[float] = []
        split_dims: list[int] = []
        for box in boxes:
            integrand_bounds = _lp_pointwise_power_bounds_refined(
                model,
                box,
                p,
                forward_refine_splits=forward_refine_splits,
                forward_refine_max_cells=forward_refine_max_cells,
            )
            width = float(integrand_bounds.upper) - float(integrand_bounds.lower)
            indicators.append(width * _box_volume(box))
            if use_jacobian_splitting:
                jacobian = model.eval_jacobian(box)
                split_dims.append(_choose_split_dim(box, jacobian))
            else:
                split_dims.append(_choose_split_dim(box, None))

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
        integrand_bounds = _lp_pointwise_power_bounds_refined(
            model,
            box,
            p,
            forward_refine_splits=forward_refine_splits,
            forward_refine_max_cells=forward_refine_max_cells,
        )
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


def _interval_second_derivative_bounds_relu(value: Interval) -> Interval:
    _ = value
    return Interval.point(0.0)


def _interval_second_derivative_bounds_sigmoid(value: Interval) -> Interval:
    sigmoid_bounds = _apply_monotone_bounds(IntervalTensor((value.lower,), (value.upper,)), _sigmoid_scalar)
    sigma = Interval(sigmoid_bounds.lower[0], sigmoid_bounds.upper[0])
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    return sigma * (one - sigma) * (one - (two * sigma))


def _interval_second_derivative_bounds_tanh(value: Interval) -> Interval:
    tanh_bounds = _apply_monotone_bounds(IntervalTensor((value.lower,), (value.upper,)), tanh)
    tanh_interval = Interval(tanh_bounds.lower[0], tanh_bounds.upper[0])
    one = Interval.point(1.0)
    two = Interval.point(2.0)
    return -(two * tanh_interval * (one - (tanh_interval * tanh_interval)))


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

    lower_raw = output_mid - output_rad
    upper_raw = output_mid + output_rad
    lower = torch.nextafter(lower_raw, torch.full_like(output_mid, float("-inf")))
    upper = torch.nextafter(upper_raw, torch.full_like(output_mid, float("inf")))

    # Preserve mathematically exact zeros. In particular, products involving an
    # exact zero interval should remain [0, 0] rather than being widened to a
    # tiny outward-rounded enclosure around zero.
    exact_zero_mask = (output_mid == 0.0) & (output_rad == 0.0)
    lower = torch.where(exact_zero_mask, torch.zeros_like(lower), lower)
    upper = torch.where(exact_zero_mask, torch.zeros_like(upper), upper)
    lower_rows = lower.tolist()
    upper_rows = upper.tolist()
    return [
        [Interval.from_bounds(lower_rows[row_idx][col_idx], upper_rows[row_idx][col_idx]) for col_idx in range(len(lower_rows[row_idx]))]
        for row_idx in range(len(lower_rows))
    ]


def _zero_hessian(output_dim: int, input_dim: int) -> list[list[list[Interval]]]:
    return [
        [[Interval.point(0.0) for _ in range(input_dim)] for _ in range(input_dim)]
        for _ in range(output_dim)
    ]


def _outer_product_interval(row_left: list[Interval], row_right: list[Interval]) -> list[list[Interval]]:
    size = len(row_left)
    if size != len(row_right):
        raise ValueError("Rows must have matching lengths for outer-product intervals.")
    return [
        [row_left[i] * row_right[j] for j in range(size)]
        for i in range(size)
    ]


def _hessian_compose(
    local_jacobian: list[list[Interval]],
    local_hessian: list[list[list[Interval]]],
    previous_jacobian: list[list[Interval]],
    previous_hessian: list[list[list[Interval]]],
) -> tuple[list[list[Interval]], list[list[list[Interval]]]]:
    new_jacobian = _matrix_multiply(local_jacobian, previous_jacobian)
    if not local_jacobian:
        return new_jacobian, []

    output_dim = len(local_jacobian)
    layer_input_dim = len(local_jacobian[0])
    base_input_dim = len(previous_jacobian[0]) if previous_jacobian else 0
    new_hessian = _zero_hessian(output_dim, base_input_dim)

    for out_idx in range(output_dim):
        acc = [[Interval.point(0.0) for _ in range(base_input_dim)] for _ in range(base_input_dim)]

        # Chain-rule term: sum_a J_g[k,a] * H_f[a,:,:]
        for a in range(layer_input_dim):
            coeff = local_jacobian[out_idx][a]
            for i in range(base_input_dim):
                for j in range(base_input_dim):
                    acc[i][j] = acc[i][j] + (coeff * previous_hessian[a][i][j])

        # Curvature term: sum_{a,b} H_g[k,a,b] * J_f[a,:] ⊗ J_f[b,:]
        for a in range(layer_input_dim):
            for b in range(layer_input_dim):
                coeff_h = local_hessian[out_idx][a][b]
                if float(coeff_h.lower) == 0.0 and float(coeff_h.upper) == 0.0:
                    continue
                outer = _outer_product_interval(previous_jacobian[a], previous_jacobian[b])
                for i in range(base_input_dim):
                    for j in range(base_input_dim):
                        acc[i][j] = acc[i][j] + (coeff_h * outer[i][j])

        new_hessian[out_idx] = acc

    return new_jacobian, new_hessian


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


def _hessian_for_layer(layer, pre_activation: IntervalTensor) -> list[list[list[Interval]]]:
    if len(pre_activation.shape) != 1:
        raise NotImplementedError("Interval Hessians currently support flat vectors only.")
    size = len(pre_activation.lower)
    if isinstance(layer, nn.Linear):
        return _zero_hessian(layer.out_features, size)
    if isinstance(layer, nn.Flatten):
        return _zero_hessian(size, size)
    if isinstance(layer, nn.ReLU):
        second_derivatives = [
            _interval_second_derivative_bounds_relu(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(size)
        ]
        tensor = _zero_hessian(size, size)
        for idx, val in enumerate(second_derivatives):
            tensor[idx][idx][idx] = val
        return tensor
    if isinstance(layer, nn.Sigmoid):
        second_derivatives = [
            _interval_second_derivative_bounds_sigmoid(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(size)
        ]
        tensor = _zero_hessian(size, size)
        for idx, val in enumerate(second_derivatives):
            tensor[idx][idx][idx] = val
        return tensor
    if isinstance(layer, nn.Tanh):
        second_derivatives = [
            _interval_second_derivative_bounds_tanh(Interval(pre_activation.lower[idx], pre_activation.upper[idx]))
            for idx in range(size)
        ]
        tensor = _zero_hessian(size, size)
        for idx, val in enumerate(second_derivatives):
            tensor[idx][idx][idx] = val
        return tensor
    raise NotImplementedError(
        f"Interval Hessian currently supports nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, and nn.Flatten; got {type(layer).__name__}."
    )


def _sequential_layer_inputs(module: nn.Sequential, domain: IntervalTensor, enclosure_mode: str) -> list[IntervalTensor]:
    children = list(module.children())
    if not children:
        return []

    layer_inputs: list[IntervalTensor] = [domain]
    for idx in range(1, len(children)):
        prefix = nn.Sequential(*children[:idx])
        layer_inputs.append(interval_forward(prefix, domain, enclosure_mode=enclosure_mode))
    return layer_inputs


def _eval_jacobian_bounds(model, domain: IntervalTensor, enclosure_mode: str = "box") -> IntervalTensor:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.eval_jacobian(domain) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Interval Jacobian evaluation currently supports flat input boxes only.")
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")

    if isinstance(model, nn.Sequential):
        layer_inputs = _sequential_layer_inputs(model, domain, enclosure_mode=enclosure_mode)
        current_jacobian = _identity_jacobian(len(domain.lower))
        for child, pre_activation in zip(model, layer_inputs):
            local_jacobian = _jacobian_for_layer(child, pre_activation)
            current_jacobian = _matrix_multiply(local_jacobian, current_jacobian)
    else:
        local_jacobian = _jacobian_for_layer(model, domain)
        current_jacobian = local_jacobian

    lower = tuple(tuple(entry.lower for entry in row) for row in current_jacobian)
    upper = tuple(tuple(entry.upper for entry in row) for row in current_jacobian)
    return IntervalTensor.from_bounds(lower, upper)


def _eval_hessian_bounds(model, domain: IntervalTensor, enclosure_mode: str = "box") -> IntervalTensor:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.eval_hessian(domain) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Interval Hessian evaluation currently supports flat input boxes only.")
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")

    input_dim = len(domain.lower)
    if isinstance(model, nn.Sequential):
        layer_inputs = _sequential_layer_inputs(model, domain, enclosure_mode=enclosure_mode)
        current_jacobian = _identity_jacobian(input_dim)
        current_hessian = _zero_hessian(input_dim, input_dim)
        for child, pre_activation in zip(model, layer_inputs):
            local_jacobian = _jacobian_for_layer(child, pre_activation)
            local_hessian = _hessian_for_layer(child, pre_activation)
            current_jacobian, current_hessian = _hessian_compose(
                local_jacobian,
                local_hessian,
                current_jacobian,
                current_hessian,
            )
    else:
        local_jacobian = _jacobian_for_layer(model, domain)
        local_hessian = _hessian_for_layer(model, domain)
        current_hessian = local_hessian

    lower = tuple(
        tuple(tuple(entry.lower for entry in row) for row in output_slice)
        for output_slice in current_hessian
    )
    upper = tuple(
        tuple(tuple(entry.upper for entry in row) for row in output_slice)
        for output_slice in current_hessian
    )
    return IntervalTensor.from_bounds(lower, upper)


def _sobolev_pointwise_power_bounds(
    model,
    box: IntervalTensor,
    p: float,
    order: int = 1,
    output: IntervalTensor | None = None,
    jacobian: IntervalTensor | None = None,
    hessian: IntervalTensor | None = None,
) -> Interval:
    if order not in {1, 2}:
        raise ValueError("order must be either 1 or 2.")
    output = output if output is not None else model.eval(box)
    jacobian = jacobian if jacobian is not None else model.eval_jacobian(box)
    if order == 2:
        hessian = hessian if hessian is not None else model.eval_hessian(box)
    total = Interval.point(0.0)

    for lower, upper in zip(output.lower, output.upper):
        component = Interval(lower, upper)
        total = total + _interval_pow_scalar(_interval_abs_bounds(component), p)

    for row_lower, row_upper in zip(jacobian.lower, jacobian.upper):
        for entry_lower, entry_upper in zip(row_lower, row_upper):
            derivative_component = Interval(entry_lower, entry_upper)
            total = total + _interval_pow_scalar(_interval_abs_bounds(derivative_component), p)

    if order == 2 and hessian is not None:
        for out_slice_lower, out_slice_upper in zip(hessian.lower, hessian.upper):
            for row_lower, row_upper in zip(out_slice_lower, out_slice_upper):
                for entry_lower, entry_upper in zip(row_lower, row_upper):
                    second_derivative_component = Interval(entry_lower, entry_upper)
                    total = total + _interval_pow_scalar(_interval_abs_bounds(second_derivative_component), p)

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


def _hessian_is_exact_zero(hessian: IntervalTensor) -> bool:
    return all(
        float(entry_lower) == 0.0 and float(entry_upper) == 0.0
        for out_slice_lower, out_slice_upper in zip(hessian.lower, hessian.upper)
        for row_lower, row_upper in zip(out_slice_lower, out_slice_upper)
        for entry_lower, entry_upper in zip(row_lower, row_upper)
    )


def _sobolev_pointwise_power_bounds_refined(
    model,
    box: IntervalTensor,
    p: float,
    order: int,
    forward_refine_splits: int,
    forward_refine_max_cells: int,
) -> Interval:
    if forward_refine_splits <= 1:
        return _sobolev_pointwise_power_bounds(model, box, p, order=order)
    cells = _subdivide_box(box, splits_per_dim=forward_refine_splits, max_cells=forward_refine_max_cells)
    return _hull_intervals([_sobolev_pointwise_power_bounds(model, cell, p, order=order) for cell in cells])


def _sobolev_norm_bounds(
    model,
    domain: IntervalTensor,
    p: float,
    order: int,
    iterations: int,
    theta: float,
    enclosure_mode: str = "slope",
    forward_refine_splits: int = 1,
    forward_refine_max_cells: int = 256,
) -> Interval:
    if not isinstance(domain, IntervalTensor):
        raise TypeError("model.sobolev_norm(domain, p, iterations) requires an IntervalTensor domain.")
    if len(domain.shape) != 1:
        raise NotImplementedError("Sobolev integration currently supports flat input boxes only.")
    if not isfinite(p) or p <= 0.0:
        raise ValueError("p must be a positive finite real number.")
    if order not in {1, 2}:
        raise ValueError("order must be either 1 or 2.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    _validate_dorfler_theta(theta)
    if forward_refine_splits < 1:
        raise ValueError("forward_refine_splits must be at least 1.")

    boxes = [domain]
    use_jacobian_splitting = len(domain.lower) > 1
    for _ in range(iterations):
        indicators: list[float] = []
        split_dims: list[int] = []
        for box in boxes:
            integrand_bounds = _sobolev_pointwise_power_bounds_refined(
                model,
                box,
                p,
                order,
                forward_refine_splits=forward_refine_splits,
                forward_refine_max_cells=forward_refine_max_cells,
            )
            output = model.eval(box)
            jacobian = model.eval_jacobian(box)
            hessian = model.eval_hessian(box) if order == 2 else None
            derivative_zero = _jacobian_is_exact_zero(jacobian)
            second_derivative_zero = True if hessian is None else _hessian_is_exact_zero(hessian)
            if _interval_tensor_is_exact_constant(output) and derivative_zero and second_derivative_zero:
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
        integrand_bounds = _sobolev_pointwise_power_bounds_refined(
            model,
            box,
            p,
            order,
            forward_refine_splits=forward_refine_splits,
            forward_refine_max_cells=forward_refine_max_cells,
        )
        weighted = Interval.from_bounds(
            float(integrand_bounds.lower) * _box_volume(box),
            float(integrand_bounds.upper) * _box_volume(box),
        )
        integral = integral + weighted

    non_negative = Interval.from_bounds(max(0.0, float(integral.lower)), max(0.0, float(integral.upper)))
    exponent = 1.0 / p
    return _interval_pow_scalar(non_negative, exponent)


def _interval_forward(module, x: IntervalTensor, enclosure_mode: str = "box") -> IntervalTensor:
    _require_torch()
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    if isinstance(module, nn.Sequential):
        if enclosure_mode == "slope":
            return _sequential_linear_relu_relaxation(module, x)
        result = x
        for child in module:
            result = _interval_forward(child, result, enclosure_mode=enclosure_mode)
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
        left = _interval_forward(module.left, x, enclosure_mode=enclosure_mode)
        right = _interval_forward(module.right, x, enclosure_mode=enclosure_mode)
        return _interval_add(left, right)
    if isinstance(module, IntervalCat):
        parts = [_interval_forward(branch, x, enclosure_mode=enclosure_mode) for branch in module.branches]
        return _interval_cat(parts, module.dim)
    raise NotImplementedError(
        f"Interval forward currently supports nn.Sequential, nn.Flatten, nn.Linear, nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softplus, nn.LeakyReLU, nn.Softmax, nn.Identity, IntervalAdd, and IntervalCat only; got {type(module).__name__}."
    )


def interval_forward(
    module,
    x: IntervalTensor,
    enclosure_mode: str = "box",
) -> IntervalTensor:
    if isinstance(x, IntervalTensor):
        return _interval_forward(module, x, enclosure_mode=enclosure_mode)
    raise TypeError("interval_forward(module, x) requires x to be an IntervalTensor.")


def interval_forward_refine(
    module,
    x: IntervalTensor,
    enclosure_mode: str = "slope",
    splits_per_dim: int = 2,
    max_cells: int = 256,
) -> IntervalTensor:
    """Refine interval forward bounds by subdividing the input box.

    This helper computes interval bounds on multiple sub-boxes and returns the
    hull over all outputs. It is conservative and never looser than evaluating
    `interval_forward(...)` once on the full input box.
    """
    _require_torch()
    if not isinstance(x, IntervalTensor):
        raise TypeError("interval_forward_refine(module, x, ...) requires x to be an IntervalTensor.")
    if len(x.shape) != 1:
        raise NotImplementedError("interval_forward_refine currently supports flat vectors only.")
    if splits_per_dim < 1:
        raise ValueError("splits_per_dim must be at least 1.")
    cells = _subdivide_box(x, splits_per_dim=splits_per_dim, max_cells=max_cells)

    hull_lower: tuple[float, ...] | None = None
    hull_upper: tuple[float, ...] | None = None

    for cell in cells:
        cell_out = _interval_forward(module, cell, enclosure_mode=enclosure_mode)
        lower = tuple(float(v) for v in cell_out.lower)
        upper = tuple(float(v) for v in cell_out.upper)

        if hull_lower is None or hull_upper is None:
            hull_lower = lower
            hull_upper = upper
            continue

        hull_lower = tuple(min(hull_lower[idx], lower[idx]) for idx in range(len(lower)))
        hull_upper = tuple(max(hull_upper[idx], upper[idx]) for idx in range(len(upper)))

    assert hull_lower is not None and hull_upper is not None
    return IntervalTensor.from_bounds(hull_lower, hull_upper)


_ORIGINAL_EVAL = getattr(nn.Module, "eval", None) if nn is not None else None
_PATCHED = False
_ACTIVE_ENCLOSURE_MODE = "slope"


def enable_interval_eval(enclosure_mode: str = "slope") -> None:
    _require_torch()
    global _PATCHED, _ACTIVE_ENCLOSURE_MODE
    if enclosure_mode not in {"box", "slope"}:
        raise ValueError("enclosure_mode must be either 'box' or 'slope'.")
    _ACTIVE_ENCLOSURE_MODE = enclosure_mode
    if _PATCHED:
        return

    def eval_with_interval(self, interval: IntervalTensor | None = None):
        result = _ORIGINAL_EVAL(self)
        if interval is None:
            return result
        if not isinstance(interval, IntervalTensor):
            raise TypeError("model.eval(interval) requires an IntervalTensor input.")
        return interval_forward(self, interval, enclosure_mode=_ACTIVE_ENCLOSURE_MODE)

    def lpnorm_with_interval(
        self,
        domain: IntervalTensor,
        p: float,
        iterations: int = 0,
        theta: float = 0.5,
        forward_refine_splits: int = 1,
        forward_refine_max_cells: int = 256,
        method: str = "interval",
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        output: str = "interval",
    ):
        _ORIGINAL_EVAL(self)
        if method == "pz":
            return pz_l2norm(
                self,
                domain,
                p=p,
                iterations=iterations,
                theta=theta,
                chebyshev_degree=chebyshev_degree,
                residual_subdivisions=residual_subdivisions,
                output=output,
            )
        if method != "interval":
            raise ValueError("method must be either 'interval' or 'pz'.")
        return _lpnorm_bounds(
            self,
            domain,
            p,
            iterations,
            theta,
            enclosure_mode=_ACTIVE_ENCLOSURE_MODE,
            forward_refine_splits=forward_refine_splits,
            forward_refine_max_cells=forward_refine_max_cells,
        )

    def eval_jacobian_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _eval_jacobian_bounds(self, domain, enclosure_mode=_ACTIVE_ENCLOSURE_MODE)

    def eval_hessian_with_interval(self, domain: IntervalTensor):
        _ORIGINAL_EVAL(self)
        return _eval_hessian_bounds(self, domain, enclosure_mode=_ACTIVE_ENCLOSURE_MODE)

    def eval_pz_twojet_with_interval(
        self,
        domain: PolynomialZonotope,
        *,
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        reduce: bool = False,
        return_trace: bool = False,
    ):
        _ORIGINAL_EVAL(self)
        if not isinstance(domain, PolynomialZonotope):
            raise TypeError("model.eval_pz_twojet(domain) requires a PolynomialZonotope input.")
        return pz_twojet_forward(
            self,
            domain,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
            reduce=reduce,
            return_trace=return_trace,
        )

    def eval_pz_value_with_interval(
        self,
        domain: PolynomialZonotope,
        *,
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        reduce: bool = False,
        return_trace: bool = False,
    ):
        _ORIGINAL_EVAL(self)
        if not isinstance(domain, PolynomialZonotope):
            raise TypeError(
                "model.eval_pz_value(domain) requires a PolynomialZonotope input."
            )
        return pz_value_forward(
            self,
            domain,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
            reduce=reduce,
            return_trace=return_trace,
        )

    def pz_l2norm_with_interval(
        self,
        domain: IntervalTensor,
        p: float = 2.0,
        *,
        iterations: int = 0,
        theta: float = 0.5,
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        output: str = "interval",
    ):
        _ORIGINAL_EVAL(self)
        return pz_l2norm(
            self,
            domain,
            p=p,
            iterations=iterations,
            theta=theta,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
            output=output,
        )

    def pz_sobolev_norm_with_interval(
        self,
        domain: IntervalTensor,
        p: float = 2.0,
        order: int = 1,
        *,
        iterations: int = 0,
        theta: float = 0.5,
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        output: str = "interval",
    ):
        _ORIGINAL_EVAL(self)
        return pz_sobolev_norm(
            self,
            domain,
            p=p,
            order=order,
            iterations=iterations,
            theta=theta,
            chebyshev_degree=chebyshev_degree,
            residual_subdivisions=residual_subdivisions,
            output=output,
        )

    def sobolev_norm_with_interval(
        self,
        domain: IntervalTensor,
        p: float,
        order: int = 1,
        iterations: int = 0,
        theta: float = 0.5,
        forward_refine_splits: int = 1,
        forward_refine_max_cells: int = 256,
        method: str = "interval",
        chebyshev_degree: int = 5,
        residual_subdivisions: int = 128,
        output: str = "interval",
    ):
        _ORIGINAL_EVAL(self)
        if method == "pz":
            return pz_sobolev_norm(
                self,
                domain,
                p=p,
                order=order,
                iterations=iterations,
                theta=theta,
                chebyshev_degree=chebyshev_degree,
                residual_subdivisions=residual_subdivisions,
                output=output,
            )
        if method != "interval":
            raise ValueError("method must be either 'interval' or 'pz'.")
        return _sobolev_norm_bounds(
            self,
            domain,
            p,
            order,
            iterations,
            theta,
            enclosure_mode=_ACTIVE_ENCLOSURE_MODE,
            forward_refine_splits=forward_refine_splits,
            forward_refine_max_cells=forward_refine_max_cells,
        )

    nn.Module.eval = eval_with_interval
    nn.Module.lpnorm = lpnorm_with_interval
    nn.Module.eval_jacobian = eval_jacobian_with_interval
    nn.Module.eval_hessian = eval_hessian_with_interval
    nn.Module.eval_pz_value = eval_pz_value_with_interval
    nn.Module.eval_pz_twojet = eval_pz_twojet_with_interval
    nn.Module.pz_l2norm = pz_l2norm_with_interval
    nn.Module.pz_sobolev_norm = pz_sobolev_norm_with_interval
    nn.Module.sobolev_norm = sobolev_norm_with_interval
    _PATCHED = True
