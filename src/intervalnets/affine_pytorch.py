from __future__ import annotations

from typing import Literal

from .affine import AffineTensor

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


_AFFINE_TANH_MODES = {"chebyshev", "min_range"}


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for affine PyTorch activation transforms.")


def _require_torch_affine_vector(x: AffineTensor) -> tuple[torch.Tensor, torch.Tensor]:
    _require_torch()
    if not isinstance(x.c, torch.Tensor) or not isinstance(x.G, torch.Tensor):
        raise TypeError("Affine activation transforms currently require torch-backed AffineTensor inputs.")
    if x.c.ndim != 1:
        raise ValueError(f"Expected 1D affine center, got shape {tuple(x.c.shape)}.")
    if x.G.ndim != 2 or x.G.shape[0] != x.c.shape[0]:
        raise ValueError(
            f"Expected generator matrix of shape (n, k) matching center shape {(x.c.shape[0],)}, got {tuple(x.G.shape)}."
        )
    return x.c.to(dtype=torch.float64), x.G.to(dtype=torch.float64)


def _vectorized_line_from_endpoints(
    lower: torch.Tensor,
    upper: torch.Tensor,
    f_lower: torch.Tensor,
    f_upper: torch.Tensor,
    degenerate_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = upper - lower
    safe_width = torch.where(degenerate_mask, torch.ones_like(width), width)
    alpha = (f_upper - f_lower) / safe_width
    beta = 0.5 * (f_lower + f_upper - alpha * (lower + upper))
    alpha = torch.where(degenerate_mask, torch.zeros_like(alpha), alpha)
    beta = torch.where(degenerate_mask, f_lower, beta)
    return alpha, beta


def _sampled_eps_bound(
    lower: torch.Tensor,
    upper: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    func,
    samples: int = 257,
) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, steps=samples, dtype=lower.dtype, device=lower.device)
    points = lower.unsqueeze(-1) + (upper - lower).unsqueeze(-1) * grid
    values = func(points)
    linear_values = alpha.unsqueeze(-1) * points + beta.unsqueeze(-1)
    eps = torch.max(torch.abs(values - linear_values), dim=-1).values
    eps = torch.nextafter(eps, torch.full_like(eps, float("inf")))
    return torch.clamp(eps, min=0.0)


def _tanh_residual_extrema(lower: torch.Tensor, upper: torch.Tensor, slope: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual_lower = torch.tanh(lower) - slope * lower
    residual_upper = torch.tanh(upper) - slope * upper

    r_min = torch.minimum(residual_lower, residual_upper)
    r_max = torch.maximum(residual_lower, residual_upper)

    slope_clamped = torch.clamp(slope, min=0.0, max=1.0)
    root_tanh_abs = torch.sqrt(torch.clamp(1.0 - slope_clamped, min=0.0))

    eps = torch.finfo(lower.dtype).eps
    root_tanh_abs = torch.clamp(root_tanh_abs, max=1.0 - eps)

    x_pos = torch.atanh(root_tanh_abs)
    x_neg = -x_pos

    pos_inside = (x_pos >= lower) & (x_pos <= upper)
    neg_inside = (x_neg >= lower) & (x_neg <= upper)

    residual_pos = torch.tanh(x_pos) - slope * x_pos
    residual_neg = torch.tanh(x_neg) - slope * x_neg

    r_min = torch.where(pos_inside, torch.minimum(r_min, residual_pos), r_min)
    r_max = torch.where(pos_inside, torch.maximum(r_max, residual_pos), r_max)
    r_min = torch.where(neg_inside, torch.minimum(r_min, residual_neg), r_min)
    r_max = torch.where(neg_inside, torch.maximum(r_max, residual_neg), r_max)
    return r_min, r_max


def _tanh_delta_for_slope(lower: torch.Tensor, upper: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    r_min, r_max = _tanh_residual_extrema(lower, upper, slope)
    return 0.5 * (r_max - r_min)


def _objective_for_slope(
    lower: torch.Tensor,
    upper: torch.Tensor,
    slope: torch.Tensor,
    mode: Literal["chebyshev", "min_range"],
) -> torch.Tensor:
    delta = _tanh_delta_for_slope(lower, upper, slope)
    if mode == "chebyshev":
        return delta
    half_width = 0.5 * (upper - lower)
    return slope * half_width + delta


def _optimize_tanh_slope(lower: torch.Tensor, upper: torch.Tensor, mode: Literal["chebyshev", "min_range"], iterations: int = 64) -> torch.Tensor:
    left = torch.zeros_like(lower)
    right = torch.ones_like(lower)
    phi = (5.0**0.5 - 1.0) / 2.0

    c = right - phi * (right - left)
    d = left + phi * (right - left)
    fc = _objective_for_slope(lower, upper, c, mode)
    fd = _objective_for_slope(lower, upper, d, mode)

    for _ in range(iterations):
        move_left = fc > fd
        left = torch.where(move_left, c, left)
        right = torch.where(move_left, right, d)

        c = right - phi * (right - left)
        d = left + phi * (right - left)
        fc = _objective_for_slope(lower, upper, c, mode)
        fd = _objective_for_slope(lower, upper, d, mode)

    slope = 0.5 * (left + right)
    return torch.clamp(slope, min=0.0, max=1.0)


def _tanh_affine_parameters(
    lower: torch.Tensor,
    upper: torch.Tensor,
    mode: Literal["chebyshev", "min_range"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    slope = _optimize_tanh_slope(lower, upper, mode=mode)
    r_min, r_max = _tanh_residual_extrema(lower, upper, slope)
    offset = 0.5 * (r_min + r_max)
    delta = 0.5 * (r_max - r_min)

    delta = torch.nextafter(delta, torch.full_like(delta, float("inf")))
    delta = torch.clamp(delta, min=0.0)
    return slope, offset, delta


def _append_error_generators(alpha: torch.Tensor, beta: torch.Tensor, eps: torch.Tensor, center: torch.Tensor, generators: torch.Tensor) -> AffineTensor:
    transformed_center = alpha * center + beta
    scaled_generators = alpha.unsqueeze(-1) * generators
    n = center.shape[0]
    fresh_noise = torch.diag_embed(eps)
    transformed_generators = torch.cat((scaled_generators, fresh_noise), dim=-1)
    return AffineTensor(transformed_center, transformed_generators)


def affine_relu_transform(x: AffineTensor) -> AffineTensor:
    center, generators = _require_torch_affine_vector(x)
    lower, upper = x.to_bounds()
    lower = lower.to(dtype=torch.float64)
    upper = upper.to(dtype=torch.float64)

    degenerate_mask = lower == upper
    f_lower = torch.relu(lower)
    f_upper = torch.relu(upper)
    alpha, beta = _vectorized_line_from_endpoints(lower, upper, f_lower, f_upper, degenerate_mask)

    crossing = (lower < 0.0) & (upper > 0.0)
    line_at_zero = beta
    endpoint_error_lower = torch.abs(alpha * lower + beta - f_lower)
    endpoint_error_upper = torch.abs(alpha * upper + beta - f_upper)
    crossing_eps = torch.maximum(torch.maximum(endpoint_error_lower, endpoint_error_upper), torch.abs(line_at_zero))

    eps = torch.zeros_like(lower)
    eps = torch.where(crossing, crossing_eps, eps)
    eps = torch.where(degenerate_mask, torch.zeros_like(eps), eps)
    eps = torch.nextafter(eps, torch.full_like(eps, float("inf")))
    return _append_error_generators(alpha, beta, eps, center, generators)


def affine_tanh_transform(x: AffineTensor, mode: str = "min_range") -> AffineTensor:
    if mode not in _AFFINE_TANH_MODES:
        raise ValueError("mode must be either 'chebyshev' or 'min_range'.")

    center, generators = _require_torch_affine_vector(x)
    lower, upper = x.to_bounds()
    lower = lower.to(dtype=torch.float64)
    upper = upper.to(dtype=torch.float64)

    degenerate_mask = lower == upper
    alpha, beta, eps = _tanh_affine_parameters(lower, upper, mode=mode)

    exact_value = torch.tanh(lower)
    alpha = torch.where(degenerate_mask, torch.zeros_like(alpha), alpha)
    beta = torch.where(degenerate_mask, exact_value, beta)
    eps = torch.where(degenerate_mask, torch.zeros_like(eps), eps)
    return _append_error_generators(alpha, beta, eps, center, generators)


def affine_sigmoid_transform(x: AffineTensor) -> AffineTensor:
    center, generators = _require_torch_affine_vector(x)
    lower, upper = x.to_bounds()
    lower = lower.to(dtype=torch.float64)
    upper = upper.to(dtype=torch.float64)

    degenerate_mask = lower == upper
    f_lower = torch.sigmoid(lower)
    f_upper = torch.sigmoid(upper)
    alpha, beta = _vectorized_line_from_endpoints(lower, upper, f_lower, f_upper, degenerate_mask)

    eps = _sampled_eps_bound(lower, upper, alpha, beta, torch.sigmoid)
    eps = torch.where(degenerate_mask, torch.zeros_like(eps), eps)
    return _append_error_generators(alpha, beta, eps, center, generators)
