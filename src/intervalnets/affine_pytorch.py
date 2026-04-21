from __future__ import annotations

from typing import Callable

from .affine import AffineTensor

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


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
    func: Callable[[torch.Tensor], torch.Tensor],
    samples: int = 257,
) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, steps=samples, dtype=lower.dtype, device=lower.device)
    points = lower.unsqueeze(-1) + (upper - lower).unsqueeze(-1) * grid
    values = func(points)
    linear_values = alpha.unsqueeze(-1) * points + beta.unsqueeze(-1)
    eps = torch.max(torch.abs(values - linear_values), dim=-1).values
    eps = torch.nextafter(eps, torch.full_like(eps, float("inf")))
    return torch.clamp(eps, min=0.0)


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


def affine_tanh_transform(x: AffineTensor) -> AffineTensor:
    center, generators = _require_torch_affine_vector(x)
    lower, upper = x.to_bounds()
    lower = lower.to(dtype=torch.float64)
    upper = upper.to(dtype=torch.float64)

    degenerate_mask = lower == upper
    f_lower = torch.tanh(lower)
    f_upper = torch.tanh(upper)
    alpha, beta = _vectorized_line_from_endpoints(lower, upper, f_lower, f_upper, degenerate_mask)

    eps = _sampled_eps_bound(lower, upper, alpha, beta, torch.tanh)
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
