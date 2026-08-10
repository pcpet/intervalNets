"""Benchmark the fast factored deep hybrid certificate on the saved 100D PINN."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from intervalnets import (
    IntervalTensor,
    PZIntegrationCell,
    DeepHybridOneJetResult,
    integrate_deep_hybrid_onejet_squared,
    load_tanh_mlp_checkpoint,
    scalar_hybrid_onejet_reverse,
)


SEED = 20260806
REPEATS = 7
VALIDATION_SAMPLES = 16_384


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "notebooks" / "checkpoints" / "pinn_100d_poisson.pt"
    model = load_tanh_mlp_checkpoint(checkpoint).double().eval()
    box = IntervalTensor.from_bounds([-0.1] * 100, [0.1] * 100)
    cell = PZIntegrationCell.from_affine_box(box)

    # One warm run initializes the activation-certificate and linear-algebra paths.
    warm = scalar_hybrid_onejet_reverse(model, cell.domain)
    assert isinstance(warm, DeepHybridOneJetResult)
    integrate_deep_hybrid_onejet_squared(warm, cell)

    construction_times: list[float] = []
    integration_times: list[float] = []
    total_times: list[float] = []
    result = warm
    squared = None
    for _ in range(REPEATS):
        start = perf_counter()
        result = scalar_hybrid_onejet_reverse(model, cell.domain)
        constructed = perf_counter()
        squared = integrate_deep_hybrid_onejet_squared(result, cell)
        finished = perf_counter()
        construction_times.append(constructed - start)
        integration_times.append(finished - constructed)
        total_times.append(finished - start)
    assert squared is not None

    generator = torch.Generator().manual_seed(SEED + 222)
    samples = -0.1 + 0.2 * torch.rand(
        (VALIDATION_SAMPLES, 100), generator=generator, dtype=torch.float64
    )
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    hidden = samples.detach()
    value_noise_blocks = [hidden / 0.1]
    derivative_noise_blocks = []
    hidden_linears = list(model.children())[0:-1:2]
    for layer, factor in zip(hidden_linears, result.factors):
        xi = torch.cat(value_noise_blocks, dim=1)
        preactivation = layer(hidden)
        polynomial_preactivation = (
            factor.preactivation_center.unsqueeze(0)
            + xi @ factor.preactivation_coefficients.T
        )
        if not torch.allclose(
            preactivation, polynomial_preactivation, rtol=2e-11, atol=2e-11
        ):
            raise AssertionError("Cached preactivation PZ does not realize the sample.")
        hidden = torch.tanh(preactivation)
        value_core = (
            factor.value_intercepts.unsqueeze(0)
            + factor.value_slopes.unsqueeze(0) * preactivation
        )
        value_eta = torch.where(
            factor.value_approximation_radii.unsqueeze(0) == 0.0,
            torch.zeros_like(hidden),
            (hidden - value_core) / factor.value_approximation_radii.unsqueeze(0),
        )
        affine_argument = xi @ factor.preactivation_coefficients.T
        derivative_core = (
            factor.center.unsqueeze(0)
            + xi @ factor.linear_coefficients.T
            + factor.quadratic_coefficients.unsqueeze(0) * affine_argument.square()
        )
        exact_derivative = 1.0 - hidden.square()
        derivative_eta = torch.where(
            factor.approximation_radii.unsqueeze(0) == 0.0,
            torch.zeros_like(hidden),
            (exact_derivative - derivative_core)
            / factor.approximation_radii.unsqueeze(0),
        )
        value_noise_blocks.append(value_eta)
        derivative_noise_blocks.append(derivative_eta)
    realizing_noise = torch.cat((*value_noise_blocks, *derivative_noise_blocks), dim=1)
    factored_gradients = result.jacobian.evaluate(realizing_noise)
    factor_gradient_max_error = float(
        torch.max(torch.abs(factored_gradients - gradients)).detach().item()
    )
    max_realizing_noise = float(torch.max(torch.abs(realizing_noise)).detach().item())
    value_enclosure = result.value.interval_enclosure()
    jacobian_enclosure = result.jacobian.interval_enclosure()
    value_lower = torch.as_tensor(value_enclosure.lower).reshape(1, 1)
    value_upper = torch.as_tensor(value_enclosure.upper).reshape(1, 1)
    jacobian_lower = torch.as_tensor(jacobian_enclosure.lower).reshape(1, 100)
    jacobian_upper = torch.as_tensor(jacobian_enclosure.upper).reshape(1, 100)
    value_violations = int(torch.count_nonzero((values < value_lower) | (values > value_upper)))
    jacobian_violations = int(
        torch.count_nonzero((gradients < jacobian_lower) | (gradients > jacobian_upper))
    )
    sampled_w12 = float(
        (values.square().squeeze(1) + gradients.square().sum(1)).mean().detach()
    )
    volume = float(cell.volume)
    normalized_squared_lower = float(squared.lower) / volume
    normalized_squared_upper = float(squared.upper) / volume

    print(
        {
            "architecture": [100, 50, 50, 50, 1],
            "checkpoint": str(checkpoint.relative_to(root)),
            "representation": "uncompressed_factored_reverse_hybrid",
            "hidden_layers": len(result.factors),
            "quadratic_neurons_per_layer": [
                int(torch.count_nonzero(factor.approximation_degrees == 2))
                for factor in result.factors
            ],
            "value_noise_symbols": result.jacobian.num_value_noise - 100,
            "derivative_noise_symbols": result.jacobian.num_derivative_noise,
            "factored_jacobian_degree": result.jacobian.max_degree,
            "gradient_spectral_bound": result.gradient_spectral_bound,
            "normalized_W12_squared_lower": normalized_squared_lower,
            "normalized_W12_squared_upper": normalized_squared_upper,
            "normalized_W12_lower": sqrt(max(0.0, normalized_squared_lower)),
            "normalized_W12_upper": sqrt(max(0.0, normalized_squared_upper)),
            "sampled_normalized_W12": sqrt(sampled_w12),
            "construction_seconds_median": median(construction_times),
            "integration_seconds_median": median(integration_times),
            "total_seconds_median": median(total_times),
            "total_seconds_range": [min(total_times), max(total_times)],
            "value_soundness_violations": value_violations,
            "jacobian_soundness_violations": jacobian_violations,
            "factor_gradient_max_error": factor_gradient_max_error,
            "max_realizing_noise_magnitude": max_realizing_noise,
        }
    )


if __name__ == "__main__":
    main()
