"""Benchmark refined hybrid L2/W12 certificates on both saved 100D PINNs."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from intervalnets import (
    DeepHybridOneJetResult,
    IntervalTensor,
    PZIntegrationCell,
    integrate_hybrid_onejet_squared,
    integrate_hybrid_value_squared,
    load_tanh_mlp_checkpoint,
    scalar_hybrid_onejet_reverse,
)


SEED = 20260806
REPEATS = 7
VALIDATION_SAMPLES = 16_384


def _architecture(model: torch.nn.Sequential) -> list[int]:
    linears = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    return [linears[0].in_features, *(layer.out_features for layer in linears)]


def _norm_record(squared, volume: float) -> dict[str, float]:
    squared_lower = max(0.0, float(squared.lower))
    squared_upper = max(0.0, float(squared.upper))
    lower = sqrt(squared_lower)
    upper = sqrt(squared_upper)
    normalized_squared_lower = squared_lower / volume
    normalized_squared_upper = squared_upper / volume
    normalized_lower = sqrt(normalized_squared_lower)
    normalized_upper = sqrt(normalized_squared_upper)
    return {
        "squared_lower": squared_lower,
        "squared_upper": squared_upper,
        "squared_absolute_width": squared_upper - squared_lower,
        "squared_relative_width": (
            (squared_upper - squared_lower) / squared_upper
            if squared_upper > 0.0
            else 0.0
        ),
        "lower": lower,
        "upper": upper,
        "absolute_width": upper - lower,
        "relative_width": (upper - lower) / upper if upper > 0.0 else 0.0,
        "domain_volume_normalized_squared_lower": normalized_squared_lower,
        "domain_volume_normalized_squared_upper": normalized_squared_upper,
        "domain_volume_normalized_squared_width": (
            normalized_squared_upper - normalized_squared_lower
        ),
        "domain_volume_normalized_lower": normalized_lower,
        "domain_volume_normalized_upper": normalized_upper,
        "domain_volume_normalized_width": normalized_upper - normalized_lower,
    }


def _benchmark(checkpoint: Path) -> dict[str, object]:
    model = load_tanh_mlp_checkpoint(checkpoint).double().eval()
    box = IntervalTensor.from_bounds([-0.1] * 100, [0.1] * 100)
    cell = PZIntegrationCell.from_affine_box(box)

    warm = scalar_hybrid_onejet_reverse(model, cell.domain)
    integrate_hybrid_value_squared(warm, cell)
    integrate_hybrid_onejet_squared(warm, cell)

    construction_times: list[float] = []
    l2_times: list[float] = []
    w12_times: list[float] = []
    result = warm
    l2_squared = None
    w12_squared = None
    for _ in range(REPEATS):
        start = perf_counter()
        result = scalar_hybrid_onejet_reverse(model, cell.domain)
        constructed = perf_counter()
        l2_squared = integrate_hybrid_value_squared(result, cell)
        l2_done = perf_counter()
        w12_squared = integrate_hybrid_onejet_squared(result, cell)
        done = perf_counter()
        construction_times.append(constructed - start)
        l2_times.append(l2_done - constructed)
        w12_times.append(done - l2_done)
    assert l2_squared is not None and w12_squared is not None

    generator = torch.Generator().manual_seed(SEED + 222)
    samples = -0.1 + 0.2 * torch.rand(
        (VALIDATION_SAMPLES, 100), generator=generator, dtype=torch.float64
    )
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    sampled_l2_squared = float(values.square().mean().detach())
    sampled_w12_squared = float(
        (values.square().squeeze(1) + gradients.square().sum(1)).mean().detach()
    )

    l2 = _norm_record(l2_squared, float(cell.volume))
    w12 = _norm_record(w12_squared, float(cell.volume))
    l2["sampled_domain_volume_normalized"] = sqrt(sampled_l2_squared)
    w12["sampled_domain_volume_normalized"] = sqrt(sampled_w12_squared)

    value_enclosure = result.value.interval_enclosure() if isinstance(result, DeepHybridOneJetResult) else result.final.Y.interval_enclosure()
    jacobian_enclosure = result.jacobian.interval_enclosure() if isinstance(result, DeepHybridOneJetResult) else result.final.J.interval_enclosure()
    value_lower = torch.as_tensor(value_enclosure.lower).reshape(1, 1)
    value_upper = torch.as_tensor(value_enclosure.upper).reshape(1, 1)
    jacobian_lower = torch.as_tensor(jacobian_enclosure.lower).reshape(1, 100)
    jacobian_upper = torch.as_tensor(jacobian_enclosure.upper).reshape(1, 100)

    record: dict[str, object] = {
        "checkpoint": str(checkpoint.relative_to(checkpoint.parents[2])),
        "architecture": _architecture(model),
        "physical_domain_volume": float(cell.volume),
        "integration_semantics": "pointwise_residual_absolute_moments_parity",
        "l2": l2,
        "w12": w12,
        "timings_seconds_median": {
            "onejet_construction": median(construction_times),
            "l2_integration": median(l2_times),
            "w12_integration": median(w12_times),
            "construction_plus_l2_plus_w12": median(
                [a + b + c for a, b, c in zip(construction_times, l2_times, w12_times)]
            ),
        },
        "sample_soundness_violations": {
            "value": int(torch.count_nonzero((values < value_lower) | (values > value_upper))),
            "jacobian_entries": int(
                torch.count_nonzero((gradients < jacobian_lower) | (gradients > jacobian_upper))
            ),
        },
    }
    if isinstance(result, DeepHybridOneJetResult):
        record["quadratic_neurons_per_layer"] = [
            int(torch.count_nonzero(factor.approximation_degrees == 2))
            for factor in result.factors
        ]
        record["deep_w12_integration_note"] = (
            "absolute moments and parity are applied after the exact factored "
            "Jacobian is collapsed to its affine core plus certified pointwise remainder"
        )
    else:
        record["quadratic_neurons_per_layer"] = [
            int(torch.count_nonzero(result.derivative_degrees == 2))
        ]
    return record


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    root = Path(__file__).resolve().parents[1]
    checkpoints = [
        root / "notebooks" / "checkpoints" / "pinn_100d_poisson_shallow_300.pt",
        root / "notebooks" / "checkpoints" / "pinn_100d_poisson.pt",
    ]
    records = [_benchmark(path) for path in checkpoints]
    output = root / "notebooks" / "benchmark_outputs" / "hybrid_absolute_moment_norms.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
