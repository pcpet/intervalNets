"""Benchmark dependency-preserving graph/Hilbert norm certificates."""

from __future__ import annotations

from dataclasses import asdict
import json
from math import sqrt
from pathlib import Path
from time import perf_counter

import torch

from intervalnets import (
    IntervalTensor,
    PZIntegrationCell,
    certify_hybrid_graph_norms,
    load_tanh_mlp_checkpoint,
    scalar_hybrid_onejet_reverse,
)


SEED = 20260806
VALIDATION_SAMPLES = 16_384
POLYNOMIAL_DEGREE = 5
RESIDUAL_SUBDIVISIONS = 2048
DERIVATIVE_CERTIFICATE_SUBDIVISIONS = 64


def _architecture(model: torch.nn.Sequential) -> list[int]:
    linears = [layer for layer in model if isinstance(layer, torch.nn.Linear)]
    return [linears[0].in_features, *(layer.out_features for layer in linears)]


def _norm_record(squared, volume: float, sampled: float) -> dict[str, float]:
    normalized_squared_lower = max(0.0, float(squared.lower) / volume)
    normalized_squared_upper = max(0.0, float(squared.upper) / volume)
    lower = sqrt(normalized_squared_lower)
    upper = sqrt(normalized_squared_upper)
    width = upper - lower
    return {
        "domain_volume_normalized_lower": lower,
        "domain_volume_normalized_upper": upper,
        "domain_volume_normalized_width": width,
        "relative_norm_width": width / upper if upper > 0.0 else 0.0,
        "sampled_domain_volume_normalized": sampled,
        "upper_over_sampled": upper / sampled,
        "lower_over_sampled": lower / sampled,
        "raw_squared_lower": float(squared.lower),
        "raw_squared_upper": float(squared.upper),
    }


def _benchmark(checkpoint: Path) -> dict[str, object]:
    model = load_tanh_mlp_checkpoint(checkpoint).double().eval()
    cell = PZIntegrationCell.from_affine_box(
        IntervalTensor.from_bounds([-0.1] * 100, [0.1] * 100)
    )
    construction_start = perf_counter()
    result = scalar_hybrid_onejet_reverse(model, cell.domain)
    construction_seconds = perf_counter() - construction_start

    certificate_start = perf_counter()
    certificate = certify_hybrid_graph_norms(
        model,
        result,
        cell,
        polynomial_degree=POLYNOMIAL_DEGREE,
        residual_subdivisions=RESIDUAL_SUBDIVISIONS,
        derivative_certificate_subdivisions=DERIVATIVE_CERTIFICATE_SUBDIVISIONS,
    )
    certificate_seconds = perf_counter() - certificate_start

    generator = torch.Generator().manual_seed(SEED + 222)
    samples = -0.1 + 0.2 * torch.rand(
        (VALIDATION_SAMPLES, 100), generator=generator, dtype=torch.float64
    )
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    sampled_l2 = sqrt(float(values.square().mean().detach()))
    sampled_w12 = sqrt(
        float(
            (
                values.square().squeeze(1) + gradients.square().sum(dim=1)
            ).mean().detach()
        )
    )
    volume = float(cell.volume)
    l2 = _norm_record(certificate.l2_squared, volume, sampled_l2)
    w12 = _norm_record(certificate.w12_squared, volume, sampled_w12)
    old_l2 = _norm_record(certificate.previous_l2_squared, volume, sampled_l2)
    old_w12 = _norm_record(certificate.previous_w12_squared, volume, sampled_w12)
    l2_contained = (
        l2["domain_volume_normalized_lower"]
        <= sampled_l2
        <= l2["domain_volume_normalized_upper"]
    )
    w12_contained = (
        w12["domain_volume_normalized_lower"]
        <= sampled_w12
        <= w12["domain_volume_normalized_upper"]
    )
    assert l2_contained and w12_contained

    return {
        "checkpoint": str(checkpoint.relative_to(checkpoint.parents[2])),
        "architecture": _architecture(model),
        "physical_domain_volume": volume,
        "method": "dependency_preserving_graph_hilbert",
        "settings": {
            "polynomial_degree": POLYNOMIAL_DEGREE,
            "residual_subdivisions": RESIDUAL_SUBDIVISIONS,
            "derivative_certificate_subdivisions": DERIVATIVE_CERTIFICATE_SUBDIVISIONS,
            "validation_samples": VALIDATION_SAMPLES,
        },
        "l2": l2,
        "w12": w12,
        "sample_containment_diagnostic": {
            "l2": l2_contained,
            "w12": w12_contained,
        },
        "previous_absolute_moment_parity": {"l2": old_l2, "w12": old_w12},
        "improvement": {
            "l2_relative_width_reduction": (
                old_l2["relative_norm_width"] - l2["relative_norm_width"]
            ),
            "w12_relative_width_reduction": (
                old_w12["relative_norm_width"] - w12["relative_norm_width"]
            ),
            "l2_upper_reduction_fraction": (
                old_l2["domain_volume_normalized_upper"]
                - l2["domain_volume_normalized_upper"]
            )
            / old_l2["domain_volume_normalized_upper"],
            "w12_upper_reduction_fraction": (
                old_w12["domain_volume_normalized_upper"]
                - w12["domain_volume_normalized_upper"]
            )
            / old_w12["domain_volume_normalized_upper"],
        },
        "decomposition": certificate.normalized_diagnostics,
        "value_layers": [asdict(layer) for layer in certificate.value.layers],
        "gradient": {
            "factor_remainders": list(certificate.gradient.factor_remainders),
            "product_projection_remainders": list(
                certificate.gradient.product_projection_remainders
            ),
            "moment_states": certificate.gradient.moment_states,
        },
        "w12_witness": asdict(certificate.w12_witness),
        "complexity": {
            "value_moment_states": certificate.value.moment_states,
            "gradient_moment_states": certificate.gradient.moment_states,
        },
        "timings_seconds": {
            "hybrid_onejet_construction": construction_seconds,
            "graph_hilbert_certification": certificate_seconds,
            "total_certification": construction_seconds + certificate_seconds,
        },
    }


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    root = Path(__file__).resolve().parents[1]
    checkpoints = [
        root / "notebooks" / "checkpoints" / "pinn_100d_poisson_shallow_300.pt",
        root / "notebooks" / "checkpoints" / "pinn_100d_poisson.pt",
    ]
    records = [_benchmark(path) for path in checkpoints]
    output = (
        root
        / "notebooks"
        / "benchmark_outputs"
        / "hybrid_graph_hilbert_norms.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
