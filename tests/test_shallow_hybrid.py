from __future__ import annotations

from math import sqrt

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from intervalnets import (
    IntervalTensor,
    PZIntegrationCell,
    PolynomialZonotope,
    integrate_pz_onejet_squared,
    integrate_shallow_hybrid_onejet_squared,
    shallow_scalar_hybrid_onejet_reverse,
)
from intervalnets.pytorch import pz_value_forward


def _model() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.2, -0.4], [0.7, 0.9], [-1.1, 0.3]]))
        model[0].bias.copy_(torch.tensor([0.0, 0.15, -0.05]))
        model[2].weight.copy_(torch.tensor([[0.8, -0.6, 0.5]]))
        model[2].bias.copy_(torch.tensor([0.1]))
    return model


def test_shallow_reverse_hybrid_is_uncompressed_and_sound() -> None:
    model = _model()
    lower = torch.tensor([-0.6, -0.5], dtype=torch.float64)
    upper = torch.tensor([0.6, 0.5], dtype=torch.float64)
    domain = PolynomialZonotope.from_box(lower, upper)
    result = shallow_scalar_hybrid_onejet_reverse(
        model,
        domain,
        derivative_flatness_threshold=1.0,
    )

    quadratic_count = int(torch.count_nonzero(result.derivative_degrees == 2))
    expected_domain_terms = 2 + (3 if quadratic_count else 0)
    assert len(result.domain_coefficients) == expected_domain_terms
    assert len(result.derivative_error_generators) == 3
    assert len(result.final.J.terms) == expected_domain_terms + 3

    samples = lower + (upper - lower) * torch.rand((4096, 2), generator=torch.Generator().manual_seed(7))
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    enclosure = result.final.J.interval_enclosure()
    enclosure_lower = torch.as_tensor(enclosure.lower).reshape(1, 2)
    enclosure_upper = torch.as_tensor(enclosure.upper).reshape(1, 2)
    assert torch.all(gradients >= enclosure_lower)
    assert torch.all(gradients <= enclosure_upper)


def test_shallow_reverse_reuses_preactivation_for_value_and_derivative() -> None:
    model = _model()
    lower = torch.tensor([-0.6, -0.5], dtype=torch.float64)
    upper = torch.tensor([0.6, 0.5], dtype=torch.float64)
    domain = PolynomialZonotope.from_box(lower, upper)
    result = shallow_scalar_hybrid_onejet_reverse(model, domain)

    input_coefficients = torch.stack(
        [domain.terms[exponent] for exponent in sorted(domain.terms)]
    )
    expected_center = model[0].weight @ domain.center + model[0].bias
    expected_coefficients = model[0].weight @ input_coefficients.T
    expected_radius = torch.sum(torch.abs(expected_coefficients), dim=1)
    assert torch.allclose(result.preactivation_center, expected_center)
    assert torch.allclose(result.preactivation_coefficients, expected_coefficients)
    assert torch.all(result.preactivation_lower <= expected_center - expected_radius)
    assert torch.all(result.preactivation_upper >= expected_center + expected_radius)

    # The specialized value construction uses that same prepared data and
    # reproduces the generic affine-tanh value enclosure.
    generic = pz_value_forward(model, domain).interval_enclosure()
    specialized = result.final.Y.interval_enclosure()
    assert torch.allclose(
        torch.as_tensor(specialized.lower), torch.as_tensor(generic.lower)
    )
    assert torch.allclose(
        torch.as_tensor(specialized.upper), torch.as_tensor(generic.upper)
    )
    assert set(result.timings) >= {
        "preactivation_preparation",
        "activation_certification",
        "value_construction",
        "reverse_jacobian_construction",
        "onejet_construction",
    }


def test_shallow_direct_integral_matches_generic_uncompressed_reference() -> None:
    model = _model()
    box = IntervalTensor.from_bounds([-0.25, -0.2], [0.25, 0.2])
    cell = PZIntegrationCell.from_affine_box(box)
    result = shallow_scalar_hybrid_onejet_reverse(
        model,
        cell.domain,
        derivative_flatness_threshold=1.0,
    )

    specialized = integrate_shallow_hybrid_onejet_squared(result, cell)
    reference = integrate_pz_onejet_squared(result.final, cell)
    assert float(specialized.lower) == pytest.approx(float(reference.lower), rel=2e-12, abs=2e-12)
    assert float(specialized.upper) == pytest.approx(float(reference.upper), rel=2e-12, abs=2e-12)

    # The exact sampled norm must be enclosed as a coarse additional check.
    generator = torch.Generator().manual_seed(11)
    samples = torch.tensor(box.lower, dtype=torch.float64) + (
        torch.tensor(box.upper, dtype=torch.float64)
        - torch.tensor(box.lower, dtype=torch.float64)
    ) * torch.rand((10000, 2), generator=generator)
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    sampled_squared = float(
        (values.square().squeeze(1) + gradients.square().sum(1)).mean().detach()
    )
    volume = float(cell.volume)
    assert float(specialized.lower) / volume <= sampled_squared
    assert sampled_squared <= float(specialized.upper) / volume
    assert sqrt(max(0.0, float(specialized.upper) / volume)) > 0.0
