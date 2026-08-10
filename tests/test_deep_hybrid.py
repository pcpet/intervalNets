from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from intervalnets import (
    DeepHybridOneJetResult,
    IntervalTensor,
    PZIntegrationCell,
    PolynomialZonotope,
    integrate_deep_hybrid_onejet_squared,
    integrate_deep_hybrid_value_squared,
    integrate_hybrid_onejet_squared,
    integrate_hybrid_value_squared,
    integrate_pz_value_squared,
    scalar_hybrid_onejet_reverse,
    shallow_scalar_hybrid_onejet_reverse,
)


def _deep_model() -> nn.Sequential:
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.Tanh(),
        nn.Linear(3, 2),
        nn.Tanh(),
        nn.Linear(2, 1),
    ).double()
    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor([[1.1, -0.3], [0.5, 0.8], [-0.9, 0.4]])
        )
        model[0].bias.copy_(torch.tensor([0.0, 0.1, -0.15]))
        model[2].weight.copy_(
            torch.tensor([[0.7, -0.4, 0.5], [-0.2, 0.9, 0.6]])
        )
        model[2].bias.copy_(torch.tensor([0.05, -0.08]))
        model[4].weight.copy_(torch.tensor([[0.8, -0.55]]))
        model[4].bias.copy_(torch.tensor([0.12]))
    return model


def _realizing_noises(result: DeepHybridOneJetResult, samples: torch.Tensor):
    value_noises = [samples / 0.35]
    derivative_noises = []
    for factor in result.factors:
        xi = torch.cat(value_noises, dim=1)
        preactivation = (
            factor.preactivation_center.unsqueeze(0)
            + xi @ factor.preactivation_coefficients.T
        )
        exact_value = torch.tanh(preactivation)
        value_core = (
            factor.value_intercepts.unsqueeze(0)
            + factor.value_slopes.unsqueeze(0) * preactivation
        )
        value_eta = torch.where(
            factor.value_approximation_radii.unsqueeze(0) == 0.0,
            torch.zeros_like(exact_value),
            (exact_value - value_core)
            / factor.value_approximation_radii.unsqueeze(0),
        )
        affine_argument = xi @ factor.preactivation_coefficients.T
        derivative_core = (
            factor.center.unsqueeze(0)
            + xi @ factor.linear_coefficients.T
            + factor.quadratic_coefficients.unsqueeze(0)
            * affine_argument.square()
        )
        exact_derivative = 1.0 - exact_value.square()
        derivative_eta = torch.where(
            factor.approximation_radii.unsqueeze(0) == 0.0,
            torch.zeros_like(exact_derivative),
            (exact_derivative - derivative_core)
            / factor.approximation_radii.unsqueeze(0),
        )
        value_noises.append(value_eta)
        derivative_noises.append(derivative_eta)
    return torch.cat((*value_noises, *derivative_noises), dim=1)


def test_deep_factored_hybrid_retains_all_factors_and_is_sound() -> None:
    model = _deep_model()
    lower = torch.full((2,), -0.35, dtype=torch.float64)
    upper = torch.full((2,), 0.35, dtype=torch.float64)
    domain = PolynomialZonotope.from_box(lower, upper)
    result = scalar_hybrid_onejet_reverse(
        model, domain, derivative_flatness_threshold=1.0
    )
    assert isinstance(result, DeepHybridOneJetResult)
    assert len(result.factors) == 2
    assert result.jacobian.num_noise == 2 + 2 * (3 + 2)
    assert sum(factor.center.numel() for factor in result.factors) == 5

    generator = torch.Generator().manual_seed(31)
    samples = lower + (upper - lower) * torch.rand((2048, 2), generator=generator)
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    noises = _realizing_noises(result, samples.detach())
    assert torch.max(torch.abs(noises)) <= 1.0 + 5e-12
    factored_gradients = result.jacobian.evaluate(noises)
    assert torch.allclose(factored_gradients, gradients, rtol=2e-11, atol=2e-11)

    enclosure = result.jacobian.interval_enclosure()
    enclosure_lower = torch.as_tensor(enclosure.lower).reshape(1, 2)
    enclosure_upper = torch.as_tensor(enclosure.upper).reshape(1, 2)
    assert torch.all(gradients >= enclosure_lower)
    assert torch.all(gradients <= enclosure_upper)


def test_deep_factored_integral_encloses_sampled_w12() -> None:
    model = _deep_model()
    box = IntervalTensor.from_bounds([-0.35, -0.35], [0.35, 0.35])
    cell = PZIntegrationCell.from_affine_box(box)
    result = scalar_hybrid_onejet_reverse(
        model, cell.domain, derivative_flatness_threshold=1.0
    )
    integral = integrate_deep_hybrid_onejet_squared(result, cell)

    generator = torch.Generator().manual_seed(37)
    lower = torch.tensor(box.lower, dtype=torch.float64)
    upper = torch.tensor(box.upper, dtype=torch.float64)
    samples = lower + (upper - lower) * torch.rand((12000, 2), generator=generator)
    samples.requires_grad_(True)
    values = model(samples)
    gradients = torch.autograd.grad(values.sum(), samples)[0]
    sampled = float(
        (values.square().squeeze(1) + gradients.square().sum(1)).mean().detach()
    )
    normalized_lower = float(integral.lower) / float(cell.volume)
    normalized_upper = float(integral.upper) / float(cell.volume)
    assert normalized_lower <= sampled <= normalized_upper
    assert result.gradient_spectral_bound > 0.0

    value_integral = integrate_deep_hybrid_value_squared(result, cell)
    explicit_value_integral = integrate_pz_value_squared(result.value, cell)
    assert float(value_integral.lower) == pytest.approx(
        float(explicit_value_integral.lower), rel=2e-12, abs=2e-12
    )
    assert float(value_integral.upper) == pytest.approx(
        float(explicit_value_integral.upper), rel=2e-12, abs=2e-12
    )


def test_depth_generic_dispatch_recovers_shallow_implementation() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1)).double()
    torch.manual_seed(41)
    domain = PolynomialZonotope.from_box(
        torch.full((2,), -0.2, dtype=torch.float64),
        torch.full((2,), 0.2, dtype=torch.float64),
    )
    direct = shallow_scalar_hybrid_onejet_reverse(
        model, domain, derivative_flatness_threshold=1.0
    )
    dispatched = scalar_hybrid_onejet_reverse(
        model, domain, derivative_flatness_threshold=1.0
    )
    assert type(dispatched) is type(direct)
    assert torch.equal(dispatched.final.Y.center, direct.final.Y.center)
    assert torch.equal(dispatched.final.J.center, direct.final.J.center)
    cell = PZIntegrationCell.from_affine_box(
        IntervalTensor.from_bounds([-0.2, -0.2], [0.2, 0.2])
    )
    expected = integrate_hybrid_onejet_squared(direct, cell)
    actual = integrate_hybrid_onejet_squared(dispatched, cell)
    assert float(actual.lower) == float(expected.lower)
    assert float(actual.upper) == float(expected.upper)
    expected_l2 = integrate_hybrid_value_squared(direct, cell)
    actual_l2 = integrate_hybrid_value_squared(dispatched, cell)
    assert float(actual_l2.lower) == float(expected_l2.lower)
    assert float(actual_l2.upper) == float(expected_l2.upper)
