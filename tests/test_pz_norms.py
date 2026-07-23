import pytest

from intervalnets import IntervalTensor, PZTwoJet, PolynomialZonotope, enable_interval_eval
from intervalnets.pz_integration import PZIntegrationCell, integrate_over_cell
from intervalnets.pz_norms import pz_sum_squares, pz_symmetric_hessian_sum_squares, pz_twojet_l2_integrand, pz_twojet_w22_integrand

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

pytestmark = pytest.mark.skipif(torch is None, reason="PyTorch not installed")


def _assert_same_pz(left: PolynomialZonotope, right: PolynomialZonotope):
    assert left.num_noise == right.num_noise
    assert left.noise_kinds == right.noise_kinds
    assert left.shape == right.shape
    if torch is not None and isinstance(left.center, torch.Tensor):
        assert torch.allclose(left.center, right.center)
        assert set(left.terms) == set(right.terms)
        for exponent in left.terms:
            assert torch.allclose(left.terms[exponent], right.terms[exponent])
    else:
        assert left.center == pytest.approx(right.center)
        assert left.terms == pytest.approx(right.terms)


def test_pz_symmetric_hessian_sum_squares_matches_dense_full_sum_for_symmetric_hessian():
    center = torch.tensor(
        [
            [[1.0, 2.0, -0.5], [2.0, -1.0, 0.75], [-0.5, 0.75, 1.5]],
            [[-0.25, 1.25, 0.5], [1.25, 0.5, -1.5], [0.5, -1.5, 2.0]],
        ],
        dtype=torch.float64,
    )
    coeff = torch.tensor(
        [
            [[0.2, -0.1, 0.3], [-0.1, 0.4, -0.2], [0.3, -0.2, 0.1]],
            [[-0.3, 0.2, 0.15], [0.2, -0.05, 0.35], [0.15, 0.35, -0.25]],
        ],
        dtype=torch.float64,
    )
    hessian = PolynomialZonotope(center, {(1,): coeff}, num_noise=1, noise_kinds=("domain",))

    optimized = pz_symmetric_hessian_sum_squares(hessian)
    dense = pz_sum_squares(hessian)

    _assert_same_pz(optimized, dense)


def test_pz_twojet_w22_integrand_uses_symmetric_hessian_accumulation_equivalent_to_dense_sum():
    y = PolynomialZonotope.constant(torch.tensor([0.5], dtype=torch.float64), num_noise=1, noise_kinds=("domain",))
    j = PolynomialZonotope.constant(torch.tensor([[1.0, -2.0]], dtype=torch.float64), num_noise=1, noise_kinds=("domain",))
    h_center = torch.tensor([[[1.0, 0.25], [0.25, -0.5]]], dtype=torch.float64)
    h_coeff = torch.tensor([[[0.1, -0.2], [-0.2, 0.3]]], dtype=torch.float64)
    h = PolynomialZonotope(h_center, {(1,): h_coeff}, num_noise=1, noise_kinds=("domain",))
    jet = PZTwoJet(y, j, h)

    optimized = pz_twojet_w22_integrand(jet)
    dense = pz_sum_squares(y) + pz_sum_squares(j) + pz_sum_squares(h)

    _assert_same_pz(optimized, dense)


def _small_tanh_model(input_dim=1, hidden_dim=2, output_dim=1):
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim, dtype=torch.float64),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim, dtype=torch.float64),
    ).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(-0.4, 0.5, steps=hidden_dim * input_dim, dtype=torch.float64).reshape(hidden_dim, input_dim))
        model[0].bias.copy_(torch.linspace(-0.1, 0.15, steps=hidden_dim, dtype=torch.float64))
        model[2].weight.copy_(torch.linspace(0.25, -0.35, steps=output_dim * hidden_dim, dtype=torch.float64).reshape(output_dim, hidden_dim))
        model[2].bias.copy_(torch.linspace(0.05, 0.1, steps=output_dim, dtype=torch.float64))
    return model


def test_pz_l2norm_zero_network_returns_zero_interval():
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(2, 2, dtype=torch.float64), nn.Tanh(), nn.Linear(2, 1, dtype=torch.float64)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.2, -0.1], [0.05, 0.15]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([0.1, -0.2], dtype=torch.float64))
        model[2].weight.zero_()
        model[2].bias.zero_()

    domain = IntervalTensor.from_bounds([-1.0, -2.0], [3.0, 4.0])
    bounds = model.lpnorm(domain, p=2.0, method="pz", iterations=1, chebyshev_degree=3, residual_subdivisions=16)

    assert bounds.lower <= 0.0 <= bounds.upper
    assert bounds.upper < 1e-10


def test_pz_l2norm_constant_network_matches_exact_value():
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1, dtype=torch.float64), nn.Tanh(), nn.Linear(1, 1, dtype=torch.float64)).double()
    with torch.no_grad():
        model[0].weight.fill_(0.2)
        model[0].bias.fill_(0.1)
        model[2].weight.zero_()
        model[2].bias.fill_(3.0)

    domain = IntervalTensor.from_bounds([0.0], [2.0])
    bounds = model.lpnorm(domain, p=2.0, method="pz", iterations=1, chebyshev_degree=3, residual_subdivisions=16)
    exact = 18.0**0.5

    assert bounds.lower <= exact <= bounds.upper
    assert bounds.upper - bounds.lower < 1e-10


def test_pz_l2norm_refinement_tightens_interval():
    enable_interval_eval()
    model = _small_tanh_model()
    domain = IntervalTensor.from_bounds([-1.0], [1.0])

    coarse = model.lpnorm(domain, p=2.0, method="pz", iterations=0, chebyshev_degree=3, residual_subdivisions=16)
    refined = model.lpnorm(domain, p=2.0, method="pz", iterations=2, chebyshev_degree=3, residual_subdivisions=16)

    assert refined.lower >= coarse.lower
    assert refined.upper <= coarse.upper
    assert refined.upper - refined.lower <= coarse.upper - coarse.lower


def test_pz_l2norm_accepts_dorfler_theta_parameter():
    enable_interval_eval()
    model = _small_tanh_model()
    domain = IntervalTensor.from_bounds([-1.0], [1.0])

    bounds = model.lpnorm(domain, p=2.0, method="pz", iterations=2, theta=0.5, chebyshev_degree=3, residual_subdivisions=16)

    assert bounds.lower <= bounds.upper


def test_pz_norms_validate_adaptive_and_forward_parameters():
    enable_interval_eval()
    model = _small_tanh_model()
    domain = IntervalTensor.from_bounds([0.0], [1.0])

    with pytest.raises(ValueError):
        model.lpnorm(domain, p=2.0, method="pz", iterations=-1, chebyshev_degree=3, residual_subdivisions=16)
    with pytest.raises(ValueError):
        model.lpnorm(domain, p=2.0, method="pz", iterations=1, theta=0.0, chebyshev_degree=3, residual_subdivisions=16)
    with pytest.raises(ValueError):
        model.lpnorm(domain, p=2.0, method="pz", iterations=0, chebyshev_degree=-1, residual_subdivisions=16)
    with pytest.raises(ValueError):
        model.lpnorm(domain, p=2.0, method="pz", iterations=0, chebyshev_degree=3, residual_subdivisions=0)
    with pytest.raises(ValueError):
        model.lpnorm(domain, p=2.0, iterations=0, forward_refine_splits=0)


def test_pz_l2norm_contains_monte_carlo_estimate():
    enable_interval_eval()
    torch.manual_seed(7)
    model = _small_tanh_model(input_dim=2, hidden_dim=2, output_dim=1)
    domain = IntervalTensor.from_bounds([-1.0, -0.5], [1.0, 1.5])

    bounds = model.lpnorm(domain, p=2.0, method="pz", iterations=1, chebyshev_degree=3, residual_subdivisions=24)

    sample_count = 2048
    with torch.no_grad():
        samples = torch.rand(sample_count, 2, dtype=torch.float64)
        samples[:, 0] = 2.0 * samples[:, 0] - 1.0
        samples[:, 1] = 2.0 * samples[:, 1] - 0.5
        values = model(samples).squeeze(-1)
        estimate = float((4.0 * torch.mean(values.abs() ** 2.0)).sqrt().item())

    assert bounds.lower <= estimate <= bounds.upper


def test_pz_w12_order_one_sobolev_behavior():
    enable_interval_eval()
    model = _small_tanh_model()
    domain = IntervalTensor.from_bounds([-0.7], [0.9])

    default_order = model.pz_sobolev_norm(domain, iterations=1, chebyshev_degree=3, residual_subdivisions=16)
    order_one = model.pz_sobolev_norm(domain, order=1, iterations=1, chebyshev_degree=3, residual_subdivisions=16)

    assert order_one.lower <= default_order.upper
    assert default_order.lower <= order_one.upper


def test_pz_w22_order_two_sobolev_behavior():
    enable_interval_eval()
    model = _small_tanh_model()
    domain = IntervalTensor.from_bounds([-0.7], [0.9])

    w12 = model.pz_sobolev_norm(domain, order=1, iterations=1, chebyshev_degree=3, residual_subdivisions=16)
    w22 = model.pz_sobolev_norm(domain, order=2, iterations=1, chebyshev_degree=3, residual_subdivisions=16)

    assert w22.lower >= w12.lower
    assert w22.upper >= w12.upper


def test_pz_domain_integration_of_odd_monomials_gives_zero():
    cell = PZIntegrationCell.from_bounds([-1.0], [1.0])
    x = cell.domain[0]

    result = integrate_over_cell(x + x * x * x, cell, output="pz")

    assert result.num_noise == 0
    assert result.center == pytest.approx(0.0)
    assert result.terms == {}


def test_pz_approximation_noise_remains_after_domain_integration():
    cell = PZIntegrationCell.from_bounds([-1.0], [1.0])
    expr = PolynomialZonotope(
        1.0,
        {(1, 0): 2.0, (0, 1): 0.25},
        num_noise=2,
        noise_kinds=("domain", "approximation_symbolic"),
    )

    result = integrate_over_cell(expr, cell, output="pz")

    assert result.noise_kinds == ("approximation_symbolic",)
    assert result.center == pytest.approx(2.0)
    assert result.terms[(1,)] == pytest.approx(0.5)


def test_pz_output_keeps_only_non_domain_noise_after_integrating_network_integrand():
    enable_interval_eval()
    model = _small_tanh_model()
    cell = PZIntegrationCell.from_bounds([-0.5], [0.5])
    jet = model.eval_pz_twojet(cell.domain, chebyshev_degree=3, residual_subdivisions=16)
    integrand = pz_twojet_l2_integrand(jet)

    result = integrate_over_cell(integrand, cell, output="pz")

    assert "domain" not in result.noise_kinds
    assert result.num_noise == len(result.noise_kinds)
    assert result.num_noise > 0


def test_pz_interval_output_intervalizes_retained_approximation_noise_after_integration():
    cell = PZIntegrationCell.from_bounds([-1.0], [1.0])
    expr = PolynomialZonotope(
        1.0,
        {(0, 1): 0.25},
        num_noise=2,
        noise_kinds=("domain", "approximation_symbolic"),
    )

    symbolic = integrate_over_cell(expr, cell, output="pz")
    interval = integrate_over_cell(expr, cell, output="interval")

    assert symbolic.noise_kinds == ("approximation_symbolic",)
    assert symbolic.terms[(1,)] == pytest.approx(0.5)
    assert interval.lower == pytest.approx(1.5)
    assert interval.upper == pytest.approx(2.5)
