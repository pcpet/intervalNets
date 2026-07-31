from dataclasses import replace

import pytest

from intervalnets import PZOneJet, PZTwoJet, PolynomialZonotope
from intervalnets.pz_integration import (
    IntegratedPZResult,
    POINTWISE_RESIDUAL_KINDS,
    PZIntegrationCell,
    integrate_over_cell,
    integrate_pz_over_domain,
    integrate_pz_onejet_squared,
    integrate_pz_twojet_squared,
)
from intervalnets.pz_tanh import (
    affine_tanh_double_prime_enclosure,
    affine_tanh_enclosure,
    affine_tanh_prime_enclosure,
    tanh_pz_scalar,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_direct_onejet_square_uses_exact_pointwise_jacobian_box_range():
    cell = PZIntegrationCell.from_bounds([-1.0], [1.0])
    value = PolynomialZonotope.constant(
        torch.tensor([1.0], dtype=torch.float64),
        num_noise=1,
        noise_kinds=("domain",),
    )
    jacobian = PolynomialZonotope.constant(
        torch.tensor([[2.0]], dtype=torch.float64),
        num_noise=1,
        noise_kinds=("domain",),
    ).add_independent_errors(
        torch.tensor([[0.5]], dtype=torch.float64),
        kind="approximation_pointwise",
    )
    value = value.with_num_noise(jacobian.num_noise).with_noise_kinds(
        jacobian.noise_kinds
    )

    result = integrate_pz_onejet_squared(PZOneJet(value, jacobian), cell)

    assert result.lower == pytest.approx(6.5)
    assert result.upper == pytest.approx(14.5)


def _assert_interval_close(left, right):
    assert float(left.lower) == pytest.approx(float(right.lower), rel=1e-12, abs=1e-12)
    assert float(left.upper) == pytest.approx(float(right.upper), rel=1e-12, abs=1e-12)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.parametrize("kind", ["l2", "w12", "w22"])
def test_direct_twojet_square_matches_explicit_with_unequal_tensor_supports(kind):
    from intervalnets.pz_norms import pz_twojet_l2_integrand, pz_twojet_w12_integrand, pz_twojet_w22_integrand

    torch.manual_seed(19)
    kinds = ("domain", "approximation_symbolic", "approximation_pointwise")
    y = PolynomialZonotope(torch.randn(2, dtype=torch.float64), {(1, 0, 0): torch.randn(2, dtype=torch.float64)}, num_noise=3, noise_kinds=kinds)
    j = PolynomialZonotope(torch.randn(2, 2, dtype=torch.float64), {(0, 1, 0): torch.randn(2, 2, dtype=torch.float64), (1, 0, 1): torch.randn(2, 2, dtype=torch.float64)}, num_noise=3, noise_kinds=kinds)
    h = PolynomialZonotope(torch.randn(2, 2, 2, dtype=torch.float64), {(2, 0, 0): torch.randn(2, 2, 2, dtype=torch.float64), (0, 0, 1): torch.randn(2, 2, 2, dtype=torch.float64)}, num_noise=3, noise_kinds=kinds)
    jet = PZTwoJet(y, j, h)
    cell = PZIntegrationCell.from_bounds((-2.0,), (2.0,))
    constructors = {"l2": pz_twojet_l2_integrand, "w12": pz_twojet_w12_integrand, "w22": pz_twojet_w22_integrand}

    direct = integrate_pz_twojet_squared(jet, cell, kind)
    explicit = integrate_over_cell(constructors[kind](jet), cell, output="interval")

    _assert_interval_close(direct, explicit)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_direct_twojet_square_canonicalizes_pointwise_cancellation_and_keeps_odd_domain_terms():
    kinds = ("domain", "approximation_pointwise")
    zero_j = PolynomialZonotope.constant(torch.zeros((2, 1), dtype=torch.float64), num_noise=2, noise_kinds=kinds)
    zero_h = PolynomialZonotope.constant(torch.zeros((2, 1, 1), dtype=torch.float64), num_noise=2, noise_kinds=kinds)
    cell = PZIntegrationCell.from_bounds((-1.0,), (1.0,))

    cancelling_y = PolynomialZonotope(torch.tensor([1.0, 1.0], dtype=torch.float64), {(0, 1): torch.tensor([1.0, -1.0], dtype=torch.float64)}, num_noise=2, noise_kinds=kinds)
    cancelling = integrate_pz_twojet_squared(PZTwoJet(cancelling_y, zero_j, zero_h), cell, "l2")
    assert float(cancelling.lower) == pytest.approx(0.0, abs=1e-14)
    assert float(cancelling.upper) == pytest.approx(8.0)

    odd_y = PolynomialZonotope(torch.tensor([0.0], dtype=torch.float64), {(1, 0): torch.tensor([1.0], dtype=torch.float64), (0, 1): torch.tensor([1.0], dtype=torch.float64)}, num_noise=2, noise_kinds=kinds)
    odd_jet = PZTwoJet(odd_y, zero_j[:1], zero_h[:1])
    direct = integrate_pz_twojet_squared(odd_jet, cell, "l2")
    from intervalnets.pz_norms import pz_twojet_l2_integrand
    explicit = integrate_over_cell(pz_twojet_l2_integrand(odd_jet), cell, output="interval")
    _assert_interval_close(direct, explicit)
    assert float(direct.lower) < -5.0  # The odd alpha*eta term receives full measure.


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_direct_twojet_square_falls_back_for_polynomial_density():
    kinds = ("domain",)
    cell = PZIntegrationCell.from_bounds((-1.0,), (1.0,))
    cell = replace(cell, jacobian_density=PolynomialZonotope.constant(1.0, num_noise=1, noise_kinds=kinds))
    jet = PZTwoJet(
        PolynomialZonotope(torch.tensor([1.0], dtype=torch.float64), {(1,): torch.tensor([0.5], dtype=torch.float64)}, num_noise=1, noise_kinds=kinds),
        PolynomialZonotope.constant(torch.zeros((1, 1), dtype=torch.float64), num_noise=1, noise_kinds=kinds),
        PolynomialZonotope.constant(torch.zeros((1, 1, 1), dtype=torch.float64), num_noise=1, noise_kinds=kinds),
    )
    from intervalnets.pz_norms import pz_twojet_l2_integrand

    direct = integrate_pz_twojet_squared(jet, cell, "l2")
    explicit = integrate_over_cell(pz_twojet_l2_integrand(jet), cell, output="interval")
    _assert_interval_close(direct, explicit)


def test_direct_twojet_square_vectorizes_float_coefficients(monkeypatch):
    import intervalnets.pz_integration as pz_integration

    if pz_integration.np is None:
        pytest.skip("NumPy not installed")
    kinds = ("domain", "approximation_pointwise")
    y = PolynomialZonotope(
        (1.0, -0.5),
        {
            (1, 0): (0.25, 0.75),
            (0, 1): (-0.1, 0.2),
        },
        num_noise=2,
        noise_kinds=kinds,
    )
    zero_j = PolynomialZonotope.constant(((0.0,), (0.0,)), num_noise=2, noise_kinds=kinds)
    zero_h = PolynomialZonotope.constant((((0.0,),), ((0.0,),)), num_noise=2, noise_kinds=kinds)
    jet = PZTwoJet(y, zero_j, zero_h)
    cell = PZIntegrationCell.from_bounds((-1.0,), (1.0,))

    from intervalnets.pz_norms import pz_twojet_l2_integrand

    explicit = integrate_over_cell(pz_twojet_l2_integrand(jet), cell, output="interval")

    def fail_python_dot(*args, **kwargs):
        raise AssertionError("float coefficients should use the vectorized NumPy path")

    monkeypatch.setattr(pz_integration, "_weighted_dot", fail_python_dot)
    direct = integrate_pz_twojet_squared(jet, cell, "l2")

    _assert_interval_close(direct, explicit)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_direct_value_squared_affine_fast_path_matches_explicit_reference():
    from intervalnets.pz_integration import integrate_pz_value_squared
    from intervalnets.pz_norms import pz_sum_squares

    cell = PZIntegrationCell.from_bounds([-1.0, -0.5], [1.0, 0.5])
    value = PolynomialZonotope(
        torch.tensor([0.2, -0.1], dtype=torch.float64),
        {
            (1, 0, 0, 0): torch.tensor([0.4, -0.2], dtype=torch.float64),
            (0, 1, 0, 0): torch.tensor([0.3, 0.1], dtype=torch.float64),
            (0, 0, 1, 0): torch.tensor([0.05, -0.07], dtype=torch.float64),
            (0, 0, 0, 1): torch.tensor([-0.02, 0.08], dtype=torch.float64),
        },
        num_noise=4,
        noise_kinds=(
            "domain",
            "domain",
            "approximation_pointwise",
            "approximation_symbolic",
        ),
    )

    direct = integrate_pz_value_squared(value, cell)
    explicit = integrate_over_cell(pz_sum_squares(value), cell, output="interval")

    _assert_interval_close(direct, explicit)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_adaptive_squared_contribution_avoids_explicit_integrand(monkeypatch):
    import intervalnets.pz_integration as pz_integration
    from intervalnets import IntervalTensor, enable_interval_eval

    enable_interval_eval()
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 2, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(2, 1, dtype=torch.float64),
    )
    domain = IntervalTensor.from_bounds([-0.5], [0.5])

    def fail_explicit_integrand(*args, **kwargs):
        raise AssertionError("adaptive affine cells must use direct squared integration")

    def fail_python_dot(*args, **kwargs):
        raise AssertionError("adaptive scalar coefficients should use a vectorized path")

    monkeypatch.setattr(pz_integration, "_squared_twojet_integrand", fail_explicit_integrand)
    monkeypatch.setattr(pz_integration, "_weighted_dot", fail_python_dot)
    cached = pz_integration._evaluate_squared_contribution_cache(
        model,
        domain,
        integrand_kind="w22",
        chebyshev_degree=3,
        residual_subdivisions=16,
    )

    assert cached.contribution.lower <= cached.contribution.upper


def test_integrating_pointwise_residual_adds_radius_not_symbolic_moment():
    z = PolynomialZonotope(
        1.0,
        {
            (1, 0): 3.0,  # odd domain monomial integrates to zero
            (2, 0): 6.0,  # exact contribution: 6 * int alpha^2 = 4
            (0, 1): 0.25,  # pointwise residual contribution: 0.25 * volume 2
        },
        num_noise=2,
        noise_kinds=("domain", "pointwise_residual"),
    )

    result = integrate_pz_over_domain(z)

    assert isinstance(result, IntegratedPZResult)
    assert result.measure == 2.0
    assert result.polynomial.num_noise == 1
    assert result.polynomial.noise_kinds == ("pointwise_residual",)
    assert result.polynomial.center == pytest.approx(6.0)  # 2 * center + 6 * 2/3
    assert result.polynomial.terms == {}
    assert result.interval_radius == pytest.approx(0.5)


def test_geometric_volume_scales_pointwise_residual_radius():
    z = PolynomialZonotope(
        0.0,
        {(0, 1): 2.0},
        num_noise=2,
        noise_kinds=("domain", "approximation_pointwise"),
    )

    result = integrate_pz_over_domain(z, volume=7.5)

    assert result.polynomial.center == 0.0
    assert result.polynomial.terms == {}
    assert result.interval_radius == pytest.approx(15.0)
    assert result.measure == 7.5


def test_symbolic_mode_keeps_residual_symbol_and_integrates_by_moments_only_when_requested():
    z = PolynomialZonotope(
        0.0,
        {(0, 1): 0.25},
        num_noise=2,
        noise_kinds=("domain", "pointwise_residual"),
    )

    result = integrate_pz_over_domain(z, mode="symbolic")

    assert result.interval_radius == 0.0
    assert result.polynomial.center == 0.0
    assert result.polynomial.terms[(1,)] == pytest.approx(0.5)
    assert result.polynomial.noise_kinds == ("pointwise_residual",)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_tanh_pz_scalar_marks_default_residual_as_pointwise():
    z = PolynomialZonotope(torch.tensor(0.1, dtype=torch.float64), {(1,): torch.tensor(0.05, dtype=torch.float64)}, num_noise=1, noise_kinds=("domain",))

    out = tanh_pz_scalar(z, chebyshev_degree=3, residual_subdivisions=16)

    assert out.noise_kinds[-1] == "approximation_pointwise"


def test_affine_tanh_residuals_integrate_as_pointwise_interval_radius():
    from intervalnets.pytorch import _affine_enclosure_pz

    cell = PZIntegrationCell.from_bounds((-1.0,), (1.0,))
    x = cell.domain[0]
    helpers = (
        affine_tanh_enclosure,
        affine_tanh_prime_enclosure,
        affine_tanh_double_prime_enclosure,
    )

    for helper in helpers:
        enclosure = helper((-1.0, 1.0))
        residual = _affine_enclosure_pz(
            x, slope=enclosure.p, intercept=enclosure.q, radius=enclosure.delta
        ) - (enclosure.p * x + enclosure.q)

        assert residual.noise_kinds[-1] in POINTWISE_RESIDUAL_KINDS

        result = integrate_over_cell(x * residual, cell, output="interval")

        # If the residual noise were treated as an ordinary symbolic monomial,
        # the odd domain factor would integrate to zero. Pointwise residual
        # handling instead accumulates it as interval radius.
        assert result.lower < 0.0
        assert result.upper > 0.0
        assert max(abs(float(result.lower)), abs(float(result.upper))) == pytest.approx(
            2.0 * enclosure.delta
        )

def test_affine_cell_integrates_one_dimensional_polynomial_exactly():
    cell = PZIntegrationCell.from_bounds((1.0,), (3.0,))
    x = cell.domain[0]
    expr = x * x

    result = integrate_over_cell(expr, cell, output="pz")

    assert cell.domain.noise_kinds == ("domain",)
    assert cell.domain.center == (2.0,)
    assert cell.domain.terms[(1,)] == (1.0,)
    assert cell.jacobian_density == pytest.approx(1.0)
    assert cell.volume == pytest.approx(2.0)
    assert result.num_noise == 0
    assert result.center == pytest.approx(26.0 / 3.0)
    assert result.terms == {}


def test_affine_cell_integrates_two_dimensional_polynomial_exactly():
    cell = PZIntegrationCell.from_bounds((1.0, -2.0), (3.0, 4.0))
    x = cell.domain[0]
    y = cell.domain[1]
    expr = x + 2.0 * y * y

    result = integrate_over_cell(expr, cell, output="pz")

    # integral over [1,3]x[-2,4] of x + 2 y^2 dxdy
    expected = 24.0 + 96.0
    assert cell.jacobian_density == pytest.approx(3.0)
    assert cell.volume == pytest.approx(12.0)
    assert result.center == pytest.approx(expected)
    assert result.terms == {}


def test_integrate_over_cell_interval_combines_symbolic_and_pointwise_residuals():
    cell = PZIntegrationCell.from_bounds((0.0,), (2.0,))
    x = cell.domain[0]
    expr = x + PolynomialZonotope(
        0.0,
        {(0, 1): 0.5, (0, 0, 1): 0.25},
        num_noise=3,
        noise_kinds=("domain", "approximation_symbolic", "approximation_pointwise"),
    )

    result = integrate_over_cell(expr, cell, output="interval")

    assert result.lower == pytest.approx(0.5)
    assert result.upper == pytest.approx(3.5)


def test_non_affine_fixed_orientation_hook_requires_certificates():
    cell = PZIntegrationCell.from_bounds((0.0,), (1.0,))

    with pytest.raises(NotImplementedError):
        PZIntegrationCell.from_fixed_orientation_domain(cell.domain, cell.domain_noise_indices)

@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_sum_squares_vector_matrix_tensor_entries():
    from intervalnets.pz_norms import pz_sum_squares

    z = PolynomialZonotope(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        {(1,): torch.ones(2, 2, dtype=torch.float64)},
        num_noise=1,
        noise_kinds=("domain",),
    )

    out = pz_sum_squares(z)

    assert out.shape == ()
    assert torch.allclose(out.center, torch.tensor(30.0, dtype=torch.float64))
    assert torch.allclose(out.terms[(1,)], torch.tensor(20.0, dtype=torch.float64))
    assert torch.allclose(out.terms[(2,)], torch.tensor(4.0, dtype=torch.float64))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_twojet_norms_constant_and_affine_match_closed_forms():
    from intervalnets import IntervalTensor, enable_interval_eval

    enable_interval_eval()
    const_model = torch.nn.Linear(1, 2, dtype=torch.float64)
    with torch.no_grad():
        const_model.weight.zero_()
        const_model.bias.copy_(torch.tensor([3.0, -4.0], dtype=torch.float64))
    domain = IntervalTensor.from_bounds([0.0], [2.0])

    const_norm = const_model.pz_l2norm(domain, p=2.0)

    assert const_norm.lower == pytest.approx((50.0) ** 0.5)
    assert const_norm.upper == pytest.approx((50.0) ** 0.5)

    affine_model = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        affine_model.weight.fill_(1.0)
        affine_model.bias.zero_()
    affine_domain = IntervalTensor.from_bounds([-1.0], [1.0])

    l2 = affine_model.pz_l2norm(affine_domain)
    w12 = affine_model.pz_sobolev_norm(affine_domain, order=1)
    w22 = affine_model.pz_sobolev_norm(affine_domain, order=2)

    assert l2.lower == pytest.approx((2.0 / 3.0) ** 0.5)
    assert l2.upper == pytest.approx((2.0 / 3.0) ** 0.5)
    assert w12.lower == pytest.approx((8.0 / 3.0) ** 0.5)
    assert w12.upper == pytest.approx((8.0 / 3.0) ** 0.5)
    assert w22.lower == pytest.approx(w12.lower)
    assert w22.upper == pytest.approx(w12.upper)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_norm_public_methods_reject_non_l2_p():
    from intervalnets import IntervalTensor, enable_interval_eval

    enable_interval_eval()
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    domain = IntervalTensor.from_bounds([-1.0], [1.0])

    with pytest.raises(NotImplementedError, match="p=2.0"):
        model.pz_l2norm(domain, p=1.0)
    with pytest.raises(NotImplementedError, match="p=2.0"):
        model.pz_sobolev_norm(domain, p=3.0, order=1)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_tanh_w22_norm_contains_dense_autograd_quadrature():
    from intervalnets import IntervalTensor, enable_interval_eval

    enable_interval_eval()
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, dtype=torch.float64), torch.nn.Tanh()).double()
    with torch.no_grad():
        model[0].weight.fill_(0.7)
        model[0].bias.fill_(0.1)
    domain = IntervalTensor.from_bounds([-0.5], [0.5])

    bounds = model.pz_sobolev_norm(domain, order=2, chebyshev_degree=5, residual_subdivisions=96)
    xs = torch.linspace(-0.5, 0.5, steps=401, dtype=torch.float64)
    values = []
    for x_value in xs:
        x = x_value.reshape(1).clone().detach().requires_grad_(True)
        y = model(x)[0]
        grad = torch.autograd.grad(y, x, create_graph=True)[0][0]
        hess = torch.autograd.grad(grad, x)[0][0]
        values.append((y.detach() ** 2 + grad.detach() ** 2 + hess.detach() ** 2).reshape(()))
    dense_integral = torch.trapezoid(torch.stack(values), xs).item()
    dense_norm = dense_integral ** 0.5

    assert bounds.lower <= dense_norm <= bounds.upper
