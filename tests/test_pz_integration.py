import pytest

from intervalnets import PolynomialZonotope
from intervalnets.pz_integration import IntegratedPZResult, integrate_pz_over_domain
from intervalnets.pz_tanh import tanh_pz_scalar

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


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

    out = tanh_pz_scalar(z, remez_degree=3, residual_subdivisions=16)

    assert out.noise_kinds[-1] == "approximation_pointwise"
