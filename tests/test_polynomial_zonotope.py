import pytest

from intervalnets import PZTwoJet, PolynomialZonotope

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def test_from_box_interval_enclosure_fallback():
    z = PolynomialZonotope.from_box((-1.0, 2.0), (3.0, 4.0))
    assert z.num_noise == 2
    assert z.shape == (2,)
    enclosure = z.interval_enclosure()
    assert enclosure.lower[0] <= -1.0 <= enclosure.upper[0]
    assert enclosure.lower[1] <= 2.0 <= enclosure.upper[1]
    assert enclosure.lower[0] <= 3.0 <= enclosure.upper[0]
    assert enclosure.lower[1] <= 4.0 <= enclosure.upper[1]


def test_addition_merges_equal_exponents():
    z1 = PolynomialZonotope(1.0, {(1,): 2.0}, num_noise=1)
    z2 = PolynomialZonotope(3.0, {(1,): 4.0}, num_noise=1)
    out = z1 + z2
    assert out.center == 4.0
    assert out.terms[(1,)] == 6.0


def test_scalar_polynomial_multiplication_convolves_exponents():
    z = PolynomialZonotope(1.0, {(1,): 2.0}, num_noise=1)
    out = z * z
    assert out.center == 1.0
    assert out.terms[(1,)] == 4.0
    assert out.terms[(2,)] == 4.0


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_torch_scalar_times_vector_and_stack_and_tensor_product():
    scalar = PolynomialZonotope(torch.tensor(2.0), {(1,): torch.tensor(3.0)}, num_noise=1)
    vector = PolynomialZonotope.constant(torch.tensor([1.0, -1.0]), num_noise=1)
    product = scalar * vector
    assert product.shape == (2,)
    assert torch.allclose(product.center, torch.tensor([2.0, -2.0]))
    assert torch.allclose(product.terms[(1,)], torch.tensor([3.0, -3.0]))

    stacked = PolynomialZonotope.stack((product[0], product[1]))
    assert stacked.shape == (2,)
    assert torch.allclose(stacked.center, product.center)

    outer = vector.tensor_product(vector)
    assert outer.shape == (2, 2)
    assert torch.allclose(outer.center, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_torch_box_enclosure_contains_corners():
    lower = torch.tensor([-2.0, 1.0])
    upper = torch.tensor([4.0, 5.0])
    z = PolynomialZonotope.from_box(lower, upper)
    interval = z.interval_enclosure()
    lo, hi = interval.to_torch(dtype=torch.float64)
    assert torch.all(lo <= lower.double())
    assert torch.all(hi >= upper.double())


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_twojet_from_input_initializes_physical_input_derivatives():
    lower = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    upper = torch.tensor([3.0, 4.0], dtype=torch.float64)
    X = PolynomialZonotope.from_box(lower, upper)

    jet = PZTwoJet.from_input(X, input_dim=2)

    assert jet.Y is X
    assert jet.J.num_noise == X.num_noise
    assert jet.H.num_noise == X.num_noise
    assert jet.J.shape == (2, 2)
    assert jet.H.shape == (2, 2, 2)
    assert jet.J.terms == {}
    assert jet.H.terms == {}
    assert torch.allclose(jet.J.center, torch.eye(2, dtype=torch.float64))
    assert torch.allclose(jet.H.center, torch.zeros(2, 2, 2, dtype=torch.float64))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_linear_map_contracts_value_jacobian_and_hessian_shapes():
    center = torch.tensor([1.0, -2.0], dtype=torch.float64)
    coeff = torch.tensor([0.5, 1.5], dtype=torch.float64)
    z = PolynomialZonotope(center, {(1,): coeff}, num_noise=1)
    weight = torch.tensor([[2.0, -1.0], [0.0, 3.0], [1.0, 1.0]], dtype=torch.float64)
    bias = torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64)

    out = z.linear_map(weight, bias)

    assert out.shape == (3,)
    assert torch.allclose(out.center, weight.matmul(center) + bias)
    assert torch.allclose(out.terms[(1,)], weight.matmul(coeff))

    jac = PolynomialZonotope.constant(torch.arange(6, dtype=torch.float64).reshape(2, 3), num_noise=1)
    jac_out = jac.linear_map(weight, bias=None)
    assert jac_out.shape == (3, 3)
    assert torch.allclose(jac_out.center, torch.einsum("ij,jk->ik", weight, jac.center))

    hess = PolynomialZonotope.constant(torch.arange(18, dtype=torch.float64).reshape(2, 3, 3), num_noise=1)
    hess_out = hess.linear_map(weight, bias=None)
    assert hess_out.shape == (3, 3, 3)
    assert torch.allclose(hess_out.center, torch.einsum("ij,jkl->ikl", weight, hess.center))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_twojet_linear_forward_matches_layer_affine_map():
    from torch import nn
    from intervalnets.pytorch import _pz_twojet_linear_forward

    layer = nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 0.5], [3.0, -2.0]], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))

    X = PolynomialZonotope.from_box(torch.tensor([-1.0, 0.0], dtype=torch.float64), torch.tensor([1.0, 2.0], dtype=torch.float64))
    jet = PZTwoJet.from_input(X, input_dim=2)
    out = _pz_twojet_linear_forward(layer, jet)

    assert out.Y.shape == (3,)
    assert out.J.shape == (3, 2)
    assert out.H.shape == (3, 2, 2)
    assert torch.allclose(out.Y.center, layer.weight.detach().matmul(X.center) + layer.bias.detach())
    assert torch.allclose(out.J.center, layer.weight.detach())
    assert torch.allclose(out.H.center, torch.zeros(3, 2, 2, dtype=torch.float64))
    for exp, coeff in X.terms.items():
        assert torch.allclose(out.Y.terms[exp], layer.weight.detach().matmul(coeff))


def test_evaluate_polynomial_uses_power_basis_and_horner_dependencies():
    z = PolynomialZonotope(1.0, {(1,): 2.0}, num_noise=1)
    out = z.evaluate_polynomial((3.0, 4.0, 5.0))
    expected = 3.0 + 4.0 * z + 5.0 * z * z
    assert out.center == expected.center
    assert out.terms == expected.terms


def test_add_independent_error_extends_existing_exponents():
    z = PolynomialZonotope(1.0, {(1, 2): 3.0}, num_noise=2)
    out = z.add_independent_error(0.25)
    assert out.num_noise == 3
    assert out.center == 1.0
    assert out.terms[(1, 2, 0)] == 3.0
    assert out.terms[(0, 0, 1)] == 0.25


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_stack_aligns_sliced_scalars_and_merges_exponents():
    z1 = PolynomialZonotope(torch.tensor([1.0, 2.0], dtype=torch.float64), {(1,): torch.tensor([0.5, 1.5], dtype=torch.float64)}, num_noise=1)
    z2 = PolynomialZonotope(torch.tensor(3.0, dtype=torch.float64), {(0, 1): torch.tensor(2.0, dtype=torch.float64)}, num_noise=2)
    stacked = PolynomialZonotope.stack((z1[1], z2))
    assert stacked.num_noise == 2
    assert stacked.shape == (2,)
    assert torch.allclose(stacked.center, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(stacked.terms[(1, 0)], torch.tensor([1.5, 0.0], dtype=torch.float64))
    assert torch.allclose(stacked.terms[(0, 1)], torch.tensor([0.0, 2.0], dtype=torch.float64))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_tanh_pz_scalar_adds_certified_fresh_noise_and_encloses_samples():
    from intervalnets.pz_tanh import tanh_pz_scalar

    z = PolynomialZonotope.from_box(torch.tensor(-0.5, dtype=torch.float64), torch.tensor(0.75, dtype=torch.float64))
    out = tanh_pz_scalar(z, remez_degree=5, residual_subdivisions=64)
    assert out.shape == ()
    assert out.num_noise == z.num_noise + 1
    assert any(exp[-1] == 1 for exp in out.terms)
    enclosure = out.interval_enclosure()
    lo, hi = enclosure.to_torch(dtype=torch.float64)
    for value in torch.linspace(-0.5, 0.75, steps=9, dtype=torch.float64):
        expected = torch.tanh(value)
        assert lo <= expected <= hi


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_twojet_tanh_forward_preserves_shapes_and_encloses_autograd_samples():
    from intervalnets.pytorch import _pz_twojet_tanh_forward

    lower = torch.tensor([-0.4, 0.2], dtype=torch.float64)
    upper = torch.tensor([0.6, 0.8], dtype=torch.float64)
    X = PolynomialZonotope.from_box(lower, upper)
    jet = PZTwoJet.from_input(X, input_dim=2)

    out = _pz_twojet_tanh_forward(jet, remez_degree=5, residual_subdivisions=64)

    assert out.Y.shape == (2,)
    assert out.J.shape == (2, 2)
    assert out.H.shape == (2, 2, 2)
    assert out.Y.num_noise == X.num_noise + 2
    assert out.J.num_noise == out.Y.num_noise
    assert out.H.num_noise == out.Y.num_noise

    y_lo, y_hi = out.Y.interval_enclosure().to_torch(dtype=torch.float64)
    j_lo, j_hi = out.J.interval_enclosure().to_torch(dtype=torch.float64)
    h_lo, h_hi = out.H.interval_enclosure().to_torch(dtype=torch.float64)

    for x0 in torch.linspace(float(lower[0]), float(upper[0]), steps=5, dtype=torch.float64):
        for x1 in torch.linspace(float(lower[1]), float(upper[1]), steps=5, dtype=torch.float64):
            point = torch.stack((x0, x1)).requires_grad_(True)
            value = torch.tanh(point)
            rows = []
            hessians = []
            for i in range(2):
                grad = torch.autograd.grad(value[i], point, create_graph=True, retain_graph=True)[0]
                rows.append(grad)
                h_rows = []
                for j in range(2):
                    h_rows.append(torch.autograd.grad(grad[j], point, retain_graph=True)[0])
                hessians.append(torch.stack(h_rows))
            jac = torch.stack(rows)
            hess = torch.stack(hessians)
            assert torch.all(y_lo <= value.detach()) and torch.all(value.detach() <= y_hi)
            assert torch.all(j_lo <= jac.detach()) and torch.all(jac.detach() <= j_hi)
            assert torch.all(h_lo <= hess.detach()) and torch.all(hess.detach() <= h_hi)


def test_noise_metadata_defaults_and_validation():
    z = PolynomialZonotope(0.0, {(1, 0): 2.0}, num_noise=2)
    assert z.noise_kinds == ("unknown", "unknown")
    with pytest.raises(ValueError, match="noise_kinds length"):
        PolynomialZonotope(0.0, {(1,): 2.0}, num_noise=1, noise_kinds=("domain", "extra"))


def test_domain_noise_stays_first_when_approximation_error_is_appended_fallback():
    z = PolynomialZonotope.from_box((-1.0, 2.0), (3.0, 4.0))
    out = z.add_independent_error(0.25)
    assert out.noise_kinds == ("domain", "domain", "approximation")
    assert out.terms[(1, 0, 0)] == z.terms[(1, 0)]
    assert out.terms[(0, 1, 0)] == z.terms[(0, 1)]
    assert out.terms[(0, 0, 1)] == (0.25, 0.25)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_tanh_residual_noise_is_appended_and_labeled_after_domain_noise():
    from intervalnets.pz_tanh import tanh_pz_scalar

    z = PolynomialZonotope.from_box(torch.tensor(-0.5, dtype=torch.float64), torch.tensor(0.75, dtype=torch.float64))
    out = tanh_pz_scalar(z, remez_degree=5, residual_subdivisions=64)

    assert z.noise_kinds == ("domain",)
    assert out.noise_kinds == ("domain", "approximation")
    assert any(exp == (0, 1) for exp in out.terms)
