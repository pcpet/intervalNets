import pytest

from intervalnets import PolynomialZonotope

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
