import math

import pytest

from intervalnets import PolynomialZonotope, enable_interval_eval, pz_twojet_forward
from intervalnets.pz_tanh import certify_tanh_residual_subdivision, compute_tanh_polynomial

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _poly(coeffs, x):
    acc = 0.0
    for coeff in reversed(coeffs):
        acc = acc * x + float(coeff)
    return acc


def _eval_pz(z: PolynomialZonotope, eps):
    if torch is None:
        raise ImportError("PyTorch is required for this test helper.")
    value = z.center.clone() if isinstance(z.center, torch.Tensor) else torch.tensor(z.center, dtype=torch.float64)
    eps = torch.as_tensor(eps, dtype=value.dtype, device=value.device)
    for exp, coeff in z.terms.items():
        monomial = torch.ones((), dtype=value.dtype, device=value.device)
        for idx, power in enumerate(exp):
            if power:
                monomial = monomial * eps[idx].pow(power)
        value = value + coeff * monomial
    return value


def _assert_contains(interval, sample, atol=1e-10):
    lo, hi = interval.to_torch(dtype=torch.float64)
    sample = sample.detach().to(dtype=torch.float64)
    assert torch.all(sample >= lo - atol), f"sample below lower bound: {sample} < {lo}"
    assert torch.all(sample <= hi + atol), f"sample above upper bound: {sample} > {hi}"


def test_arithmetic_merges_equal_exponents_and_scales_fallback_coefficients():
    z1 = PolynomialZonotope(1.0, {(1,): 2.0}, num_noise=1)
    z2 = PolynomialZonotope(3.0, {(1,): -5.0}, num_noise=1)

    summed = z1 + z2
    scaled = -2.5 * z1
    product = z1 * z2

    assert summed.center == 4.0
    assert summed.terms[(1,)] == -3.0
    assert scaled.center == -2.5
    assert scaled.terms[(1,)] == -5.0
    assert product.center == 3.0
    assert product.terms[(1,)] == 1.0
    assert product.terms[(2,)] == -10.0


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_arithmetic_scalar_times_vector_matrix_tensor_pz_and_outer_products():
    scalar = PolynomialZonotope(torch.tensor(2.0, dtype=torch.float64), {(1,): torch.tensor(-0.5, dtype=torch.float64)}, num_noise=1)
    vector = PolynomialZonotope.constant(torch.tensor([1.0, -3.0], dtype=torch.float64), num_noise=1)
    matrix = PolynomialZonotope.constant(torch.arange(1.0, 5.0, dtype=torch.float64).reshape(2, 2), num_noise=1)
    tensor = PolynomialZonotope.constant(torch.arange(1.0, 9.0, dtype=torch.float64).reshape(2, 2, 2), num_noise=1)

    vector_product = scalar * vector
    matrix_product = scalar * matrix
    tensor_product = scalar * tensor

    assert vector_product.shape == (2,)
    assert matrix_product.shape == (2, 2)
    assert tensor_product.shape == (2, 2, 2)
    assert torch.allclose(vector_product.center, torch.tensor([2.0, -6.0], dtype=torch.float64))
    assert torch.allclose(vector_product.terms[(1,)], torch.tensor([-0.5, 1.5], dtype=torch.float64))
    assert torch.allclose(matrix_product.center, 2.0 * matrix.center)
    assert torch.allclose(matrix_product.terms[(1,)], -0.5 * matrix.center)
    assert torch.allclose(tensor_product.center, 2.0 * tensor.center)
    assert torch.allclose(tensor_product.terms[(1,)], -0.5 * tensor.center)

    outer = vector.tensor_product(vector)
    assert outer.shape == (2, 2)
    assert torch.allclose(outer.center, torch.tensor([[1.0, -3.0], [-3.0, 9.0]], dtype=torch.float64))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_interval_enclosure_contains_random_noise_samples():
    torch.manual_seed(0)
    z = PolynomialZonotope(
        torch.tensor([0.5, -1.0], dtype=torch.float64),
        {
            (1, 0, 0): torch.tensor([0.25, -0.5], dtype=torch.float64),
            (0, 2, 0): torch.tensor([-0.1, 0.2], dtype=torch.float64),
            (1, 0, 1): torch.tensor([0.05, 0.15], dtype=torch.float64),
        },
        num_noise=3,
    )
    enclosure = z.interval_enclosure()
    for _ in range(128):
        eps = 2.0 * torch.rand(3, dtype=torch.float64) - 1.0
        _assert_contains(enclosure, _eval_pz(z, eps))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_shape_correctness_for_pz_twojet_network_outputs():
    d, m = 3, 2
    model = nn.Sequential(nn.Linear(d, 4, dtype=torch.float64), nn.Tanh(), nn.Linear(4, m, dtype=torch.float64)).double()
    domain = PolynomialZonotope.from_box(torch.full((d,), -0.2, dtype=torch.float64), torch.full((d,), 0.3, dtype=torch.float64))

    out = pz_twojet_forward(model, domain, chebyshev_degree=5, residual_subdivisions=64)

    assert out.Y.shape == (m,)
    assert out.J.shape == (m, d)
    assert out.H.shape == (m, d, d)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_affine_only_network_matches_exact_affine_value_jacobian_and_zero_hessian():
    d, h, m = 2, 3, 2
    model = nn.Sequential(nn.Linear(d, h, dtype=torch.float64), nn.Linear(h, m, dtype=torch.float64)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [0.5, 3.0], [-1.5, 0.25]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64))
        model[1].weight.copy_(torch.tensor([[2.0, -1.0, 0.5], [-0.25, 1.5, -2.0]], dtype=torch.float64))
        model[1].bias.copy_(torch.tensor([-0.4, 0.7], dtype=torch.float64))
    domain = PolynomialZonotope.from_box(torch.tensor([-1.0, 0.25], dtype=torch.float64), torch.tensor([0.5, 1.25], dtype=torch.float64))

    out = pz_twojet_forward(model, domain)
    expected_weight = model[1].weight.detach().matmul(model[0].weight.detach())
    expected_bias = model[1].weight.detach().matmul(model[0].bias.detach()) + model[1].bias.detach()

    assert torch.allclose(out.Y.center, expected_weight.matmul(domain.center) + expected_bias)
    for exp, coeff in domain.terms.items():
        assert torch.allclose(out.Y.terms[exp], expected_weight.matmul(coeff))
    assert torch.allclose(out.J.center, expected_weight)
    assert out.J.terms == {}
    assert torch.equal(out.H.center, torch.zeros(m, d, d, dtype=torch.float64))
    assert out.H.terms == {}


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_tanh_residual_certificate_bounds_sampled_residuals_by_delta():
    approx = compute_tanh_polynomial((-1.25, 0.75), degree=5, subdivisions=96)
    delta, metadata = certify_tanh_residual_subdivision((approx.lower, approx.upper), approx.coeffs, subdivisions=96)

    assert approx.delta == delta
    assert metadata["method"] == "outward-rounded-subdivision"
    for x in torch.linspace(approx.lower, approx.upper, steps=101, dtype=torch.float64):
        residual = math.tanh(float(x)) - _poly(approx.coeffs, float(x))
        assert abs(residual) <= approx.delta


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_small_tanh_network_pz_twojet_encloses_autograd_samples():
    enable_interval_eval()
    d, m = 2, 1
    model = nn.Sequential(nn.Linear(d, 2, dtype=torch.float64), nn.Tanh(), nn.Linear(2, m, dtype=torch.float64)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.4, -0.2], [0.1, 0.3]], dtype=torch.float64))
        model[0].bias.copy_(torch.tensor([0.05, -0.1], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor([[0.5, -0.3]], dtype=torch.float64))
        model[2].bias.copy_(torch.tensor([0.02], dtype=torch.float64))
    lower = torch.tensor([-0.4, -0.2], dtype=torch.float64)
    upper = torch.tensor([0.5, 0.3], dtype=torch.float64)
    domain = PolynomialZonotope.from_box(lower, upper)

    out = model.eval_pz_twojet(domain, chebyshev_degree=5, residual_subdivisions=64)
    y_interval = out.Y.interval_enclosure()
    j_interval = out.J.interval_enclosure()
    h_interval = out.H.interval_enclosure()

    samples = [lower, upper, (lower + upper) / 2]
    samples.extend(lower + (upper - lower) * torch.tensor(pair, dtype=torch.float64) for pair in ((0.2, 0.8), (0.7, 0.1), (0.9, 0.6)))
    for sample in samples:
        x = sample.clone().detach().requires_grad_(True)
        y = model(x)
        jac_rows = []
        hessians = []
        for i in range(m):
            grad = torch.autograd.grad(y[i], x, create_graph=True, retain_graph=True)[0]
            jac_rows.append(grad)
            hess_rows = []
            for j in range(d):
                hess_rows.append(torch.autograd.grad(grad[j], x, retain_graph=True)[0])
            hessians.append(torch.stack(hess_rows))
        jac = torch.stack(jac_rows)
        hess = torch.stack(hessians)

        _assert_contains(y_interval, y)
        _assert_contains(j_interval, jac)
        _assert_contains(h_interval, hess)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_pz_twojet_trace_is_opt_in_and_records_sequential_children():
    from intervalnets import PZTwoJet, PZTwoJetTraceResult

    model = nn.Sequential(nn.Identity(), nn.Linear(2, 3, dtype=torch.float64), nn.Tanh(), nn.Linear(3, 1, dtype=torch.float64)).double()
    domain = PolynomialZonotope.from_box(
        torch.tensor([-0.2, 0.1], dtype=torch.float64),
        torch.tensor([0.3, 0.4], dtype=torch.float64),
    )

    untraced = pz_twojet_forward(model, domain, residual_subdivisions=32)
    traced = pz_twojet_forward(model, domain, residual_subdivisions=32, return_trace=True)

    assert isinstance(untraced, PZTwoJet)
    assert isinstance(traced, PZTwoJetTraceResult)
    assert len(traced.records) == 1 + len(model)
    assert traced.records[0].layer_index == -1
    assert traced.records[0].layer_name == "input"
    assert traced.records[0].layer_type == "Input"
    assert [record.layer_type for record in traced.records[1:]] == ["Identity", "Linear", "Tanh", "Linear"]

    for record in traced.records:
        assert set(record.summary) == {"Y", "J", "H"}
        for component_name in ("Y", "J", "H"):
            component = getattr(record.jet, component_name)
            summary = record.summary[component_name]
            assert summary["shape"] == component.shape
            assert summary["num_noise"] == component.num_noise
            assert summary["noise_kinds"] == component.noise_kinds
            assert summary["term_count"] == len(component.terms)
            assert summary["max_degree"] == max((sum(exp) for exp in component.terms), default=0)

    assert traced.final.Y.shape == untraced.Y.shape
    assert traced.final.J.shape == untraced.J.shape
    assert traced.final.H.shape == untraced.H.shape
    assert traced.final.Y.num_noise == untraced.Y.num_noise
    assert traced.final.J.num_noise == untraced.J.num_noise
    assert traced.final.H.num_noise == untraced.H.num_noise
    for traced_interval, untraced_interval in (
        (traced.final.Y.interval_enclosure(), untraced.Y.interval_enclosure()),
        (traced.final.J.interval_enclosure(), untraced.J.interval_enclosure()),
        (traced.final.H.interval_enclosure(), untraced.H.interval_enclosure()),
    ):
        traced_lower, traced_upper = traced_interval.to_torch(dtype=torch.float64)
        untraced_lower, untraced_upper = untraced_interval.to_torch(dtype=torch.float64)
        assert torch.allclose(traced_lower, untraced_lower)
        assert torch.allclose(traced_upper, untraced_upper)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_eval_pz_twojet_return_trace_uses_monkey_patched_method():
    from intervalnets import PZTwoJetTraceResult

    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 2, dtype=torch.float64), nn.Tanh()).double()
    domain = PolynomialZonotope.from_box(torch.tensor([-0.1], dtype=torch.float64), torch.tensor([0.2], dtype=torch.float64))

    traced = model.eval_pz_twojet(domain, residual_subdivisions=32, return_trace=True)

    assert isinstance(traced, PZTwoJetTraceResult)
    assert len(traced.records) == 1 + len(model)
    assert traced.records[-1].jet is traced.final
