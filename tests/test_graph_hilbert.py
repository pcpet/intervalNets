from __future__ import annotations

from math import sqrt

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from intervalnets import (
    DeepHybridOneJetResult,
    HilbertValueCertificate,
    IntervalTensor,
    PZIntegrationCell,
    PolynomialZonotope,
    SparseReferenceMomentBackend,
    build_factored_jacobian_graph,
    build_hilbert_value_certificate,
    certify_hybrid_graph_norms,
    integrate_hybrid_onejet_squared,
    integrate_hybrid_value_squared,
    neumann_polynomial_witness,
    scalar_hybrid_onejet_reverse,
)
from intervalnets.pz_integration import integrate_pz_value_squared


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
            torch.tensor([[0.6, -0.2], [0.3, 0.5], [-0.4, 0.25]])
        )
        model[0].bias.copy_(torch.tensor([0.05, -0.1, 0.15]))
        model[2].weight.copy_(
            torch.tensor([[0.5, -0.25, 0.3], [-0.15, 0.4, 0.35]])
        )
        model[2].bias.copy_(torch.tensor([0.02, -0.04]))
        model[4].weight.copy_(torch.tensor([[0.7, -0.45]]))
        model[4].bias.copy_(torch.tensor([0.3]))
    return model


def test_factored_graph_evaluation_preserves_shared_noise_exactly() -> None:
    model = _deep_model()
    domain = PolynomialZonotope.from_box(
        torch.full((2,), -0.2, dtype=torch.float64),
        torch.full((2,), 0.2, dtype=torch.float64),
    )
    result = scalar_hybrid_onejet_reverse(model, domain)
    assert isinstance(result, DeepHybridOneJetResult)
    graph = build_factored_jacobian_graph(result.jacobian)
    noise = -1.0 + 2.0 * torch.rand(
        (128, result.jacobian.num_noise),
        generator=torch.Generator().manual_seed(101),
        dtype=torch.float64,
    )
    assert torch.allclose(
        graph.evaluate(noise), result.jacobian.evaluate(noise), rtol=2e-13, atol=2e-13
    )
    # Reusing the exact same noise vector must be deterministic; no graph node
    # may silently manufacture an independent copy of a residual symbol.
    assert torch.equal(graph.evaluate(noise), graph.evaluate(noise))


def test_sparse_reference_moment_matches_explicit_pz_integral() -> None:
    model = _deep_model()
    box = IntervalTensor.from_bounds([-0.15, -0.1], [0.15, 0.1])
    cell = PZIntegrationCell.from_affine_box(box)
    result = scalar_hybrid_onejet_reverse(model, cell.domain)
    assert isinstance(result, DeepHybridOneJetResult)
    graph = build_factored_jacobian_graph(result.jacobian)
    backend = SparseReferenceMomentBackend(graph)
    polynomial = dict(backend.expand())
    zero = (0,) * graph.num_domain_noise
    center = polynomial.pop(zero)
    pz = PolynomialZonotope(
        center,
        polynomial,
        num_noise=graph.num_domain_noise,
        noise_kinds=("domain",) * graph.num_domain_noise,
    )
    explicit = integrate_pz_value_squared(pz, cell)
    expected = backend.sum_squares() * float(cell.volume)
    assert float(explicit.lower) == pytest.approx(expected, rel=5e-12, abs=5e-12)
    assert float(explicit.upper) == pytest.approx(expected, rel=5e-12, abs=5e-12)
    assert backend.diagnostics["output_terms"] > 1


def test_hilbert_value_compression_contains_sampled_l2_and_is_positive() -> None:
    model = _deep_model()
    box = IntervalTensor.from_bounds([-0.1, -0.1], [0.1, 0.1])
    cell = PZIntegrationCell.from_affine_box(box)
    certificate = build_hilbert_value_certificate(
        model, cell.domain, polynomial_degree=5, residual_subdivisions=256
    )
    assert certificate.remainder < certificate.nominal_norm
    lower = certificate.nominal_norm - certificate.remainder
    upper = certificate.nominal_norm + certificate.remainder
    samples = -0.1 + 0.2 * torch.rand(
        (20000, 2), generator=torch.Generator().manual_seed(103), dtype=torch.float64
    )
    sampled = sqrt(float(model(samples).square().mean()))
    assert 0.0 < lower <= sampled <= upper
    assert certificate.moment_states > 0


def test_larger_residual_budget_tightens_hilbert_compression() -> None:
    model = _deep_model()
    cell = PZIntegrationCell.from_affine_box(
        IntervalTensor.from_bounds([-0.1, -0.1], [0.1, 0.1])
    )
    coarse = build_hilbert_value_certificate(
        model, cell.domain, polynomial_degree=5, residual_subdivisions=64
    )
    fine = build_hilbert_value_certificate(
        model, cell.domain, polynomial_degree=5, residual_subdivisions=256
    )
    assert fine.remainder <= coarse.remainder
    assert fine.nominal_norm == pytest.approx(coarse.nominal_norm, rel=2e-14)


def test_neumann_witness_is_certified_for_affine_function() -> None:
    domain = PolynomialZonotope.from_box(
        torch.tensor([-0.1, -0.2], dtype=torch.float64),
        torch.tensor([0.1, 0.2], dtype=torch.float64),
    )
    value = HilbertValueCertificate(
        center=torch.tensor(0.4, dtype=torch.float64),
        domain_coefficients=torch.tensor([0.2, -0.1], dtype=torch.float64),
        remainder=0.0,
        polynomial_degree=1,
        residual_subdivisions=1,
        layers=(),
        moment_states=0,
        preactivation_centers=(),
        preactivation_coefficients=(),
        preactivation_remainders=(),
    )
    witness = neumann_polynomial_witness(value, domain)
    exact = sqrt(
        0.4**2
        + (0.2**2 + (-0.1) ** 2) / 3.0
        + (0.2 / 0.1) ** 2
        + (-0.1 / 0.2) ** 2
    )
    assert 0.0 < witness.lower_bound <= exact


def test_combined_graph_certificate_intersects_old_bounds_and_scales() -> None:
    model = _deep_model()
    box = IntervalTensor.from_bounds([-0.1, -0.1], [0.1, 0.1])
    cell = PZIntegrationCell.from_affine_box(box)
    result = scalar_hybrid_onejet_reverse(model, cell.domain)
    old_l2 = integrate_hybrid_value_squared(result, cell)
    old_w12 = integrate_hybrid_onejet_squared(result, cell)
    certificate = certify_hybrid_graph_norms(
        model,
        result,
        cell,
        polynomial_degree=5,
        residual_subdivisions=256,
        derivative_certificate_subdivisions=32,
    )
    assert float(certificate.l2_squared.lower) >= float(old_l2.lower)
    assert float(certificate.l2_squared.upper) <= float(old_l2.upper)
    assert float(certificate.w12_squared.lower) >= float(old_w12.lower)
    assert float(certificate.w12_squared.upper) <= float(old_w12.upper)
    volume = float(cell.volume)
    normalized_l2_lower = sqrt(float(certificate.l2_squared.lower) / volume)
    raw_l2_lower = sqrt(float(certificate.l2_squared.lower))
    assert raw_l2_lower == pytest.approx(sqrt(volume) * normalized_l2_lower)
    assert normalized_l2_lower > 0.0
