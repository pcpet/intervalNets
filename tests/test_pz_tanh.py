import math

from intervalnets import Interval
from intervalnets.pz_tanh import (
    TanhApproximation,
    certify_tanh_residual_subdivision,
    compute_tanh_polynomial,
)


def _poly(coeffs, x):
    acc = 0.0
    for coeff in reversed(coeffs):
        acc = acc * x + coeff
    return acc


def test_compute_tanh_polynomial_returns_certified_metadata():
    approx = compute_tanh_polynomial(Interval(-1.0, 1.0), remez_degree=5, subdivisions=32)

    assert isinstance(approx, TanhApproximation)
    assert approx.degree == 5
    assert len(approx.coeffs) == 6
    assert approx.lower == -1.0
    assert approx.upper == 1.0
    assert approx.delta >= 0.0
    assert approx.metadata["remez_degree"] == 5
    assert "not a proof" in approx.metadata["proof_note"]
    assert approx.metadata["residual_certification"]["method"] == "outward-rounded-subdivision"


def test_subdivision_certificate_bounds_sampled_residuals():
    coeffs = (0.0, 1.0)  # p(x)=x is intentionally crude away from zero.
    delta, metadata = certify_tanh_residual_subdivision((-1.0, 1.0), coeffs, subdivisions=64)

    assert delta > 0.0
    assert metadata["subdivisions"] == 64
    assert "not final root-isolation" in metadata["note"]
    for idx in range(41):
        x = -1.0 + idx / 20.0
        assert abs(math.tanh(x) - _poly(coeffs, x)) <= delta


def test_compute_tanh_polynomial_validates_proposal_with_certificate():
    approx = compute_tanh_polynomial((-2.0, 0.5), degree=3, subdivisions=80)
    delta, _ = certify_tanh_residual_subdivision((approx.lower, approx.upper), approx.coeffs, subdivisions=80)

    assert approx.delta == delta
    for x in (-2.0, -1.25, -0.1, 0.5):
        assert abs(math.tanh(x) - _poly(approx.coeffs, x)) <= approx.delta
