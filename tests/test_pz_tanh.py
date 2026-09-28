import math

import pytest

from intervalnets import Interval
from intervalnets.pz_tanh import (
    AffineTanhEnclosure,
    TanhApproximation,
    affine_tanh_double_prime_enclosure,
    affine_tanh_enclosure,
    affine_tanh_prime_enclosure,
    quadratic_tanh_prime_enclosure,
    certify_tanh_residual_subdivision,
    compute_tanh_polynomial,
)


def _poly(coeffs, x):
    acc = 0.0
    for coeff in reversed(coeffs):
        acc = acc * x + coeff
    return acc


def test_compute_tanh_polynomial_returns_certified_metadata():
    approx = compute_tanh_polynomial(
        Interval(-1.0, 1.0), chebyshev_degree=5, subdivisions=32
    )

    assert isinstance(approx, TanhApproximation)
    assert approx.degree == 5
    assert len(approx.coeffs) == 6
    assert approx.lower == -1.0
    assert approx.upper == 1.0
    assert approx.delta >= 0.0
    assert approx.metadata["chebyshev_degree"] == 5
    assert "not a proof" in approx.metadata["proof_note"]
    assert (
        approx.metadata["residual_certification"]["method"]
        == "outward-rounded-subdivision"
    )


def test_compute_tanh_polynomial_accepts_deprecated_remez_alias():
    with pytest.warns(DeprecationWarning, match="remez_degree is deprecated"):
        approx = compute_tanh_polynomial(
            Interval(-1.0, 1.0), remez_degree=5, subdivisions=32
        )

    assert approx.metadata["chebyshev_degree"] == 5
    assert approx.metadata["legacy_remez_degree"] == 5
    assert "remez_degree" not in approx.metadata


def test_subdivision_certificate_bounds_sampled_residuals():
    coeffs = (0.0, 1.0)  # p(x)=x is intentionally crude away from zero.
    delta, metadata = certify_tanh_residual_subdivision(
        (-1.0, 1.0), coeffs, subdivisions=64
    )

    assert delta > 0.0
    assert metadata["subdivisions"] == 64
    assert "not final root-isolation" in metadata["note"]
    for idx in range(41):
        x = -1.0 + idx / 20.0
        assert abs(math.tanh(x) - _poly(coeffs, x)) <= delta


def test_compute_tanh_polynomial_validates_proposal_with_certificate():
    approx = compute_tanh_polynomial((-2.0, 0.5), degree=3, subdivisions=80)
    delta, _ = certify_tanh_residual_subdivision(
        (approx.lower, approx.upper), approx.coeffs, subdivisions=80
    )

    assert approx.delta == delta
    for x in (-2.0, -1.25, -0.1, 0.5):
        assert abs(math.tanh(x) - _poly(approx.coeffs, x)) <= approx.delta


def _tanh_prime(x):
    t = math.tanh(x)
    return 1.0 - t * t


def _tanh_double_prime(x):
    t = math.tanh(x)
    return -2.0 * t + 2.0 * t * t * t


@pytest.mark.parametrize(
    ("helper", "func", "name", "interval"),
    [
        (affine_tanh_enclosure, math.tanh, "tanh", (-2.0, 1.25)),
        (affine_tanh_prime_enclosure, _tanh_prime, "tanh_prime", (-1.5, 1.75)),
        (
            affine_tanh_double_prime_enclosure,
            _tanh_double_prime,
            "tanh_double_prime",
            (-2.0, 2.0),
        ),
    ],
)
def test_affine_tanh_enclosures_bound_sampled_values(helper, func, name, interval):
    enclosure = helper(interval)

    assert isinstance(enclosure, AffineTanhEnclosure)
    assert enclosure.lower == interval[0]
    assert enclosure.upper == interval[1]
    assert enclosure.function == name
    assert enclosure.delta >= 0.0
    assert enclosure.metadata["method"] == "finite-stationary-candidates"
    assert "outward_rounding" in enclosure.metadata
    assert len(enclosure.metadata["x_candidates"]) >= 2
    for idx in range(101):
        x = interval[0] + (interval[1] - interval[0]) * idx / 100.0
        assert abs(func(x) - (enclosure.p * x + enclosure.q)) <= enclosure.delta


@pytest.mark.parametrize(
    ("helper", "func", "name"),
    [
        (affine_tanh_enclosure, math.tanh, "tanh"),
        (affine_tanh_prime_enclosure, _tanh_prime, "tanh_prime"),
        (affine_tanh_double_prime_enclosure, _tanh_double_prime, "tanh_double_prime"),
    ],
)
def test_affine_tanh_enclosures_handle_point_intervals(helper, func, name):
    enclosure = helper(Interval(0.25, 0.25))

    assert enclosure.p == 0.0
    assert enclosure.q == func(0.25)
    assert enclosure.delta == 0.0
    assert enclosure.lower == 0.25
    assert enclosure.upper == 0.25
    assert enclosure.function == name
    assert enclosure.metadata["method"] == "point-interval"
    assert enclosure.metadata["x_candidates"] == (0.25,)


@pytest.mark.parametrize(
    "interval",
    [(-2.0, 2.0), (-1.576, 1.613), (-1.0, 1.0), (-0.35, 0.8), (0.2, 1.4)],
)
def test_quadratic_tanh_prime_enclosure_bounds_sampled_values(interval) -> None:
    lower, upper = interval
    enclosure = quadratic_tanh_prime_enclosure(Interval(lower, upper))
    c, b, a = enclosure.coeffs

    assert enclosure.delta >= 0.0
    assert enclosure.metadata["certificate_subdivisions"] == 64
    for index in range(2001):
        x = lower + (upper - lower) * index / 2000.0
        residual = _tanh_prime(x) - (c + b * x + a * x * x)
        assert abs(residual) <= enclosure.delta


def test_quadratic_tanh_prime_enclosure_is_tighter_on_symmetric_bump() -> None:
    interval = Interval(-1.6, 1.6)
    affine = affine_tanh_prime_enclosure(interval)
    quadratic = quadratic_tanh_prime_enclosure(interval)

    assert quadratic.delta < 0.4 * affine.delta
