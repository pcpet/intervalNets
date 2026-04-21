from intervalnets.affine import AffineTensor
from intervalnets.interval import Interval
from math import inf, nextafter
import pytest


def test_point_interval_preserves_degenerate_bounds_on_initialization() -> None:
    interval = Interval.point(1.0)
    assert interval.lower == 1.0
    assert interval.upper == 1.0


def test_interval_addition_contains_exact_result() -> None:
    a = Interval.from_bounds([1.0, -2.0], [1.5, -1.5])
    b = Interval.from_bounds([3.0, 4.0], [3.25, 5.0])
    c = a + b
    assert c.contains([4.0, 2.0])
    assert c.contains([4.75, 3.5])


def test_interval_multiplication_handles_sign_changes() -> None:
    a = Interval.from_bounds(-2.0, 3.0)
    b = Interval.from_bounds(-4.0, 5.0)
    product = a * b
    assert product.lower <= -12.0
    assert product.upper >= 15.0


def test_interval_list_multiplication_encloses_standard_corner_products() -> None:
    a_lo, a_hi = -1.0, 2.0
    b_lo, b_hi = -3.0, 4.0

    standard_products = (
        a_lo * b_lo,
        a_lo * b_hi,
        a_hi * b_lo,
        a_hi * b_hi,
    )
    standard_lower = nextafter(min(standard_products), -inf)
    standard_upper = nextafter(max(standard_products), inf)

    product = Interval.from_bounds([a_lo], [a_hi]) * Interval.from_bounds([b_lo], [b_hi])

    assert product.lower[0] <= standard_lower
    assert product.upper[0] >= standard_upper


def test_interval_division_rejects_zero_crossing_denominator() -> None:
    numerator = Interval.from_bounds(1.0, 2.0)
    denominator = Interval.from_bounds(-1.0, 1.0)
    try:
        _ = numerator / denominator
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("Expected ZeroDivisionError for denominator interval containing zero.")


def test_from_bounds_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        _ = Interval.from_bounds([1.0, 2.0], [1.0])


def test_from_bounds_rejects_lower_greater_than_upper() -> None:
    with pytest.raises(ValueError):
        _ = Interval.from_bounds(2.0, 1.0)


def test_from_bounds_rejects_nested_lower_greater_than_upper() -> None:
    with pytest.raises(ValueError):
        _ = Interval.from_bounds([[0.0, 3.0]], [[1.0, 2.0]])


def test_interval_division_rejects_vector_denominator() -> None:
    numerator = Interval.from_bounds([1.0, 2.0], [3.0, 4.0])
    denominator = Interval.from_bounds([2.0, 3.0], [4.0, 5.0])
    with pytest.raises(NotImplementedError):
        _ = numerator / denominator


def test_affine_add_and_subtract_preserve_center_and_generator_structure() -> None:
    left = AffineTensor.from_bounds([0.0, 2.0], [2.0, 4.0])
    right = AffineTensor.from_bounds([-1.0, 1.0], [1.0, 3.0])

    summed = left + right
    diffed = left - right

    assert summed.c == (1.0, 5.0)
    assert diffed.c == (1.0, 1.0)
    assert len(summed.G[0]) == 4
    assert len(diffed.G[0]) == 4


def test_affine_map_matches_wc_plus_b_and_wg_for_fallback_backend() -> None:
    x = AffineTensor.from_bounds([-1.0, 2.0], [3.0, 4.0])
    W = ((2.0, -1.0), (0.5, 3.0))
    b = (0.25, -0.75)

    mapped = x.affine_map(W, b)

    expected_center = (2.0 * x.c[0] - 1.0 * x.c[1] + 0.25, 0.5 * x.c[0] + 3.0 * x.c[1] - 0.75)
    assert mapped.c == expected_center

    # Generator transform is WG. x.G is diagonal from from_bounds.
    assert mapped.G[0][0] == 2.0 * x.G[0][0]
    assert mapped.G[0][1] == -1.0 * x.G[1][1]
    assert mapped.G[1][0] == 0.5 * x.G[0][0]
    assert mapped.G[1][1] == 3.0 * x.G[1][1]
