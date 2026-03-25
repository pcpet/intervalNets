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
