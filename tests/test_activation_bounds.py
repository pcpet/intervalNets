"""High-precision references and regression cases for real tanh'' ranges."""
from decimal import Decimal, Inexact, localcontext
import math
import random

import pytest

from intervalnets import Interval, tanh_double_prime_bounds


def _reference(x):
    if math.isinf(x):
        return Decimal(0)
    z = Decimal.from_float(abs(x))
    with localcontext() as ctx:
        ctx.prec = 100 + (max(0, -z.adjusted()) if z else 0)
        # Independent positive-exponential formula; no tanh saturation.
        e = (2*z).exp()
        result = -8*e*(e-1)/(e+1)**3
        return result if x >= 0 else -result


def _critical_reference():
    with localcontext() as ctx:
        ctx.prec = 100
        root = Decimal(3).sqrt()
        return (2+root).ln()/2, 4/(3*root)


def _range_reference(a, b):
    c, maximum = _critical_reference()
    with localcontext() as ctx:
        ctx.prec = 100
        points = [_reference(a), _reference(b)]
        if Decimal.from_float(a) <= c <= Decimal.from_float(b):
            points.append(-maximum)
        if Decimal.from_float(a) <= -c <= Decimal.from_float(b):
            points.append(maximum)
        return min(points), max(points)


CASES = [
    (-0.1, 0.1), (-1.0, 1.0), (-2.0, 2.0), (0.0, 0.5),
    (0.5, 1.0), (1.0, 2.0), (2.0, 3.0), (-3.0, -2.0),
    (5.0, 6.0), (20.0, 21.0), (-21.0, -20.0),
    (350.0, 351.0), (373.0, 374.0), (400.0, 401.0),
    (-math.inf, math.inf), (0.0, math.inf), (-math.inf, 0.0),
]


@pytest.mark.parametrize("a,b", CASES)
def test_range_encloses_high_precision_extrema_and_is_tight(a, b):
    bounds = tanh_double_prime_bounds(Interval(a, b))
    lo, hi = _range_reference(a, b)
    assert Decimal.from_float(bounds.lower) <= lo
    assert Decimal.from_float(bounds.upper) >= hi
    # Up to two extra binary64 steps permit conservative critical brackets
    # and the explicit far-tail underflow enclosure.
    assert bounds.lower >= math.nextafter(math.nextafter(float(lo), -math.inf), -math.inf)
    assert bounds.upper <= math.nextafter(math.nextafter(float(hi), math.inf), math.inf)


@pytest.mark.parametrize("x", [
    0.0, -0.0, math.nextafter(0.0, math.inf), -math.nextafter(0.0, math.inf),
    1e-300, -1e-300, 1e-20, -1e-20, 0.25, -0.25, 1.0,
    20.0, -20.0, 373.0, -373.0, 400.0, -400.0,
])
def test_point_intervals_enclose_the_real_value(x):
    bounds = tanh_double_prime_bounds(Interval.point(x))
    reference = _reference(x)
    assert Decimal.from_float(bounds.lower) <= reference <= Decimal.from_float(bounds.upper)
    if x == 0.0:
        assert bounds.as_tuple() == (0.0, 0.0)
    elif x > 0:
        assert bounds.upper <= 0
    else:
        assert bounds.lower >= 0


def test_adjacent_floats_around_both_critical_points():
    critical, _ = _critical_reference()
    c = float(critical)
    for sign in [-1.0, 1.0]:
        center = sign*c
        points = [
            math.nextafter(center, -math.inf), center,
            math.nextafter(center, math.inf),
        ]
        for i, a in enumerate(points):
            for b in points[i:]:
                bounds = tanh_double_prime_bounds(Interval(a, b))
                lo, hi = _range_reference(a, b)
                assert Decimal.from_float(bounds.lower) <= lo
                assert Decimal.from_float(bounds.upper) >= hi


def test_random_intervals_contain_samples_and_respect_odd_symmetry():
    rng = random.Random(20260928)
    for _ in range(150):
        a = rng.uniform(-30, 30)
        b = a + 10**rng.uniform(-10, 1)
        bounds = tanh_double_prime_bounds(Interval(a, b))
        reflected = tanh_double_prime_bounds(Interval(-b, -a))
        assert reflected.lower == -bounds.upper
        assert reflected.upper == -bounds.lower
        for i in range(9):
            x = a + (b-a)*i/8
            # Guard the sampled float against a rounded endpoint overshoot.
            x = min(b, max(a, x))
            assert Decimal.from_float(bounds.lower) <= _reference(x) <= Decimal.from_float(bounds.upper)


def test_does_not_depend_on_or_change_decimal_context():
    expected = tanh_double_prime_bounds(Interval(0.5, 1.0)).as_tuple()
    with localcontext() as ctx:
        ctx.prec = 3
        ctx.traps[Inexact] = True
        before = ctx.copy()
        assert tanh_double_prime_bounds(Interval(0.5, 1.0)).as_tuple() == expected
        assert ctx.prec == before.prec
        assert ctx.traps == before.traps
        assert ctx.flags == before.flags


@pytest.mark.parametrize("a,b", [(math.nan, 1.0), (0.0, math.nan)])
def test_rejects_nan(a, b):
    with pytest.raises(ValueError, match="NaN"):
        tanh_double_prime_bounds(Interval(a, b))


@pytest.mark.parametrize("value", [(-1.0, 1.0), Interval([-1.0], [1.0])])
def test_requires_scalar_interval(value):
    with pytest.raises(TypeError, match="scalar Interval"):
        tanh_double_prime_bounds(value)


def test_large_finite_inputs_do_not_overflow():
    bounds = tanh_double_prime_bounds(Interval(1e300, 1e308))
    assert bounds.as_tuple() == (-math.nextafter(0.0, math.inf), 0.0)


@pytest.mark.parametrize("a,b", [(-1.0, 1.0), (0.5, 1.0), (2.0, 3.0)])
def test_strictly_improves_old_interval_product(a, b):
    t = Interval(math.tanh(a), math.tanh(b))
    old = -(Interval.point(2)*t*(Interval.point(1)-t*t))
    new = tanh_double_prime_bounds(Interval(a, b))
    assert old.lower <= new.lower <= new.upper <= old.upper
    assert new.upper-new.lower < old.upper-old.lower


def test_pytorch_adapter_encloses_scalar_bounds_and_preserves_sign():
    pytest.importorskip("torch")
    from intervalnets.pytorch import _interval_second_derivative_bounds_tanh
    for a, b in CASES[:12] + [(0.0, 0.0)]:
        value = Interval(a, b)
        real = tanh_double_prime_bounds(value)
        padded = _interval_second_derivative_bounds_tanh(value)
        assert padded.lower <= real.lower <= real.upper <= padded.upper
        if a >= 0:
            assert padded.upper <= 0
        if b <= 0:
            assert padded.lower >= 0
