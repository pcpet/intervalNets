"""References for tanh' via high-precision cosh, independent of the q formula."""
from decimal import Decimal, Inexact, localcontext
import math
import random

import pytest

from intervalnets import Interval, tanh_prime_bounds


def _reference(x):
    if math.isinf(x):
        return Decimal(0)
    z = Decimal.from_float(abs(x))
    with localcontext() as ctx:
        # Resolve the O(x**2) departure from 1 even at subnormal inputs.
        ctx.prec = 100 + (2*max(0, -z.adjusted()) if z else 0)
        cosh = (z.exp() + (-z).exp()) / 2
        return 1 / (cosh*cosh)


def _reference_range(a, b):
    values = [_reference(a), _reference(b)]
    if a <= 0 <= b:
        values.append(Decimal(1))
    return min(values), max(values)


CASES = [
    (0.0, 0.0), (-1.0, 1.0), (-2.0, 0.5), (-0.5, 2.0),
    (2.0, 3.0), (-3.0, -2.0), (5.0, 6.0), (10.0, 11.0),
    (20.0, 21.0), (-21.0, -20.0), (100.0, 101.0),
    (350.0, 351.0), (373.0, 374.0), (400.0, 401.0),
    (-math.inf, math.inf), (0.0, math.inf), (-math.inf, 0.0),
    (2.0, math.inf), (-math.inf, -2.0),
]


@pytest.mark.parametrize("a,b", CASES)
def test_prime_range_encloses_reference_extrema_and_is_tight(a, b):
    result = tanh_prime_bounds(Interval(a, b))
    lower, upper = _reference_range(a, b)
    assert 0.0 <= result.lower <= result.upper <= 1.0
    assert Decimal.from_float(result.lower) <= lower
    assert Decimal.from_float(result.upper) >= upper
    assert result.lower >= math.nextafter(math.nextafter(float(lower), -math.inf), -math.inf)
    assert result.upper <= math.nextafter(math.nextafter(float(upper), math.inf), math.inf)
    if a <= 0 <= b:
        assert result.upper == 1.0


@pytest.mark.parametrize("x", [
    0.0, -0.0, math.nextafter(0.0, math.inf), -math.nextafter(0.0, math.inf),
    1e-300, -1e-300, 1e-20, -1e-20, 1e-8, -1e-8, 0.25,
    20.0, -20.0, 373.0, -373.0, 400.0, -400.0, math.inf, -math.inf,
])
def test_prime_point_values_are_outward_rounded(x):
    result = tanh_prime_bounds(Interval.point(x))
    value = _reference(x)
    assert Decimal.from_float(result.lower) <= value <= Decimal.from_float(result.upper)
    if x == 0:
        assert result.as_tuple() == (1.0, 1.0)
    elif math.isinf(x):
        assert result.as_tuple() == (0.0, 0.0)
    else:
        assert result.lower < 1.0
        assert result.upper > 0.0


def test_prime_saturated_tail_has_positive_lower_bound():
    # math.tanh(20) is 1, so the old 1-tanh(x)**2 expression returned 0.
    result = tanh_prime_bounds(Interval(20.0, 21.0))
    assert result.lower > 2e-18
    assert result.upper < 2e-17
    assert math.tanh(20.0) == 1.0


def test_prime_bounds_are_even_and_contain_random_samples():
    rng = random.Random(20261001)
    for _ in range(150):
        a = rng.uniform(-60, 60)
        b = a + 10**rng.uniform(-10, 1)
        result = tanh_prime_bounds(Interval(a, b))
        assert tanh_prime_bounds(Interval(-b, -a)).as_tuple() == result.as_tuple()
        for i in range(9):
            x = min(b, max(a, a+(b-a)*i/8))
            value = _reference(x)
            assert Decimal.from_float(result.lower) <= value <= Decimal.from_float(result.upper)


def test_prime_uses_independent_decimal_context():
    expected = tanh_prime_bounds(Interval(20.0, 21.0)).as_tuple()
    with localcontext() as ctx:
        ctx.prec = 2
        ctx.traps[Inexact] = True
        before = ctx.copy()
        assert tanh_prime_bounds(Interval(20.0, 21.0)).as_tuple() == expected
        assert ctx.prec == before.prec
        assert ctx.traps == before.traps
        assert ctx.flags == before.flags


@pytest.mark.parametrize("a,b", [(math.nan, 1.0), (0.0, math.nan)])
def test_prime_rejects_nan(a, b):
    with pytest.raises(ValueError, match="NaN"):
        tanh_prime_bounds(Interval(a, b))


@pytest.mark.parametrize("value", [(-1.0, 1.0), Interval([-1.0], [1.0])])
def test_prime_requires_a_scalar_interval(value):
    with pytest.raises(TypeError, match="scalar Interval"):
        tanh_prime_bounds(value)


def test_prime_large_finite_inputs_keep_nonzero_upper_bound():
    result = tanh_prime_bounds(Interval(1e300, 1e308))
    assert result.as_tuple() == (0.0, math.nextafter(0.0, math.inf))


def test_prime_pytorch_adapter_preserves_range_and_saturated_values():
    pytest.importorskip("torch")
    from intervalnets.pytorch import _interval_derivative_bounds_tanh
    for a, b in CASES:
        value = Interval(a, b)
        scalar = tanh_prime_bounds(value)
        padded = _interval_derivative_bounds_tanh(value)
        assert 0 <= padded.lower <= scalar.lower <= scalar.upper <= padded.upper <= 1
        if a <= 0 <= b:
            assert padded.upper == 1.0
        if a == b == 0:
            assert padded.as_tuple() == (1.0, 1.0)
    assert _interval_derivative_bounds_tanh(Interval(20.0, 21.0)).lower > 0


def test_scalar_activation_api_imports_without_optional_torch():
    import subprocess
    import sys
    from pathlib import Path

    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    script = f"""
import sys
from importlib.abc import MetaPathFinder
sys.path.insert(0, {source_dir!r})
class NoTorch(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('PyTorch disabled for this test', name='torch')
sys.meta_path.insert(0, NoTorch())
from intervalnets import Interval, tanh_prime_bounds, tanh_double_prime_bounds
assert tanh_prime_bounds(Interval(20,21)).lower > 0
assert tanh_prime_bounds(Interval.point(0)).as_tuple() == (1.0,1.0)
assert tanh_double_prime_bounds(Interval.point(0)).as_tuple() == (0.0,0.0)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
