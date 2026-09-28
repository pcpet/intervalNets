"""Scalar activation bounds with explicit control of elementary-function errors."""

from __future__ import annotations

from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR
from math import inf, isinf, isnan, nextafter

from .interval import Interval


def _contexts(precision: int = 60) -> tuple[Context, Context]:
    # Do not depend on, or modify, the caller's decimal context.
    lower = Context(prec=precision, rounding=ROUND_FLOOR, Emin=-999999, Emax=999999)
    upper = Context(prec=precision, rounding=ROUND_CEILING, Emin=-999999, Emax=999999)
    return lower, upper


def _float_lower(value: Decimal) -> float:
    result = float(value)
    return nextafter(result, -inf) if Decimal.from_float(result) > value else result


def _float_upper(value: Decimal) -> float:
    result = float(value)
    return nextafter(result, inf) if Decimal.from_float(result) < value else result


def _critical_bounds() -> tuple[float, float, float]:
    down, up = _contexts()
    # Decimal sqrt/ln/exp are correctly rounded, but not necessarily in the
    # requested directed mode. Their adjacent decimal numbers bracket them.
    root = down.sqrt(Decimal(3))
    root_lo, root_hi = down.next_minus(root), up.next_plus(root)
    # atanh(1/sqrt(3)) = log(2 + sqrt(3))/2.
    c_lo = down.divide(down.next_minus(down.ln(down.add(2, root_lo))), 2)
    c_hi = up.divide(up.next_plus(up.ln(up.add(2, root_hi))), 2)
    maximum_hi = up.divide(4, down.multiply(3, root_lo))
    return _float_lower(c_lo), _float_upper(c_hi), _float_upper(maximum_hi)


_CRITICAL_LO, _CRITICAL_HI, _MAXIMUM_HI = _critical_bounds()
_MIN_SUBNORMAL = nextafter(0.0, inf)


def _tanh_prime_point_bounds(x: float) -> tuple[float, float]:
    if x == 0.0:
        return 1.0, 1.0
    if isinf(x):
        return 0.0, 0.0
    if abs(x) >= 400.0:
        # 0 < tanh'(x) <= 4*exp(-800) < 2**-1074 at finite endpoints.
        return 0.0, _MIN_SUBNORMAL

    down, up = _contexts()
    magnitude = Decimal.from_float(abs(x))
    exponent_lo = down.multiply(-2, magnitude)
    exponent_hi = up.multiply(-2, magnitude)
    q_lo = max(Decimal(0), down.next_minus(down.exp(exponent_lo)))
    q_hi = min(Decimal(1), up.next_plus(up.exp(exponent_hi)))

    # h(q)=4*q/(1+q)**2 is increasing for q in [0,1]. Evaluate h(q_lo)
    # downward and h(q_hi) upward, rather than subtracting tanh(x)**2
    # from 1. This also avoids cancellation near zero.
    base_lo_upper = up.add(1, q_lo)
    base_hi_lower = down.add(1, q_hi)
    lower = down.divide(
        down.multiply(4, q_lo), up.multiply(base_lo_upper, base_lo_upper)
    )
    upper = up.divide(
        up.multiply(4, q_hi), down.multiply(base_hi_lower, base_hi_lower)
    )
    return max(0.0, _float_lower(lower)), min(1.0, _float_upper(upper))


def tanh_prime_bounds(value: Interval) -> Interval:
    """Enclose the mathematical ``tanh'`` on a scalar interval.

    ``tanh'`` is even and decreases with distance from zero. Evaluate the
    largest distance for the minimum and the smallest for the maximum;
    a domain containing zero has maximum exactly 1. Endpoint values use
    outward-rounded Decimal arithmetic for 4*q/(1+q)**2 with q=exp(-2|x|).
    Bounds stay within [0,1], including subnormal/underflow cases. Infinite
    endpoints use limits. No assumption about libm accuracy is needed.
    """
    if not isinstance(value, Interval) or value.shape != ():
        raise TypeError("tanh_prime_bounds requires a scalar Interval.")
    a, b = float(value.lower), float(value.upper)
    if isnan(a) or isnan(b):
        raise ValueError("tanh_prime_bounds does not accept NaN endpoints.")

    far = max(abs(a), abs(b))
    near = 0.0 if a <= 0.0 <= b else min(abs(a), abs(b))
    far_bounds = _tanh_prime_point_bounds(far)
    upper = far_bounds[1] if near == far else _tanh_prime_point_bounds(near)[1]
    return Interval.from_bounds(far_bounds[0], upper)


def _tanh_double_prime_point_bounds(x: float) -> tuple[float, float]:
    if x == 0.0 or isinf(x):
        # The values at infinite endpoints are the limits at infinity.
        return 0.0, 0.0
    if abs(x) >= 400.0:
        # |tanh''(x)| <= 8*exp(-2*|x|) <= 8*exp(-800) < 2**-1074.
        return (-_MIN_SUBNORMAL, 0.0) if x > 0 else (0.0, _MIN_SUBNORMAL)

    magnitude = Decimal.from_float(abs(x))
    # Near zero, 1-exp(-2|x|) loses decimal places. Extra precision keeps
    # the enclosure narrow even at binary64 subnormal inputs.
    down, up = _contexts(60 + max(0, -magnitude.adjusted()))
    exponent_lo = down.multiply(-2, magnitude)
    exponent_hi = up.multiply(-2, magnitude)
    q_lo = max(Decimal(0), down.next_minus(down.exp(exponent_lo)))
    q_hi = min(Decimal(1), up.next_plus(up.exp(exponent_hi)))

    # |tanh''(x)| = 8*q*(1-q)/(1+q)**3, q=exp(-2|x|).
    # All factors are nonnegative. Bound each arithmetic operation outward;
    # ordinary tanh(x) may have already rounded to +/-1 in the tails.
    numerator_lo = down.multiply(8, down.multiply(q_lo, down.subtract(1, q_hi)))
    numerator_hi = up.multiply(8, up.multiply(q_hi, up.subtract(1, q_lo)))
    base_lo, base_hi = down.add(1, q_lo), up.add(1, q_hi)
    denominator_lo = down.multiply(down.multiply(base_lo, base_lo), base_lo)
    denominator_hi = up.multiply(up.multiply(base_hi, base_hi), base_hi)
    lower = _float_lower(down.divide(numerator_lo, denominator_hi))
    upper = _float_upper(up.divide(numerator_hi, denominator_lo))
    return (-upper, -lower) if x > 0 else (lower, upper)


def tanh_double_prime_bounds(value: Interval) -> Interval:
    """Enclose the real-valued ``tanh''`` over a scalar interval.

    Extrema occur at the endpoints or at +/-atanh(1/sqrt(3)), with
    values -/+4/(3*sqrt(3)). This gives the exact real range before
    numerical rounding; the returned binary64 endpoints round outward.
    Critical-point membership is conservative if the input overlaps the
    rounded bracket of a critical point. Infinite endpoints use limits.

    Decimal elementary functions and directed arithmetic provide the
    rounding bounds without a libm accuracy assumption or a new dependency.
    This encloses the mathematical function, not errors in an arbitrary
    floating-point implementation of its derivative.
    """
    if not isinstance(value, Interval) or value.shape != ():
        raise TypeError("tanh_double_prime_bounds requires a scalar Interval.")
    a, b = float(value.lower), float(value.upper)
    if isnan(a) or isnan(b):
        raise ValueError("tanh_double_prime_bounds does not accept NaN endpoints.")

    left = _tanh_double_prime_point_bounds(a)
    right = _tanh_double_prime_point_bounds(b)
    lower, upper = min(left[0], right[0]), max(left[1], right[1])
    if a <= _CRITICAL_HI and b >= _CRITICAL_LO:
        lower = -_MAXIMUM_HI
    if a <= -_CRITICAL_LO and b >= -_CRITICAL_HI:
        upper = _MAXIMUM_HI
    return Interval.from_bounds(max(lower, -_MAXIMUM_HI), min(upper, _MAXIMUM_HI))
