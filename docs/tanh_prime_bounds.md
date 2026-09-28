# Outward-rounded interval bounds for tanh'

The function `h(x) = tanh'(x) = sech(x)^2` is even and decreases as `abs(x)`
increases. Its only finite maximum is `h(0)=1`. Thus the real range on `[a,b]`
is obtained at the largest and smallest distance to zero:

```
far = max(abs(a), abs(b))
near = 0 if a <= 0 <= b else min(abs(a), abs(b))
range = [h(far), h(near)].
```

The previous endpoint calculation `1-tanh(x)**2` could round to zero even when
the real derivative was representable and positive. For example, on `[20,21]`
the real range is approximately `[2.2998089057e-18, 1.6993417021e-17]`, whereas
the padded old result was approximately `[-1.4013e-45, 1.4013e-45]`.

## Rounding argument

Use `q=exp(-2*abs(x))` and `h(x)=4*q/(1+q)^2`. The rational function is
increasing on `[0,1]`, since its derivative there is `4*(1-q)/(1+q)^3`.

1. Convert the binary64 endpoint to an exact Decimal. Bound the multiplication
   by -2 with directed Decimal arithmetic. The resulting exponent interval
   brackets the real exponent, including very small inputs.
2. Decimal exponential is correctly rounded to nearest even. Take its lower
   neighbor at the lower exponent and upper neighbor at the upper exponent,
   and intersect with `[0,1]`. This gives `q_lo <= q <= q_hi` without a
   system-libm accuracy assumption.
3. By monotonicity, it suffices to bound `h(q_lo)` below and `h(q_hi)` above.
   For the lower bound, round `4*q_lo` down and `(1+q_lo)^2` up before
   dividing downward. For the upper bound, use the opposite rounding
   directions at `q_hi`. All factors are positive.
4. Convert the Decimal bounds outward to binary64, comparing the exact
   Decimal image of the converted float with the original Decimal bound.
   Clip to the exact global range `[0,1]`. At zero use `[1,1]` directly.
5. For finite `abs(x)>=400`, use `0<h(x)<=4*exp(-800)<2**-1074` and return
   `[0,2**-1074]` at that endpoint. This prevents accidental return of a
   zero upper bound under underflow. At infinite endpoints the limit is zero.

The contexts are independent of the caller's Decimal context and use the
existing activation helpers. There is no additional dependency. The public
result encloses the exact real range with outward rounding.

The PyTorch interval adapter retains its float32 outward padding, clipped to
`[0,1]`. Exact zero and one point results need no padding. Deep hybrid factors
round outward after casting the scalar bounds to their tensor dtype, also
clipping to `[0,1]`. These intervals concern the mathematical derivative;
they do not bound errors from every possible floating-point/autograd formula.
Affine and quadratic PZ approximations are separate representations.

## Verification

Tests use an independent high-precision `1/cosh(x)^2` reference and include
intervals crossing zero, point intervals, signed zero, subnormal inputs and
outputs, saturated tails, unbounded intervals, NaN and shape rejection,
even symmetry, and random intervals. Jacobian and deep-factor tests check
the actual integration points, including float32 tensor conversion and
underflow. Tests supplement the rounding argument; sampled values alone
are not a proof of an enclosure.
