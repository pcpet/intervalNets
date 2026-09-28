# Interval bounds for the second derivative of tanh

For `g(x) = tanh''(x)`, extrema on `[a,b]` occur at the two endpoints
and at any included stationary points `+/-c`, where

```
c = atanh(1/sqrt(3)) = log(2 + sqrt(3))/2
g(-c) = M, g(c) = -M, M = 4/(3*sqrt(3)).
```

Indeed, `g'(x) = -2(1-tanh(x)^2)(1-3*tanh(x)^2)`, whose only finite
zeros are `+/-c`. Taking the hull of these candidate values is the exact
real range. It avoids dependency inflation in `-2*T*(1-T*T)`.

## Machine enclosure argument

`activations.tanh_double_prime_bounds` encloses this range as follows:

1. Construct a bracket for `sqrt(3)` with Decimal's correctly rounded square
   root and its two neighboring decimal numbers. Directed arithmetic and a
   similarly bracketed logarithm give a bracket for `c` and an upper bound
   for `M`. Convert these bounds outward to binary64. If an input interval
   overlaps a critical-point bracket, include the corresponding global
   extremum conservatively.
2. At each finite nonzero endpoint, use `q = exp(-2*abs(x))` and
   `abs(g(x)) = 8*q*(1-q)/(1+q)^3`. Bracket the exponent by directed
   multiplication. Decimal's exponential is correctly rounded to nearest,
   even when a directed context is supplied, so explicitly take the lower
   neighbor of the lower exponential and upper neighbor of the upper one.
   Intersect this enclosure with `[0,1]`.
3. Every remaining operation uses directed Decimal arithmetic. All factors
   are nonnegative: the lower numerator uses `q_lo*(1-q_hi)`, the upper
   uses `q_hi*(1-q_lo)`. Divide the lower numerator by the upper denominator
   and conversely for the upper bound. Apply the exact sign last.
4. Convert each Decimal bound to binary64, compare its exact Decimal image
   against the source bound, and move one binary64 step outward when needed.
   This also handles underflow to zero. Near zero, increase decimal precision
   to resolve `1-q` without losing relative tightness.
5. For finite `abs(x) >= 400`, use `abs(g(x)) <= 8*exp(-800) < 2**-1074`
   and enclose by zero and the smallest positive binary64 subnormal, with
   the appropriate sign. The values at zero and the limits at infinity
   are exactly zero. Intersect the final range with the rounded global bounds.

The independent contexts do not modify the caller's decimal context. The
argument relies on the standard library's correctly rounded Decimal
`sqrt`, `ln`, and `exp`, not on an assumed error bound for system `libm`.
See [Python's Decimal documentation](https://docs.python.org/3/library/decimal.html).

The public scalar bounds enclose the mathematical derivative. They are not
a guarantee about every floating-point/autograd expression for that derivative.
The PyTorch adapter additionally applies the backend's existing float32
padding, retaining the sign restriction. Its returned bounds can therefore
be wider than the scalar API. Existing affine PZ enclosures are unchanged.

## Regression coverage

Tests compare against high-precision positive-exponential reference values,
including both extrema, adjacent floats at the critical points, point
intervals, signed zero, subnormal inputs and outputs, saturated tails,
unbounded intervals, invalid inputs, odd symmetry, and random intervals.
They also verify strict improvement over the former interval product on
representative intervals and independence from the caller's decimal context.
Numerical tests supplement the enclosure argument above; sampled values
alone are not an enclosure proof.
