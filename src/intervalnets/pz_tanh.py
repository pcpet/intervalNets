"""Certified polynomial approximation scaffolding for ``tanh`` on scalar intervals.

The polynomial fit produced here is only a numerical proposal.  It is never
used as proof: callers must rely on ``certify_tanh_residual_subdivision`` (or a
future root-isolation/Remez certificate backend) for the rigorous residual
``delta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import acos, atanh, cos, inf, nextafter, pi, sqrt, tanh
from typing import Any, Mapping, Sequence
from warnings import warn

from .interval import Interval


@dataclass(frozen=True)
class TanhApproximation:
    """Certified scalar polynomial enclosure for ``tanh`` on an interval.

    Attributes:
        coeffs: Power-basis coefficients in ascending order, i.e.
            ``p(x) = coeffs[0] + coeffs[1] * x + ...``.
        lower: Lower endpoint of the scalar domain interval.
        upper: Upper endpoint of the scalar domain interval.
        delta: Certified non-negative residual satisfying
            ``tanh(x) - p(x) in [-delta, delta]`` for every ``x`` in the
            interval.
        degree: Configured Chebyshev approximation degree.
        metadata: Certification/proposal details, including the proposal method
            and subdivision certificate settings.
    """

    coeffs: tuple[float, ...]
    lower: float
    upper: float
    delta: float
    degree: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AffineTanhEnclosure:
    """Scalar affine-plus-error enclosure for a tanh jet on an interval.

    The returned parameters certify ``function(x) in p*x + q +
    delta*[-1, 1]`` for every scalar ``x`` in ``[lower, upper]``.
    ``metadata`` records the finite stationary candidates and the final
    floating-point safety inflation used for outward rounding.
    """

    p: float
    q: float
    delta: float
    lower: float
    upper: float
    function: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuadraticTanhEnclosure:
    """Certified quadratic-plus-error enclosure for a tanh jet.

    ``coeffs`` are power-basis coefficients ``(c, b, a)`` satisfying
    ``function(x) in c + b*x + a*x**2 + delta*[-1, 1]`` throughout the
    scalar interval.  The normalized coefficients are also recorded in
    ``metadata`` so callers can evaluate the parabola stably around the
    interval midpoint.
    """

    coeffs: tuple[float, float, float]
    delta: float
    lower: float
    upper: float
    function: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _tanh_value_from_t(j: int, t: float) -> float:
    if j == 0:
        return t
    if j == 1:
        return 1.0 - t * t
    if j == 2:
        return -2.0 * t + 2.0 * t * t * t
    raise ValueError("j must be 0, 1, or 2.")


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _append_unique(
    values: list[float], value: float, *, abs_tol: float = 1e-14
) -> None:
    if not any(abs(value - existing) <= abs_tol for existing in values):
        values.append(value)


def _append_if_in_t_interval(
    values: list[float], t: float, ta: float, tb: float
) -> None:
    tlo = min(ta, tb)
    thi = max(ta, tb)
    tol = 8.0 * 2.220446049250313e-16 * max(1.0, abs(tlo), abs(thi), abs(t))
    if -1.0 < t < 1.0 and tlo - tol <= t <= thi + tol:
        _append_unique(values, _clamp(t, -1.0 + 5e-324, 1.0 - 2.220446049250313e-16))


def _stationary_t_candidates(
    j: int, p: float, ta: float, tb: float
) -> tuple[float, ...]:
    candidates: list[float] = []
    if j == 0:
        d = max(0.0, 1.0 - p)
        s = sqrt(d)
        _append_if_in_t_interval(candidates, s, ta, tb)
        _append_if_in_t_interval(candidates, -s, ta, tb)
    elif j == 1:
        z = _clamp((3.0 * sqrt(3.0) / 4.0) * p, -1.0, 1.0)
        theta = acos(z) / 3.0
        for k in range(3):
            t = (2.0 / sqrt(3.0)) * cos(theta - 2.0 * pi * k / 3.0)
            _append_if_in_t_interval(candidates, t, ta, tb)
    elif j == 2:
        d = max(0.0, 1.0 - 1.5 * p)
        s = sqrt(d)
        for u in ((2.0 + s) / 3.0, (2.0 - s) / 3.0):
            if 0.0 <= u < 1.0:
                v = sqrt(u)
                _append_if_in_t_interval(candidates, v, ta, tb)
                _append_if_in_t_interval(candidates, -v, ta, tb)
    else:
        raise ValueError("j must be 0, 1, or 2.")
    return tuple(candidates)


def _affine_tanh_jet_enclosure(
    interval: Interval | Sequence[float], j: int, function_name: str
) -> AffineTanhEnclosure:
    lower, upper = _scalar_interval_bounds(interval)
    if lower == upper:
        value = _tanh_value_from_t(j, tanh(lower))
        return AffineTanhEnclosure(
            p=0.0,
            q=value,
            delta=0.0,
            lower=lower,
            upper=upper,
            function=function_name,
            metadata={
                "method": "point-interval",
                "x_candidates": (lower,),
                "t_candidates": (tanh(lower),),
                "residuals": (value,),
                "outward_rounding": "exact point interval; no inflation required",
            },
        )

    ta = tanh(lower)
    tb = tanh(upper)
    fa = _tanh_value_from_t(j, ta)
    fb = _tanh_value_from_t(j, tb)
    p = (fb - fa) / (upper - lower)
    t_candidates = _stationary_t_candidates(j, p, ta, tb)

    x_candidates = [lower, upper]
    for t in t_candidates:
        x = atanh(t)
        if lower < x < upper:
            _append_unique(x_candidates, x)

    residuals = []
    candidate_records = []
    for x in x_candidates:
        t = tanh(x)
        value = _tanh_value_from_t(j, t)
        residual = value - p * x
        residuals.append(residual)
        candidate_records.append({"x": x, "t": t, "residual": residual})

    rmin = min(residuals)
    rmax = max(residuals)
    q = (rmax + rmin) / 2.0
    delta = (rmax - rmin) / 2.0

    # The formulas above identify the exact real residual extrema.  Inflate the
    # final symmetric radius to account for ordinary floating-point evaluation
    # of candidates, residuals, and midpoint/radius arithmetic.
    rounding_inflation = (
        nextafter(max(abs(q), abs(delta), abs(p), abs(rmin), abs(rmax), 1.0), inf)
        * 32.0
        * 2.220446049250313e-16
    )
    delta = nextafter(delta + rounding_inflation, inf)

    return AffineTanhEnclosure(
        p=p,
        q=q,
        delta=delta,
        lower=lower,
        upper=upper,
        function=function_name,
        metadata={
            "method": "finite-stationary-candidates",
            "candidate_t_values": t_candidates,
            "x_candidates": tuple(x_candidates),
            "candidate_records": tuple(candidate_records),
            "residual_min": rmin,
            "residual_max": rmax,
            "outward_rounding": "final nextafter(delta + 32*eps*scale, +inf) safety inflation",
            "rounding_inflation": rounding_inflation,
        },
    )


def affine_tanh_enclosure(interval: Interval | Sequence[float]) -> AffineTanhEnclosure:
    """Return a certified scalar affine enclosure for ``tanh`` on ``interval``."""

    return _affine_tanh_jet_enclosure(interval, 0, "tanh")


def affine_tanh_prime_enclosure(
    interval: Interval | Sequence[float],
) -> AffineTanhEnclosure:
    """Return a certified scalar affine enclosure for ``tanh'`` on ``interval``."""

    return _affine_tanh_jet_enclosure(interval, 1, "tanh_prime")


def affine_tanh_double_prime_enclosure(
    interval: Interval | Sequence[float],
) -> AffineTanhEnclosure:
    """Return a certified scalar affine enclosure for ``tanh''`` on ``interval``."""

    return _affine_tanh_jet_enclosure(interval, 2, "tanh_double_prime")


def _tanh_prime_third_derivative_from_t(t: float) -> float:
    """Return the third derivative of ``tanh'`` as a polynomial in tanh(x)."""

    return 16.0 * t - 40.0 * t**3 + 24.0 * t**5


def _tanh_prime_first_derivative_from_t(t: float) -> float:
    return -2.0 * t + 2.0 * t**3


def _tanh_prime_second_derivative_from_t(t: float) -> float:
    return -2.0 + 8.0 * t**2 - 6.0 * t**4


def _quadratic_residual_taylor_certificate(
    lower: float,
    upper: float,
    coeffs: tuple[float, float, float],
    subdivisions: int,
) -> tuple[float, float, tuple[Mapping[str, float], ...]]:
    """Bound the residual range by a fixed, non-adaptive Taylor-form pass."""

    if subdivisions < 1:
        raise ValueError("subdivisions must be positive.")
    c, b, a = coeffs
    step = (upper - lower) / subdivisions
    residual_lower = inf
    residual_upper = -inf
    records: list[Mapping[str, float]] = []
    critical_t = (0.0, sqrt(2.0 / 3.0), -sqrt(2.0 / 3.0))
    for index in range(subdivisions):
        bin_lower = lower + index * step
        bin_upper = upper if index + 1 == subdivisions else lower + (index + 1) * step
        center = (bin_lower + bin_upper) / 2.0
        radius = (bin_upper - bin_lower) / 2.0
        t_center = tanh(center)
        function_value = 1.0 - t_center * t_center
        polynomial_value = c + b * center + a * center * center
        residual_center = function_value - polynomial_value
        residual_slope = _tanh_prime_first_derivative_from_t(t_center) - (
            b + 2.0 * a * center
        )

        ta, tb = tanh(bin_lower), tanh(bin_upper)
        second_candidates = [
            _tanh_prime_second_derivative_from_t(ta) - 2.0 * a,
            _tanh_prime_second_derivative_from_t(tb) - 2.0 * a,
        ]
        for t in critical_t:
            if ta <= t <= tb:
                second_candidates.append(
                    _tanh_prime_second_derivative_from_t(t) - 2.0 * a
                )
        second_lower = min(second_candidates)
        second_upper = max(second_candidates)
        linear_radius = abs(residual_slope) * radius
        quadratic_scale = 0.5 * radius * radius
        local_lower = (
            residual_center
            - linear_radius
            + min(0.0, quadratic_scale * second_lower)
        )
        local_upper = (
            residual_center
            + linear_radius
            + max(0.0, quadratic_scale * second_upper)
        )
        residual_lower = min(residual_lower, local_lower)
        residual_upper = max(residual_upper, local_upper)
        records.append({
            "lower": bin_lower,
            "upper": bin_upper,
            "residual_lower": local_lower,
            "residual_upper": local_upper,
        })
    return residual_lower, residual_upper, tuple(records)


def quadratic_tanh_prime_enclosure(
    interval: Interval | Sequence[float],
    *,
    certificate_subdivisions: int = 64,
) -> QuadraticTanhEnclosure:
    """Return a cheap certified three-point quadratic enclosure for ``tanh'``.

    The proposal interpolates ``tanh'`` at the lower endpoint, midpoint, and
    upper endpoint.  Its certificate is the classical interpolation remainder

    ``|f(x)-q(x)| <= sup_I |f'''| |(x-l)(x-m)(x-u)| / 3!``.

    For equally spaced nodes, the second factor has the exact maximum
    ``2*h**3/(3*sqrt(3))``, where ``h=(u-l)/2``.  Moreover
    ``f'''(x)=16*t-40*t**3+24*t**5`` with ``t=tanh(x)``; its extrema are found
    from the endpoints and the four closed-form roots of
    ``15*t**4-15*t**2+2=0``.  A fixed, non-adaptive Taylor-form pass then
    certifies and recenters the residual of the floating-point parabola.  No
    fitting iteration, optimization, root search, or adaptive refinement is
    used.
    """

    lower, upper = _scalar_interval_bounds(interval)
    midpoint = (lower + upper) / 2.0
    if lower == upper:
        value = 1.0 - tanh(lower) ** 2
        return QuadraticTanhEnclosure(
            coeffs=(value, 0.0, 0.0),
            delta=0.0,
            lower=lower,
            upper=upper,
            function="tanh_prime",
            metadata={
                "method": "point-interval",
                "midpoint": midpoint,
                "half_width": 0.0,
                "normalized_coeffs": (value, 0.0, 0.0),
                "third_derivative_candidates": (tanh(lower),),
                "third_derivative_sup": abs(
                    _tanh_prime_third_derivative_from_t(tanh(lower))
                ),
                "rounding_inflation": 0.0,
            },
        )

    half_width = (upper - lower) / 2.0
    values = tuple(1.0 - tanh(x) ** 2 for x in (lower, midpoint, upper))
    f_lower, f_midpoint, f_upper = values

    # q(x) = alpha*s**2 + beta*s + gamma, s=(x-midpoint)/half_width.
    alpha = (f_lower - 2.0 * f_midpoint + f_upper) / 2.0
    beta = (f_upper - f_lower) / 2.0
    gamma = f_midpoint
    a = alpha / (half_width * half_width)
    b = beta / half_width - 2.0 * a * midpoint
    c = gamma - beta * midpoint / half_width + a * midpoint * midpoint

    ta, tb = tanh(lower), tanh(upper)
    t_candidates = [ta, tb]
    root_disc = sqrt(105.0)
    for squared in ((15.0 - root_disc) / 30.0, (15.0 + root_disc) / 30.0):
        root = sqrt(squared)
        _append_if_in_t_interval(t_candidates, root, ta, tb)
        _append_if_in_t_interval(t_candidates, -root, ta, tb)
    third_derivative_sup = max(
        abs(_tanh_prime_third_derivative_from_t(t)) for t in t_candidates
    )
    global_interpolation_radius = (
        third_derivative_sup * half_width**3 / (9.0 * sqrt(3.0))
    )

    residual_lower, residual_upper, certificate_records = (
        _quadratic_residual_taylor_certificate(
            lower,
            upper,
            (c, b, a),
            certificate_subdivisions,
        )
    )
    residual_shift = (residual_upper + residual_lower) / 2.0
    c += residual_shift
    certificate_radius = (residual_upper - residual_lower) / 2.0

    # Account for ordinary floating-point construction/evaluation of the
    # normalized interpolant, consistently with the affine enclosure backend.
    scale = max(
        1.0,
        abs(alpha) + abs(beta) + abs(gamma),
        abs(global_interpolation_radius),
        abs(certificate_radius),
        third_derivative_sup,
    )
    rounding_inflation = nextafter(scale, inf) * 128.0 * 2.220446049250313e-16
    delta = nextafter(certificate_radius + rounding_inflation, inf)

    return QuadraticTanhEnclosure(
        coeffs=(c, b, a),
        delta=delta,
        lower=lower,
        upper=upper,
        function="tanh_prime",
        metadata={
            "method": "endpoint-midpoint-interpolation-remainder",
            "nodes": (lower, midpoint, upper),
            "values": values,
            "midpoint": midpoint,
            "half_width": half_width,
            "normalized_coeffs": (gamma + residual_shift, beta, alpha),
            "third_derivative_candidates": tuple(t_candidates),
            "third_derivative_sup": third_derivative_sup,
            "global_interpolation_radius": global_interpolation_radius,
            "certificate_subdivisions": certificate_subdivisions,
            "certificate_residual_lower": residual_lower,
            "certificate_residual_upper": residual_upper,
            "certificate_records": certificate_records,
            "residual_shift": residual_shift,
            "rounding_inflation": rounding_inflation,
            "outward_rounding": "final nextafter(delta + 128*eps*scale, +inf) safety inflation",
        },
    )


def _solve_dense_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense linear system by Gaussian elimination."""

    n = len(rhs)
    aug = [row[:] + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(aug[row][col]))
        if aug[pivot][col] == 0.0:
            raise ValueError("Singular interpolation system for tanh proposal.")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [value / scale for value in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(aug[row], aug[col])
                ]
    return [aug[row][-1] for row in range(n)]


def _chebyshev_interpolation_power_coeffs(
    lower: float, upper: float, degree: int
) -> tuple[float, ...]:
    """Return a Chebyshev-node interpolation proposal in power basis."""

    from math import cos, pi

    if degree == 0:
        return (tanh((lower + upper) / 2.0),)
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    nodes = [
        midpoint + half_width * cos((2 * k + 1) * pi / (2 * (degree + 1)))
        for k in range(degree + 1)
    ]
    vandermonde = [[node**power for power in range(degree + 1)] for node in nodes]
    values = [tanh(node) for node in nodes]
    return tuple(_solve_dense_system(vandermonde, values))


def _scalar_interval_bounds(
    interval: Interval | Sequence[float],
) -> tuple[float, float]:
    if isinstance(interval, Interval):
        lower, upper = interval.lower, interval.upper
    else:
        if len(interval) != 2:
            raise ValueError("interval must contain exactly two endpoints.")
        lower, upper = interval
    if isinstance(lower, tuple) or isinstance(upper, tuple):
        raise ValueError("tanh approximation currently expects a scalar interval.")
    lower_f = float(lower)
    upper_f = float(upper)
    if lower_f > upper_f:
        raise ValueError("interval lower endpoint must not exceed upper endpoint.")
    return lower_f, upper_f


def _poly_interval(coeffs: Sequence[float], interval: Interval) -> Interval:
    result = Interval.point(0.0)
    for coeff in reversed(tuple(float(c) for c in coeffs)):
        result = result * interval + coeff
    return result


def _tanh_interval(interval: Interval) -> Interval:
    lower, upper = _scalar_interval_bounds(interval)
    return Interval(nextafter(tanh(lower), -inf), nextafter(tanh(upper), inf))


def compute_tanh_polynomial(
    interval: Interval | Sequence[float],
    degree: int | None = None,
    *,
    chebyshev_degree: int | None = None,
    remez_degree: int | None = None,
    subdivisions: int = 64,
) -> TanhApproximation:
    """Compute and certify a polynomial approximation to ``tanh``.

    The current first-pass proposal uses a Chebyshev least-squares/interpolatory
    fit converted to the power basis (a stable near-minimax-style starting
    point, not a proof).  The public option is ``chebyshev_degree`` to reflect
    that implemented proposal method.  ``remez_degree`` is accepted only as a
    temporary deprecated alias and, when used, is recorded in metadata as
    ``legacy_remez_degree``.  Regardless of how the proposal is produced, the
    returned ``delta`` is always obtained from
    ``certify_tanh_residual_subdivision``.
    """

    candidates = [
        ("degree", degree),
        ("chebyshev_degree", chebyshev_degree),
        ("remez_degree", remez_degree),
    ]
    configured = [(name, int(value)) for name, value in candidates if value is not None]
    if not configured:
        raise TypeError("Either degree or chebyshev_degree must be supplied.")
    if len({value for _, value in configured}) != 1:
        raise ValueError(
            "degree, chebyshev_degree, and remez_degree must agree when supplied together."
        )
    configured_degree = configured[0][1]
    used_legacy_remez = remez_degree is not None
    if used_legacy_remez:
        warn(
            "remez_degree is deprecated; use chebyshev_degree because the implemented tanh proposal uses Chebyshev.fit.",
            DeprecationWarning,
            stacklevel=2,
        )
    if configured_degree < 0:
        raise ValueError("degree must be non-negative.")
    lower, upper = _scalar_interval_bounds(interval)

    if lower == upper:
        coeffs = (tanh(lower),) + (0.0,) * configured_degree
        proposal = "constant-point"
    else:
        try:
            import numpy as np
            from numpy.polynomial import Chebyshev, Polynomial
        except ImportError:
            # Chebyshev-node interpolation is a stable numerical proposal even
            # when NumPy is unavailable.  Certification below still provides
            # the proof rather than trusting this fit.
            coeffs = _chebyshev_interpolation_power_coeffs(
                lower, upper, configured_degree
            )
            proposal = "chebyshev-interpolation-proposal"
        else:
            xs = np.linspace(lower, upper, max(2 * (configured_degree + 1), 32))
            cheb = Chebyshev.fit(
                xs, np.tanh(xs), deg=configured_degree, domain=[lower, upper]
            )
            power: Polynomial = cheb.convert(kind=Polynomial)
            coeff_arr = np.asarray(power.coef, dtype=float)
            if coeff_arr.size < configured_degree + 1:
                coeff_arr = np.pad(
                    coeff_arr, (0, configured_degree + 1 - coeff_arr.size)
                )
            coeffs = tuple(float(c) for c in coeff_arr[: configured_degree + 1])
            proposal = "chebyshev-fit-proposal"

    delta, cert_meta = certify_tanh_residual_subdivision(
        (lower, upper), coeffs, subdivisions=subdivisions
    )
    metadata = {
        "chebyshev_degree": configured_degree,
        "proposal": proposal,
        "residual_certification": cert_meta,
        "proof_note": "Numerical fit is not a proof; delta is certified by interval subdivision.",
    }
    if used_legacy_remez:
        metadata["legacy_remez_degree"] = configured_degree
    return TanhApproximation(
        coeffs=tuple(float(c) for c in coeffs),
        lower=lower,
        upper=upper,
        delta=delta,
        degree=configured_degree,
        metadata=metadata,
    )


def certify_tanh_residual_subdivision(
    interval: Interval | Sequence[float],
    coeffs: Sequence[float],
    subdivisions: int = 64,
) -> tuple[float, dict[str, Any]]:
    """Certify a conservative residual bound for ``tanh(x) - p(x)``.

    This routine subdivides the scalar domain, evaluates ``tanh(I) - p(I)``
    using the repository's outward-rounded interval arithmetic, and returns the
    upward-rounded maximum absolute interval residual.  It is a rigorous,
    conservative fallback for the first implementation pass, not the final
    root-isolation implementation envisioned by the blueprint.
    """

    lower, upper = _scalar_interval_bounds(interval)
    if subdivisions <= 0:
        raise ValueError("subdivisions must be positive.")
    if not coeffs:
        raise ValueError("coeffs must not be empty.")

    width = (upper - lower) / subdivisions
    max_abs = 0.0
    worst_index = 0
    for idx in range(subdivisions):
        sub_lower = lower + idx * width
        sub_upper = upper if idx == subdivisions - 1 else lower + (idx + 1) * width
        sub_interval = Interval(nextafter(sub_lower, -inf), nextafter(sub_upper, inf))
        residual = _tanh_interval(sub_interval) - _poly_interval(coeffs, sub_interval)
        rlo, rhi = _scalar_interval_bounds(residual)
        local = max(abs(rlo), abs(rhi))
        if local > max_abs:
            max_abs = local
            worst_index = idx

    delta = nextafter(max_abs, inf)
    return delta, {
        "method": "outward-rounded-subdivision",
        "subdivisions": subdivisions,
        "worst_subdivision": worst_index,
        "interval": (lower, upper),
        "note": "Rigorous conservative fallback; not final root-isolation certification.",
    }


def _scalar_interval_from_enclosure(enclosure: Any) -> Interval:
    """Return a scalar ``Interval`` from a scalar PZ interval enclosure."""

    lower = enclosure.lower
    upper = enclosure.upper
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None and isinstance(lower, torch.Tensor):
        if lower.numel() != 1 or upper.numel() != 1:
            raise ValueError("tanh_pz_scalar expects a scalar polynomial zonotope.")
        return Interval(
            float(lower.reshape(()).item()), float(upper.reshape(()).item())
        )
    if isinstance(lower, tuple) or isinstance(upper, tuple):
        raise ValueError("tanh_pz_scalar expects a scalar polynomial zonotope.")
    return Interval(float(lower), float(upper))


def tanh_pz_scalar(
    Z_i: Any,
    chebyshev_degree: int | None = None,
    residual_subdivisions: int | None = None,
    *,
    remez_degree: int | None = None,
):
    """Enclose ``tanh(Z_i)`` for a scalar polynomial zonotope.

    The returned zonotope is ``p_i(Z_i) + Delta_i * eta_i`` where ``p_i`` is a
    numerically proposed polynomial and ``Delta_i`` is certified by subdivision
    interval arithmetic.  ``remez_degree`` is accepted only as a temporary
    deprecated alias for ``chebyshev_degree``.
    """

    from .polynomial_zonotope import PolynomialZonotope

    if not isinstance(Z_i, PolynomialZonotope):
        raise TypeError("Z_i must be a PolynomialZonotope.")
    if Z_i.shape != ():
        raise ValueError("tanh_pz_scalar expects a scalar polynomial zonotope.")
    if chebyshev_degree is None and remez_degree is None:
        raise TypeError("chebyshev_degree must be supplied.")
    if (
        chebyshev_degree is not None
        and remez_degree is not None
        and int(chebyshev_degree) != int(remez_degree)
    ):
        raise ValueError(
            "chebyshev_degree and remez_degree must agree when both are supplied."
        )
    if residual_subdivisions is None:
        raise TypeError("residual_subdivisions must be supplied.")
    interval = _scalar_interval_from_enclosure(Z_i.interval_enclosure())
    approx = compute_tanh_polynomial(
        interval,
        chebyshev_degree=chebyshev_degree,
        remez_degree=remez_degree,
        subdivisions=residual_subdivisions,
    )
    return Z_i.evaluate_polynomial(approx.coeffs).add_independent_error(
        approx.delta, kind="approximation_pointwise"
    )
