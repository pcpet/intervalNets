"""Certified polynomial approximation scaffolding for ``tanh`` on scalar intervals.

The polynomial fit produced here is only a numerical proposal.  It is never
used as proof: callers must rely on ``certify_tanh_residual_subdivision`` (or a
future root-isolation/Remez certificate backend) for the rigorous residual
``delta``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf, nextafter, tanh
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
                aug[row] = [value - factor * pivot_value for value, pivot_value in zip(aug[row], aug[col])]
    return [aug[row][-1] for row in range(n)]


def _chebyshev_interpolation_power_coeffs(lower: float, upper: float, degree: int) -> tuple[float, ...]:
    """Return a Chebyshev-node interpolation proposal in power basis."""

    from math import cos, pi

    if degree == 0:
        return (tanh((lower + upper) / 2.0),)
    midpoint = (lower + upper) / 2.0
    half_width = (upper - lower) / 2.0
    nodes = [midpoint + half_width * cos((2 * k + 1) * pi / (2 * (degree + 1))) for k in range(degree + 1)]
    vandermonde = [[node**power for power in range(degree + 1)] for node in nodes]
    values = [tanh(node) for node in nodes]
    return tuple(_solve_dense_system(vandermonde, values))


def _scalar_interval_bounds(interval: Interval | Sequence[float]) -> tuple[float, float]:
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
        raise ValueError("degree, chebyshev_degree, and remez_degree must agree when supplied together.")
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
            coeffs = _chebyshev_interpolation_power_coeffs(lower, upper, configured_degree)
            proposal = "chebyshev-interpolation-proposal"
        else:
            xs = np.linspace(lower, upper, max(2 * (configured_degree + 1), 32))
            cheb = Chebyshev.fit(xs, np.tanh(xs), deg=configured_degree, domain=[lower, upper])
            power: Polynomial = cheb.convert(kind=Polynomial)
            coeff_arr = np.asarray(power.coef, dtype=float)
            if coeff_arr.size < configured_degree + 1:
                coeff_arr = np.pad(coeff_arr, (0, configured_degree + 1 - coeff_arr.size))
            coeffs = tuple(float(c) for c in coeff_arr[: configured_degree + 1])
            proposal = "chebyshev-fit-proposal"

    delta, cert_meta = certify_tanh_residual_subdivision((lower, upper), coeffs, subdivisions=subdivisions)
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
        return Interval(float(lower.reshape(()).item()), float(upper.reshape(()).item()))
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
    if chebyshev_degree is not None and remez_degree is not None and int(chebyshev_degree) != int(remez_degree):
        raise ValueError("chebyshev_degree and remez_degree must agree when both are supplied.")
    if residual_subdivisions is None:
        raise TypeError("residual_subdivisions must be supplied.")
    interval = _scalar_interval_from_enclosure(Z_i.interval_enclosure())
    approx = compute_tanh_polynomial(
        interval,
        chebyshev_degree=chebyshev_degree,
        remez_degree=remez_degree,
        subdivisions=residual_subdivisions,
    )
    return Z_i.evaluate_polynomial(approx.coeffs).add_independent_error(approx.delta, kind="approximation_pointwise")
