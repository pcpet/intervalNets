from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter, prod
from typing import Any, Mapping, Sequence

from .interval import Interval

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover
    torch = None

Exponent = tuple[int, ...]


def box_monomial_moment(exponent: tuple[int, ...]) -> float:
    """Exact integral of a monomial over the box ``[-1, 1]^d``.

    The returned value is ``integral over [-1,1]^d of x**exponent``. Odd
    monomials cancel by symmetry.
    """

    if any(k % 2 for k in exponent):
        return 0.0
    return prod(2.0 / (k + 1) for k in exponent)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _as_tensor(value: Any, *, dtype=None, device=None):
    if torch is None:
        raise ImportError("PyTorch is required for tensor polynomial zonotopes.")
    if isinstance(value, torch.Tensor):
        out = value
        if dtype is not None or device is not None:
            out = out.to(dtype=dtype or out.dtype, device=device or out.device)
        return out
    return torch.as_tensor(value, dtype=dtype or torch.get_default_dtype(), device=device)


def _to_fallback(value: Any):
    if _is_sequence(value):
        return tuple(_to_fallback(item) for item in value)
    return float(value)


def _fallback_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        if not value:
            return (0,)
        return (len(value),) + _fallback_shape(value[0])
    return ()


def _fallback_map(value: Any, op):
    if isinstance(value, tuple):
        return tuple(_fallback_map(item, op) for item in value)
    return op(value)


def _fallback_zip(left: Any, right: Any, op):
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("Shape mismatch.")
        return tuple(_fallback_zip(l, r, op) for l, r in zip(left, right))
    if isinstance(left, tuple) or isinstance(right, tuple):
        raise ValueError("Shape mismatch.")
    return op(left, right)


def _fallback_linear_contract(matrix: Any, coeff: Any):
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    if not isinstance(coeff, tuple):
        raise ValueError("linear_map expects coefficients with a leading input axis.")
    if any(len(row) != len(coeff) for row in rows):
        raise ValueError("Linear map weight/input dimension mismatch.")
    outputs = []
    for row in rows:
        acc = None
        for weight, item in zip(row, coeff):
            term = _mul_coeff(item, weight)
            acc = term if acc is None else _add_coeff(acc, term)
        outputs.append(acc if acc is not None else 0.0)
    return tuple(outputs)


def _zero_like(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return _fallback_map(value, lambda _: 0.0)


def _add_coeff(left: Any, right: Any):
    if torch is not None and isinstance(left, torch.Tensor):
        return left + right
    return _fallback_zip(left, right, lambda l, r: l + r)


def _mul_coeff(left: Any, right: Any):
    if torch is not None and isinstance(left, torch.Tensor):
        return left * right
    if torch is not None and isinstance(right, torch.Tensor):
        return left * right
    if isinstance(left, tuple) and not isinstance(right, tuple):
        return _fallback_map(left, lambda item: item * float(right))
    if isinstance(right, tuple) and not isinstance(left, tuple):
        return _fallback_map(right, lambda item: float(left) * item)
    return _fallback_zip(left, right, lambda l, r: l * r)


def _abs_coeff(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.abs(value)
    return _fallback_map(value, abs)


def _pad_lower(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.nextafter(value, torch.full_like(value, float("-inf")))
    return _fallback_map(value, lambda item: nextafter(float(item), -inf))


def _pad_upper(value: Any):
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.nextafter(value, torch.full_like(value, float("inf")))
    return _fallback_map(value, lambda item: nextafter(float(item), inf))


def _canonical_exponent(exponent: tuple[int, ...], num_noise: int) -> Exponent:
    if len(exponent) > num_noise:
        raise ValueError("Exponent length exceeds num_noise.")
    padded = tuple(int(item) for item in exponent) + (0,) * (num_noise - len(exponent))
    if any(item < 0 for item in padded):
        raise ValueError("Exponents must be non-negative.")
    return padded



def _canonical_noise_kinds(noise_kinds: tuple[str, ...] | list[str] | None, num_noise: int) -> tuple[str, ...]:
    if num_noise < 0:
        raise ValueError("num_noise must be non-negative.")
    if noise_kinds is None:
        return ("unknown",) * num_noise
    kinds = tuple(str(kind) for kind in noise_kinds)
    if len(kinds) != num_noise:
        raise ValueError("noise_kinds length must match num_noise.")
    return kinds


def _merge_noise_kinds(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    if len(left) != len(right):
        raise ValueError("noise_kinds length mismatch.")
    merged = []
    for l_kind, r_kind in zip(left, right):
        if l_kind == r_kind:
            merged.append(l_kind)
        elif l_kind == "unknown":
            merged.append(r_kind)
        elif r_kind == "unknown":
            merged.append(l_kind)
        else:
            raise ValueError(f"Incompatible noise metadata: {l_kind!r} != {r_kind!r}.")
    return tuple(merged)

@dataclass(frozen=True, init=False)
class PolynomialZonotope:
    """Polynomial zonotope with explicit monomial dependencies.

    Represents ``center + sum(terms[alpha] * eps**alpha)`` for
    ``eps_i in [-1, 1]``. Coefficients may be torch tensors or the lightweight
    tuple/float fallback used by the interval module.
    """

    center: Any
    terms: dict[Exponent, Any]
    num_noise: int
    shape: tuple[int, ...]
    dtype: Any
    device: Any
    noise_kinds: tuple[str, ...]

    def __init__(self, center: Any, terms: Mapping[tuple[int, ...], Any] | None = None, num_noise: int | None = None, noise_kinds: tuple[str, ...] | list[str] | None = None):
        use_torch = torch is not None and (isinstance(center, torch.Tensor) or any(isinstance(v, torch.Tensor) for v in (terms or {}).values()))
        c = _as_tensor(center) if use_torch else _to_fallback(center)
        inferred_noise = max((len(exp) for exp in (terms or {})), default=0)
        p = inferred_noise if num_noise is None else int(num_noise)
        if p < inferred_noise:
            raise ValueError("num_noise is smaller than a supplied exponent length.")
        kinds = _canonical_noise_kinds(noise_kinds, p)
        clean: dict[Exponent, Any] = {}
        for exp, coeff in (terms or {}).items():
            key = _canonical_exponent(tuple(exp), p)
            value = _as_tensor(coeff, dtype=c.dtype, device=c.device) if torch is not None and isinstance(c, torch.Tensor) else _to_fallback(coeff)
            if (torch is not None and isinstance(c, torch.Tensor) and tuple(value.shape) != tuple(c.shape)) or (not (torch is not None and isinstance(c, torch.Tensor)) and _fallback_shape(value) != _fallback_shape(c)):
                raise ValueError("Term coefficient shape must match center shape.")
            clean[key] = _add_coeff(clean[key], value) if key in clean else value
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "terms", clean)
        object.__setattr__(self, "num_noise", p)
        object.__setattr__(self, "shape", tuple(c.shape) if torch is not None and isinstance(c, torch.Tensor) else _fallback_shape(c))
        object.__setattr__(self, "dtype", c.dtype if torch is not None and isinstance(c, torch.Tensor) else float)
        object.__setattr__(self, "device", c.device if torch is not None and isinstance(c, torch.Tensor) else None)
        object.__setattr__(self, "noise_kinds", kinds)

    @classmethod
    def constant(cls, value: Any, num_noise: int = 0, noise_kinds: tuple[str, ...] | list[str] | None = None) -> "PolynomialZonotope":
        return cls(value, {}, num_noise=num_noise, noise_kinds=noise_kinds)

    @classmethod
    def from_box(cls, lower: Any, upper: Any) -> "PolynomialZonotope":
        if torch is not None and (isinstance(lower, torch.Tensor) or isinstance(upper, torch.Tensor)):
            lo = _as_tensor(lower)
            hi = _as_tensor(upper, dtype=lo.dtype, device=lo.device)
            if lo.shape != hi.shape:
                raise ValueError("Lower/upper shape mismatch.")
            if torch.any(lo > hi):
                raise ValueError("Lower bounds must not exceed upper bounds.")
            center = (lo + hi) / 2
            radius = (hi - lo) / 2
            p = radius.numel()
            terms = {}
            for idx in range(p):
                coeff = torch.zeros_like(center)
                coeff.reshape(-1)[idx] = radius.reshape(-1)[idx]
                exp = [0] * p
                exp[idx] = 1
                terms[tuple(exp)] = coeff
            return cls(center, terms, num_noise=p, noise_kinds=("domain",) * p)
        lo = _to_fallback(lower); hi = _to_fallback(upper)
        def check(l, h):
            if isinstance(l, tuple):
                if len(l) != len(h): raise ValueError("Lower/upper shape mismatch.")
                for a, b in zip(l, h): check(a, b)
            elif l > h: raise ValueError("Lower bounds must not exceed upper bounds.")
        check(lo, hi)
        center = _fallback_zip(lo, hi, lambda l, h: (l + h) / 2.0)
        radius = _fallback_zip(lo, hi, lambda l, h: (h - l) / 2.0)
        flat_paths: list[tuple[int, ...]] = []
        def paths(v, prefix=()):
            if isinstance(v, tuple):
                for i, item in enumerate(v): paths(item, prefix + (i,))
            else: flat_paths.append(prefix)
        paths(radius)
        def coeff_for(path):
            def rec(v, pref=()):
                if isinstance(v, tuple): return tuple(rec(item, pref + (i,)) for i, item in enumerate(v))
                return v if pref == path else 0.0
            return rec(radius)
        terms = {}
        for i, path in enumerate(flat_paths):
            exp = [0] * len(flat_paths); exp[i] = 1
            terms[tuple(exp)] = coeff_for(path)
        return cls(center, terms, num_noise=len(flat_paths), noise_kinds=("domain",) * len(flat_paths))


    def _align(self, other: "PolynomialZonotope"):
        p = max(self.num_noise, other.num_noise)
        left = self.with_num_noise(p)
        right = other.with_num_noise(p)
        merged = _merge_noise_kinds(left.noise_kinds, right.noise_kinds)
        return left.with_noise_kinds(merged), right.with_noise_kinds(merged)

    def with_noise_kinds(self, noise_kinds: tuple[str, ...] | list[str]) -> "PolynomialZonotope":
        kinds = _canonical_noise_kinds(noise_kinds, self.num_noise)
        if kinds == self.noise_kinds: return self
        return PolynomialZonotope(self.center, self.terms, num_noise=self.num_noise, noise_kinds=kinds)

    def with_num_noise(self, num_noise: int) -> "PolynomialZonotope":
        if num_noise < self.num_noise:
            raise ValueError("num_noise cannot shrink existing exponents.")
        if num_noise == self.num_noise: return self
        return PolynomialZonotope(self.center, {exp + (0,) * (num_noise - self.num_noise): c for exp, c in self.terms.items()}, num_noise=num_noise, noise_kinds=self.noise_kinds + ("unknown",) * (num_noise - self.num_noise))

    def __add__(self, other: Any):
        if not isinstance(other, PolynomialZonotope):
            return PolynomialZonotope(_add_coeff(self.center, other), self.terms, num_noise=self.num_noise, noise_kinds=self.noise_kinds)
        left, right = self._align(other)
        if left.shape != right.shape: raise ValueError("Shape mismatch for addition.")
        terms = dict(left.terms)
        for exp, coeff in right.terms.items(): terms[exp] = _add_coeff(terms[exp], coeff) if exp in terms else coeff
        return PolynomialZonotope(_add_coeff(left.center, right.center), terms, num_noise=left.num_noise, noise_kinds=left.noise_kinds)

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other: Any):
        return self + (-other if isinstance(other, PolynomialZonotope) else -float(other))

    def __rsub__(self, other: Any):
        return (-self) + other

    def __mul__(self, other: Any):
        if not isinstance(other, PolynomialZonotope):
            return PolynomialZonotope(_mul_coeff(self.center, other), {e: _mul_coeff(c, other) for e, c in self.terms.items()}, num_noise=self.num_noise, noise_kinds=self.noise_kinds)
        left, right = self._align(other)
        if left.shape != () and right.shape != () and left.shape != right.shape:
            raise ValueError("Polynomial-zonotope multiplication requires at least one scalar coefficient shape or equal shapes.")
        terms: dict[Exponent, Any] = {}
        def add(exp, coeff): terms.__setitem__(exp, _add_coeff(terms[exp], coeff) if exp in terms else coeff)
        for exp, coeff in right.terms.items(): add(exp, _mul_coeff(left.center, coeff))
        for exp, coeff in left.terms.items(): add(exp, _mul_coeff(coeff, right.center))
        for e1, c1 in left.terms.items():
            for e2, c2 in right.terms.items(): add(tuple(a + b for a, b in zip(e1, e2)), _mul_coeff(c1, c2))
        return PolynomialZonotope(_mul_coeff(left.center, right.center), terms, num_noise=left.num_noise, noise_kinds=left.noise_kinds)

    __rmul__ = __mul__


    def evaluate_polynomial(self, coeffs: Any) -> "PolynomialZonotope":
        """Evaluate a scalar power-basis polynomial on this zonotope.

        ``coeffs`` are in ascending power order: ``c0, c1, ...``.  The
        implementation uses Horner evaluation and preserves all existing
        polynomial dependencies.
        """

        coeff_tuple = tuple(coeffs)
        if not coeff_tuple:
            raise ValueError("coeffs must not be empty.")
        result = PolynomialZonotope.constant(coeff_tuple[-1], num_noise=self.num_noise, noise_kinds=self.noise_kinds)
        for coeff in reversed(coeff_tuple[:-1]):
            result = result * self + coeff
        return result

    def add_independent_error(self, radius: Any, target_shape: tuple[int, ...] = (), *, kind: str = "approximation", metadata: str | None = None) -> "PolynomialZonotope":
        """Add a fresh independent error variable with the given radius.

        Existing exponent vectors are extended by one zero entry, while the new
        error term receives exponent ``(0, ..., 0, 1)``.  For tensor-backed
        zonotopes, ``target_shape`` may be supplied to create a coefficient of
        that shape; it must match the zonotope shape so the resulting object is
        well-formed.
        """

        new_noise = self.num_noise + 1
        terms = {exp + (0,): coeff for exp, coeff in self.terms.items()}
        if torch is not None and isinstance(self.center, torch.Tensor):
            shape = tuple(target_shape) if target_shape else self.shape
            if shape != self.shape:
                raise ValueError("target_shape must match this zonotope's coefficient shape.")
            coeff = torch.as_tensor(radius, dtype=self.center.dtype, device=self.center.device)
            if tuple(coeff.shape) == () and self.shape != ():
                coeff = torch.full_like(self.center, float(coeff.item()))
            else:
                coeff = coeff.to(dtype=self.center.dtype, device=self.center.device)
                if tuple(coeff.shape) != self.shape:
                    coeff = torch.broadcast_to(coeff, self.shape).clone()
        else:
            if target_shape and tuple(target_shape) != self.shape:
                raise ValueError("target_shape must match this zonotope's coefficient shape.")
            coeff = _mul_coeff(_zero_like(self.center), 0.0)
            coeff = _add_coeff(coeff, _to_fallback(radius)) if self.shape == () else _fallback_map(self.center, lambda _: float(radius))
        terms[(0,) * self.num_noise + (1,)] = coeff
        return PolynomialZonotope(self.center, terms, num_noise=new_noise, noise_kinds=self.noise_kinds + (str(metadata) if metadata is not None else str(kind),))


    def integrate_noise(self, noise_indices: Sequence[int]) -> "PolynomialZonotope":
        """Integrate selected noise variables coefficient-by-coefficient.

        For each monomial term, variables in ``noise_indices`` are integrated
        exactly over ``[-1, 1]`` using :func:`box_monomial_moment`. Variables
        not listed are retained, and terms with identical retained exponents are
        merged. Odd integrated exponents have zero moment and are dropped.
        """

        indices = tuple(int(index) for index in noise_indices)
        if len(set(indices)) != len(indices):
            raise ValueError("noise_indices must not contain duplicates.")
        if any(index < 0 or index >= self.num_noise for index in indices):
            raise ValueError("noise index out of range.")

        integrated = set(indices)
        retained_indices = tuple(index for index in range(self.num_noise) if index not in integrated)
        retained_kinds = tuple(self.noise_kinds[index] for index in retained_indices)
        zero_retained = (0,) * len(retained_indices)

        center = self.center
        terms: dict[Exponent, Any] = {}
        for exponent, coeff in self.terms.items():
            integrated_exponent = tuple(exponent[index] for index in indices)
            moment = box_monomial_moment(integrated_exponent)
            if moment == 0.0:
                continue
            retained_exponent = tuple(exponent[index] for index in retained_indices)
            integrated_coeff = _mul_coeff(coeff, moment)
            if retained_exponent == zero_retained:
                center = _add_coeff(center, integrated_coeff)
            else:
                terms[retained_exponent] = _add_coeff(terms[retained_exponent], integrated_coeff) if retained_exponent in terms else integrated_coeff

        return PolynomialZonotope(center, terms, num_noise=len(retained_indices), noise_kinds=retained_kinds)

    def integrate_domain_noise(self) -> "PolynomialZonotope":
        """Integrate all noise variables labeled ``"domain"``."""

        return self.integrate_noise([index for index, kind in enumerate(self.noise_kinds) if kind == "domain"])

    def integrate_domain(self, domain_indices: Sequence[int] | None = None, *, mode: str = "pointwise_interval", volume: float | None = None):
        """Integrate domain variables with pointwise-residual-safe semantics.

        This delegates to :func:`intervalnets.pz_integration.integrate_pz_over_domain`
        and returns an ``IntegratedPZResult`` that separates the exact retained
        polynomial from the scalar/tensor interval radius accumulated from
        pointwise approximation residuals.  Use ``mode="symbolic"`` only when
        approximation variables are intended to represent global symbolic
        uncertainties whose moments may be preserved.
        """

        from .pz_integration import integrate_pz_over_domain

        return integrate_pz_over_domain(self, domain_indices, mode=mode, volume=volume)

    def linear_map(self, matrix: Any, bias: Any | None = None) -> "PolynomialZonotope":
        """Apply a linear map along the leading coefficient axis.

        For a weight matrix ``A`` with shape ``(m, n)``, coefficients with
        shape ``(n,)``, ``(n, d_in)``, or ``(n, d_in, d_in)`` are mapped to
        ``(m,)``, ``(m, d_in)``, or ``(m, d_in, d_in)`` by contracting over
        the leading/output axis. ``bias`` is added to the center only.
        """

        if torch is not None and isinstance(self.center, torch.Tensor):
            weight = _as_tensor(matrix, dtype=self.center.dtype, device=self.center.device)
            if weight.ndim != 2:
                raise ValueError("linear_map weight must be a 2-D matrix.")
            if self.center.ndim < 1 or self.center.shape[0] != weight.shape[1]:
                raise ValueError("Linear map weight/input dimension mismatch.")

            def apply(coeff: Any):
                return torch.einsum("ij,j...->i...", weight, coeff)

            center = apply(self.center)
            if bias is not None:
                center = center + _as_tensor(bias, dtype=self.center.dtype, device=self.center.device)
            return PolynomialZonotope(center, {exp: apply(coeff) for exp, coeff in self.terms.items()}, num_noise=self.num_noise, noise_kinds=self.noise_kinds)

        mapped_center = _fallback_linear_contract(matrix, self.center)
        if bias is not None:
            mapped_center = _add_coeff(mapped_center, _to_fallback(bias))
        return PolynomialZonotope(
            mapped_center,
            {exp: _fallback_linear_contract(matrix, coeff) for exp, coeff in self.terms.items()},
            num_noise=self.num_noise,
            noise_kinds=self.noise_kinds,
        )

    def tensor_product(self, other: "PolynomialZonotope") -> "PolynomialZonotope":
        if torch is None or not isinstance(self.center, torch.Tensor) or not isinstance(other.center, torch.Tensor):
            raise NotImplementedError("tensor_product currently requires torch-backed coefficients.")
        left, right = self._align(other)
        def outer(a, b): return torch.einsum("...,...->...", a, b) if a.ndim == b.ndim == 0 else torch.outer(a.reshape(-1), b.reshape(-1)).reshape(*a.shape, *b.shape)
        terms: dict[Exponent, Any] = {}
        def add(exp, coeff): terms.__setitem__(exp, terms[exp] + coeff if exp in terms else coeff)
        for exp, coeff in right.terms.items(): add(exp, outer(left.center, coeff))
        for exp, coeff in left.terms.items(): add(exp, outer(coeff, right.center))
        for e1, c1 in left.terms.items():
            for e2, c2 in right.terms.items(): add(tuple(a + b for a, b in zip(e1, e2)), outer(c1, c2))
        return PolynomialZonotope(outer(left.center, right.center), terms, num_noise=left.num_noise, noise_kinds=left.noise_kinds)

    def __getitem__(self, item: Any) -> "PolynomialZonotope":
        if torch is not None and isinstance(self.center, torch.Tensor):
            return PolynomialZonotope(self.center[item], {e: c[item] for e, c in self.terms.items()}, num_noise=self.num_noise, noise_kinds=self.noise_kinds)
        return PolynomialZonotope(self.center[item], {e: c[item] for e, c in self.terms.items()}, num_noise=self.num_noise, noise_kinds=self.noise_kinds)

    @staticmethod
    def stack(items: list["PolynomialZonotope"] | tuple["PolynomialZonotope", ...], dim: int = 0) -> "PolynomialZonotope":
        if not items: raise ValueError("stack requires at least one item.")
        p = max(item.num_noise for item in items)
        aligned = [item.with_num_noise(p) for item in items]
        merged_kinds = aligned[0].noise_kinds
        for item in aligned[1:]:
            merged_kinds = _merge_noise_kinds(merged_kinds, item.noise_kinds)
        aligned = [item.with_noise_kinds(merged_kinds) for item in aligned]
        if torch is None or not isinstance(aligned[0].center, torch.Tensor):
            if dim != 0: raise NotImplementedError("fallback stack supports dim=0 only.")
            exps = set().union(*(item.terms.keys() for item in aligned))
            return PolynomialZonotope(tuple(item.center for item in aligned), {e: tuple(item.terms.get(e, _zero_like(item.center)) for item in aligned) for e in exps}, num_noise=p, noise_kinds=merged_kinds)
        exps = set().union(*(item.terms.keys() for item in aligned))
        return PolynomialZonotope(torch.stack([item.center for item in aligned], dim=dim), {e: torch.stack([item.terms.get(e, torch.zeros_like(item.center)) for item in aligned], dim=dim) for e in exps}, num_noise=p, noise_kinds=merged_kinds)

    def interval_enclosure(self):
        radius = _zero_like(self.center)
        for coeff in self.terms.values(): radius = _add_coeff(radius, _abs_coeff(coeff))
        lower = _pad_lower(_add_coeff(self.center, _mul_coeff(radius, -1.0)))
        upper = _pad_upper(_add_coeff(self.center, radius))
        try:
            from .pytorch import IntervalTensor
            if torch is not None and isinstance(lower, torch.Tensor):
                return IntervalTensor.from_bounds(lower, upper)
        except ImportError:  # pragma: no cover
            pass
        return Interval.from_bounds(lower, upper)


@dataclass(frozen=True)
class PZTwoJet:
    """Polynomial-zonotope value/Jacobian/Hessian two-jet.

    ``J`` and ``H`` are derivatives with respect to the physical input
    variable ``x``, not derivatives with respect to polynomial-zonotope noise
    variables.
    """

    Y: PolynomialZonotope
    J: PolynomialZonotope
    H: PolynomialZonotope

    @classmethod
    def from_input(cls, X: PolynomialZonotope, input_dim: int) -> "PZTwoJet":
        """Initialize the two-jet for an input polynomial zonotope.

        The value component is the input zonotope itself. The Jacobian is the
        constant identity with shape ``(input_dim, input_dim)`` and the Hessian
        is the constant zero tensor with shape
        ``(input_dim, input_dim, input_dim)``. Both constants use ``X``'s noise
        dimension so future propagation keeps dependencies aligned.
        """

        if torch is None:
            raise ImportError("PyTorch is required to initialize PZTwoJet constants.")
        if input_dim < 0:
            raise ValueError("input_dim must be non-negative.")
        kwargs = {}
        if isinstance(X.center, torch.Tensor):
            kwargs = {"dtype": X.center.dtype, "device": X.center.device}
        return cls(
            Y=X,
            J=PolynomialZonotope.constant(torch.eye(input_dim, **kwargs), num_noise=X.num_noise, noise_kinds=X.noise_kinds),
            H=PolynomialZonotope.constant(torch.zeros(input_dim, input_dim, input_dim, **kwargs), num_noise=X.num_noise, noise_kinds=X.noise_kinds),
        )
