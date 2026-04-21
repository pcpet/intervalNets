from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _to_scalar(value: Any) -> float:
    return float(value)


def _to_vector(value: Any) -> tuple[float, ...]:
    if not _is_sequence(value):
        raise ValueError("Expected a scalar or 1D sequence.")
    return tuple(float(item) for item in value)


def _pad_outward_scalar(value: float, direction: float) -> float:
    return nextafter(float(value), direction)


def _pad_outward_data(value: Any, direction: float):
    if _is_sequence(value):
        return tuple(_pad_outward_data(item, direction) for item in value)
    return _pad_outward_scalar(float(value), direction)


def _sum_abs_generators_fallback(generators: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
    if not generators:
        return tuple()
    return tuple(sum(abs(coef) for coef in row) for row in generators)


def _build_interval_generators_fallback(radius: float | tuple[float, ...]):
    if isinstance(radius, tuple):
        size = len(radius)
        return tuple(
            tuple(radius[i] if i == j else 0.0 for j in range(size))
            for i in range(size)
        )
    return (radius,)


def _flatten_noise_rows_fallback(generators, length: int):
    if isinstance(generators, tuple) and length == 0:
        return tuple()
    if length == 1 and all(not _is_sequence(item) for item in generators):
        return (tuple(float(item) for item in generators),)
    return tuple(tuple(float(item) for item in row) for row in generators)


@dataclass(frozen=True)
class AffineTensor:
    """Affine arithmetic container: x = c + G * eps, eps_i in [-1, 1].

    The last dimension of ``G`` indexes noise symbols.
    For vectors, ``c`` has shape ``(n,)`` and ``G`` has shape ``(n, k)``.
    """

    c: Any
    G: Any

    @classmethod
    def point(cls, value: Any) -> "AffineTensor":
        if torch is not None and isinstance(value, torch.Tensor):
            center = value
            generators = torch.zeros(*value.shape, 0, dtype=value.dtype, device=value.device)
            return cls(center, generators)

        if _is_sequence(value):
            center = _to_vector(value)
            generators = tuple(tuple() for _ in center)
            return cls(center, generators)

        center = _to_scalar(value)
        return cls(center, tuple())

    @classmethod
    def from_interval(cls, lower: Any, upper: Any) -> "AffineTensor":
        return cls.from_bounds(lower, upper)

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "AffineTensor":
        if torch is not None and isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
            if lower.shape != upper.shape:
                raise ValueError(f"Lower/upper shape mismatch: {tuple(lower.shape)} vs {tuple(upper.shape)}.")
            if torch.any(lower > upper):
                raise ValueError("Lower bounds must not exceed upper bounds.")
            center = (lower + upper) / 2
            radius = (upper - lower) / 2
            flat_radius = radius.reshape(-1)
            eye = torch.eye(flat_radius.numel(), dtype=flat_radius.dtype, device=flat_radius.device)
            generators = eye * flat_radius.unsqueeze(0)
            generators = generators.reshape(*radius.shape, flat_radius.numel())
            return cls(center, generators)

        if _is_sequence(lower) or _is_sequence(upper):
            lo = _to_vector(lower)
            hi = _to_vector(upper)
            if len(lo) != len(hi):
                raise ValueError(f"Lower/upper shape mismatch: {len(lo)} vs {len(hi)}.")
            if any(l_item > h_item for l_item, h_item in zip(lo, hi)):
                raise ValueError("Lower bounds must not exceed upper bounds.")
            center = tuple((l_item + h_item) / 2.0 for l_item, h_item in zip(lo, hi))
            radius = tuple((h_item - l_item) / 2.0 for l_item, h_item in zip(lo, hi))
            return cls(center, _build_interval_generators_fallback(radius))

        lo = float(lower)
        hi = float(upper)
        if lo > hi:
            raise ValueError("Lower bounds must not exceed upper bounds.")
        center = (lo + hi) / 2.0
        radius = (hi - lo) / 2.0
        return cls(center, _build_interval_generators_fallback(radius))

    def to_bounds(self):
        if torch is not None and isinstance(self.c, torch.Tensor):
            if not isinstance(self.G, torch.Tensor):
                raise ValueError("Expected torch generators for torch center.")
            if self.G.shape[:-1] != self.c.shape:
                raise ValueError(
                    f"Generator shape mismatch: center {tuple(self.c.shape)} vs generators {tuple(self.G.shape)}."
                )
            radius = torch.sum(torch.abs(self.G), dim=-1)
            lower_raw = self.c - radius
            upper_raw = self.c + radius
            lower = torch.nextafter(lower_raw, torch.full_like(lower_raw, float("-inf")))
            upper = torch.nextafter(upper_raw, torch.full_like(upper_raw, float("inf")))
            return lower, upper

        if isinstance(self.c, tuple):
            rows = _flatten_noise_rows_fallback(self.G, len(self.c))
            radius = _sum_abs_generators_fallback(rows)
            lower = _pad_outward_data(tuple(ci - ri for ci, ri in zip(self.c, radius)), -inf)
            upper = _pad_outward_data(tuple(ci + ri for ci, ri in zip(self.c, radius)), inf)
            return lower, upper

        radius = sum(abs(float(coef)) for coef in self.G)
        lower = _pad_outward_scalar(float(self.c) - radius, -inf)
        upper = _pad_outward_scalar(float(self.c) + radius, inf)
        return lower, upper

    def __add__(self, other: Any) -> "AffineTensor":
        if isinstance(other, AffineTensor):
            return self._add_affine(other)
        return self._add_scalar(float(other))

    def __radd__(self, other: Any) -> "AffineTensor":
        return self + other

    def __sub__(self, other: Any) -> "AffineTensor":
        if isinstance(other, AffineTensor):
            return self._sub_affine(other)
        return self._add_scalar(-float(other))

    def __rsub__(self, other: Any) -> "AffineTensor":
        return (-self) + other

    def __neg__(self) -> "AffineTensor":
        if torch is not None and isinstance(self.c, torch.Tensor):
            return AffineTensor(-self.c, -self.G)
        if isinstance(self.c, tuple):
            return AffineTensor(tuple(-item for item in self.c), tuple(tuple(-item for item in row) for row in self.G))
        return AffineTensor(-float(self.c), tuple(-item for item in self.G))

    def _add_scalar(self, scalar: float) -> "AffineTensor":
        if torch is not None and isinstance(self.c, torch.Tensor):
            return AffineTensor(self.c + scalar, self.G)
        if isinstance(self.c, tuple):
            return AffineTensor(tuple(item + scalar for item in self.c), self.G)
        return AffineTensor(float(self.c) + scalar, self.G)

    def _add_affine(self, other: "AffineTensor") -> "AffineTensor":
        if torch is not None and isinstance(self.c, torch.Tensor) and isinstance(other.c, torch.Tensor):
            if self.c.shape != other.c.shape:
                raise ValueError(f"Center shape mismatch: {tuple(self.c.shape)} vs {tuple(other.c.shape)}.")
            if self.G.shape[:-1] != self.c.shape or other.G.shape[:-1] != other.c.shape:
                raise ValueError("Generator tensor shape must be center shape plus noise dimension.")
            return AffineTensor(self.c + other.c, torch.cat((self.G, other.G), dim=-1))

        if isinstance(self.c, tuple) and isinstance(other.c, tuple):
            if len(self.c) != len(other.c):
                raise ValueError(f"Center shape mismatch: {len(self.c)} vs {len(other.c)}.")
            left_rows = _flatten_noise_rows_fallback(self.G, len(self.c))
            right_rows = _flatten_noise_rows_fallback(other.G, len(other.c))
            center = tuple(l + r for l, r in zip(self.c, other.c))
            generators = tuple(lrow + rrow for lrow, rrow in zip(left_rows, right_rows))
            return AffineTensor(center, generators)

        raise ValueError("Affine addition requires both operands to use compatible backends and shapes.")

    def _sub_affine(self, other: "AffineTensor") -> "AffineTensor":
        if torch is not None and isinstance(self.c, torch.Tensor) and isinstance(other.c, torch.Tensor):
            if self.c.shape != other.c.shape:
                raise ValueError(f"Center shape mismatch: {tuple(self.c.shape)} vs {tuple(other.c.shape)}.")
            if self.G.shape[:-1] != self.c.shape or other.G.shape[:-1] != other.c.shape:
                raise ValueError("Generator tensor shape must be center shape plus noise dimension.")
            return AffineTensor(self.c - other.c, torch.cat((self.G, -other.G), dim=-1))

        if isinstance(self.c, tuple) and isinstance(other.c, tuple):
            if len(self.c) != len(other.c):
                raise ValueError(f"Center shape mismatch: {len(self.c)} vs {len(other.c)}.")
            left_rows = _flatten_noise_rows_fallback(self.G, len(self.c))
            right_rows = _flatten_noise_rows_fallback(other.G, len(other.c))
            center = tuple(l - r for l, r in zip(self.c, other.c))
            generators = tuple(lrow + tuple(-item for item in rrow) for lrow, rrow in zip(left_rows, right_rows))
            return AffineTensor(center, generators)

        raise ValueError("Affine subtraction requires both operands to use compatible backends and shapes.")

    def affine_map(self, W: Any, b: Any | None = None) -> "AffineTensor":
        if torch is not None and isinstance(self.c, torch.Tensor):
            if not isinstance(W, torch.Tensor):
                raise ValueError("For torch AffineTensor, W must be a torch.Tensor.")
            if self.c.ndim != 1:
                raise ValueError(f"Affine map expects 1D center vector, got shape {tuple(self.c.shape)}.")
            if self.G.ndim != 2:
                raise ValueError(f"Affine map expects generator matrix of shape (n, k), got {tuple(self.G.shape)}.")
            if W.ndim != 2:
                raise ValueError(f"Affine map expects W with shape (m, n), got {tuple(W.shape)}.")
            in_features = self.c.shape[0]
            if W.shape[1] != in_features:
                raise ValueError(f"Dimension mismatch: W has {W.shape[1]} input features, center has {in_features}.")
            new_center = W @ self.c
            if b is not None:
                if not isinstance(b, torch.Tensor):
                    raise ValueError("For torch AffineTensor, bias b must be a torch.Tensor when provided.")
                if b.shape != new_center.shape:
                    raise ValueError(f"Bias shape mismatch: expected {tuple(new_center.shape)}, got {tuple(b.shape)}.")
                new_center = new_center + b
            new_generators = W @ self.G
            return AffineTensor(new_center, new_generators)

        center = _to_vector(self.c)
        generators = _flatten_noise_rows_fallback(self.G, len(center))
        if not _is_sequence(W):
            raise ValueError("Affine map expects W as a 2D sequence for fallback backend.")
        rows = tuple(_to_vector(row) for row in W)
        if rows and any(len(row) != len(center) for row in rows):
            raise ValueError(
                f"Dimension mismatch: every row of W must have length {len(center)} for center shape {(len(center),)}."
            )

        new_center = tuple(sum(weight * value for weight, value in zip(row, center)) for row in rows)
        if b is not None:
            bias = _to_vector(b)
            if len(bias) != len(new_center):
                raise ValueError(f"Bias shape mismatch: expected length {len(new_center)}, got {len(bias)}.")
            new_center = tuple(value + bias_item for value, bias_item in zip(new_center, bias))

        noise_count = len(generators[0]) if generators else 0
        new_generators = tuple(
            tuple(sum(weight * generators[col][noise] for col, weight in enumerate(row)) for noise in range(noise_count))
            for row in rows
        )
        return AffineTensor(new_center, new_generators)
