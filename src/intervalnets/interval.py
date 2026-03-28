from __future__ import annotations

from dataclasses import dataclass
from math import inf, nextafter
from typing import Any

Scalar = float
Data = Scalar | tuple["Data", ...]


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _to_data(value: Any) -> Data:
    if _is_sequence(value):
        return tuple(_to_data(item) for item in value)
    return float(value)


def _map_unary(value: Data, op) -> Data:
    if isinstance(value, tuple):
        return tuple(_map_unary(item, op) for item in value)
    return op(value)


def _map_binary(left: Data, right: Data, op) -> Data:
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ValueError("Mismatched interval shapes.")
        return tuple(_map_binary(l_item, r_item, op) for l_item, r_item in zip(left, right))
    if isinstance(left, tuple) or isinstance(right, tuple):
        raise ValueError("Mismatched interval shapes.")
    return op(left, right)


def outward_lower(value: Any) -> Data:
    return _map_unary(_to_data(value), lambda item: nextafter(item, -inf))


def outward_upper(value: Any) -> Data:
    return _map_unary(_to_data(value), lambda item: nextafter(item, inf))


def _validate_bounds(lower: Data, upper: Data) -> None:
    if isinstance(lower, tuple) and isinstance(upper, tuple):
        if len(lower) != len(upper):
            raise ValueError("Lower and upper bounds must have identical shapes.")
        for lower_item, upper_item in zip(lower, upper):
            _validate_bounds(lower_item, upper_item)
        return
    if isinstance(lower, tuple) or isinstance(upper, tuple):
        raise ValueError("Lower and upper bounds must have identical shapes.")
    if lower > upper:
        raise ValueError("Lower bounds must not exceed upper bounds.")


def _validate_radius(radius: Data) -> None:
    if isinstance(radius, tuple):
        for item in radius:
            _validate_radius(item)
        return
    if radius < 0.0:
        raise ValueError("Interval radii must be non-negative.")


def _contains(lower: Data, upper: Data, value: Data) -> bool:
    if isinstance(lower, tuple) and isinstance(upper, tuple) and isinstance(value, tuple):
        return len(lower) == len(upper) == len(value) and all(
            _contains(lower_item, upper_item, value_item)
            for lower_item, upper_item, value_item in zip(lower, upper, value)
        )
    if isinstance(lower, tuple) or isinstance(upper, tuple) or isinstance(value, tuple):
        return False
    return lower <= value <= upper


def _shape(value: Data) -> tuple[int, ...]:
    if isinstance(value, tuple):
        if not value:
            return (0,)
        return (len(value),) + _shape(value[0])
    return ()


def _neg(value: Data) -> Data:
    return _map_unary(value, lambda item: -item)


def _add_data(left: Data, right: Data) -> Data:
    return _map_binary(left, right, lambda l, r: l + r)


def _sub_data(left: Data, right: Data) -> Data:
    return _map_binary(left, right, lambda l, r: l - r)


def _mul_data(left: Data, right: Data) -> Data:
    return _map_binary(left, right, lambda l, r: l * r)


def _scale_data(value: Data, scalar: float) -> Data:
    return _map_unary(value, lambda item: item * scalar)


def _abs_data(value: Data) -> Data:
    return _map_unary(value, abs)


def _lower_from_mid_rad(midpoint: Data, radius: Data) -> Data:
    exact = _sub_data(midpoint, radius)
    return outward_lower(exact)


def _upper_from_mid_rad(midpoint: Data, radius: Data) -> Data:
    exact = _add_data(midpoint, radius)
    return outward_upper(exact)


def _mid_from_bounds(lower: Data, upper: Data) -> Data:
    return _map_binary(lower, upper, lambda lo, hi: (lo + hi) / 2.0)


def _rad_from_bounds(lower: Data, upper: Data) -> Data:
    return _map_binary(lower, upper, lambda lo, hi: (hi - lo) / 2.0)


def _mul_bounds(
    left_lower: Data,
    left_upper: Data,
    right_lower: Data,
    right_upper: Data,
) -> tuple[Data, Data]:
    if all(
        isinstance(item, tuple)
        for item in (left_lower, left_upper, right_lower, right_upper)
    ):
        if not (
            len(left_lower) == len(left_upper) == len(right_lower) == len(right_upper)
        ):
            raise ValueError("Mismatched interval shapes.")
        nested_bounds = tuple(
            _mul_bounds(ll, lu, rl, ru)
            for ll, lu, rl, ru in zip(left_lower, left_upper, right_lower, right_upper)
        )
        return (
            tuple(lower for lower, _ in nested_bounds),
            tuple(upper for _, upper in nested_bounds),
        )
    if any(
        isinstance(item, tuple)
        for item in (left_lower, left_upper, right_lower, right_upper)
    ):
        raise ValueError("Mismatched interval shapes.")

    candidates = (
        left_lower * right_lower,
        left_lower * right_upper,
        left_upper * right_lower,
        left_upper * right_upper,
    )
    return nextafter(min(candidates), -inf), nextafter(max(candidates), inf)


@dataclass(frozen=True, init=False)
class Interval:
    """Closed interval with midpoint-radius storage and outward-rounded arithmetic."""

    midpoint: Data
    radius: Data
    lower_bound: Data
    upper_bound: Data

    def __init__(self, lower: Any, upper: Any) -> None:
        lower_data = _to_data(lower)
        upper_data = _to_data(upper)
        _validate_bounds(lower_data, upper_data)
        midpoint = _mid_from_bounds(lower_data, upper_data)
        radius = _rad_from_bounds(lower_data, upper_data)
        _validate_radius(radius)
        object.__setattr__(self, "midpoint", midpoint)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "lower_bound", lower_data)
        object.__setattr__(self, "upper_bound", upper_data)

    @classmethod
    def point(cls, value: Any) -> "Interval":
        data = _to_data(value)
        return cls(data, data)

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "Interval":
        return cls(lower, upper)

    @classmethod
    def from_mid_rad(cls, midpoint: Any, radius: Any) -> "Interval":
        midpoint_data = _to_data(midpoint)
        radius_data = _to_data(radius)
        _validate_radius(radius_data)
        lower = _lower_from_mid_rad(midpoint_data, radius_data)
        upper = _upper_from_mid_rad(midpoint_data, radius_data)
        return cls(lower, upper)

    @property
    def lower(self) -> Data:
        return self.lower_bound

    @property
    def upper(self) -> Data:
        return self.upper_bound

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.midpoint)

    def contains(self, value: Any) -> bool:
        return _contains(self.lower, self.upper, _to_data(value))

    def as_tuple(self) -> tuple[Data, Data]:
        return self.lower, self.upper

    def __add__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        midpoint = _add_data(self.midpoint, other_interval.midpoint)
        radius = outward_upper(_add_data(self.radius, other_interval.radius))
        return Interval.from_mid_rad(midpoint, radius)

    def __radd__(self, other: Any) -> "Interval":
        return self + other

    def __sub__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        midpoint = _sub_data(self.midpoint, other_interval.midpoint)
        radius = outward_upper(_add_data(self.radius, other_interval.radius))
        return Interval.from_mid_rad(midpoint, radius)

    def __rsub__(self, other: Any) -> "Interval":
        return (other if isinstance(other, Interval) else Interval.point(other)) - self

    def __neg__(self) -> "Interval":
        return Interval.from_mid_rad(_neg(self.midpoint), self.radius)

    def __mul__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        lower, upper = _mul_bounds(
            self.lower,
            self.upper,
            other_interval.lower,
            other_interval.upper,
        )
        return Interval(lower, upper)

    def __rmul__(self, other: Any) -> "Interval":
        return self * other

    def __truediv__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        if isinstance(other_interval.lower, tuple):
            raise NotImplementedError("Vector interval division is not implemented.")
        if other_interval.lower <= 0.0 <= other_interval.upper:
            raise ZeroDivisionError("Interval division by an interval containing zero is undefined.")
        reciprocal = Interval(nextafter(1.0 / other_interval.upper, -inf), nextafter(1.0 / other_interval.lower, inf))
        return self * reciprocal

    def __repr__(self) -> str:
        return (
            "Interval("
            f"midpoint={self.midpoint!r}, "
            f"radius={self.radius!r}, "
            f"lower={self.lower!r}, "
            f"upper={self.upper!r}"
            ")"
        )
