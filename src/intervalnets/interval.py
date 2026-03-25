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


@dataclass(frozen=True)
class Interval:
    """Closed interval with outward-rounded arithmetic."""

    lower: Data
    upper: Data

    def __post_init__(self) -> None:
        lower = _to_data(self.lower)
        upper = _to_data(self.upper)
        _validate_bounds(lower, upper)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value: Any) -> "Interval":
        data = _to_data(value)
        return cls(outward_lower(data), outward_upper(data))

    @classmethod
    def from_bounds(cls, lower: Any, upper: Any) -> "Interval":
        return cls(outward_lower(lower), outward_upper(upper))

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.lower)

    def contains(self, value: Any) -> bool:
        return _contains(self.lower, self.upper, _to_data(value))

    def as_tuple(self) -> tuple[Data, Data]:
        return self.lower, self.upper

    def __add__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        lower = _map_binary(self.lower, other_interval.lower, lambda left, right: nextafter(left + right, -inf))
        upper = _map_binary(self.upper, other_interval.upper, lambda left, right: nextafter(left + right, inf))
        return Interval(lower, upper)

    def __radd__(self, other: Any) -> "Interval":
        return self + other

    def __sub__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        lower = _map_binary(self.lower, other_interval.upper, lambda left, right: nextafter(left - right, -inf))
        upper = _map_binary(self.upper, other_interval.lower, lambda left, right: nextafter(left - right, inf))
        return Interval(lower, upper)

    def __rsub__(self, other: Any) -> "Interval":
        return (other if isinstance(other, Interval) else Interval.point(other)) - self

    def __neg__(self) -> "Interval":
        return Interval(outward_lower(_neg(self.upper)), outward_upper(_neg(self.lower)))

    def __mul__(self, other: Any) -> "Interval":
        other_interval = other if isinstance(other, Interval) else Interval.point(other)
        if isinstance(self.lower, tuple) or isinstance(other_interval.lower, tuple):
            lower = _map_binary(
                self.lower,
                other_interval.lower,
                lambda left, right: nextafter(left * right, -inf),
            )
            upper = _map_binary(
                self.upper,
                other_interval.upper,
                lambda left, right: nextafter(left * right, inf),
            )
            return Interval(lower, upper)
        candidates = (
            self.lower * other_interval.lower,
            self.lower * other_interval.upper,
            self.upper * other_interval.lower,
            self.upper * other_interval.upper,
        )
        return Interval(nextafter(min(candidates), -inf), nextafter(max(candidates), inf))

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
        return f"Interval(lower={self.lower!r}, upper={self.upper!r})"
