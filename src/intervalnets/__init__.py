"""Interval arithmetic utilities for neural network evaluation."""

from .interval import Interval

__all__ = ["Interval"]

try:
    from .pytorch import IntervalAdd, IntervalCat, IntervalTensor, enable_interval_eval, interval_forward
except ImportError:  # pragma: no cover - optional dependency
    pass
else:
    __all__.extend(["IntervalTensor", "IntervalAdd", "IntervalCat", "enable_interval_eval", "interval_forward"])
