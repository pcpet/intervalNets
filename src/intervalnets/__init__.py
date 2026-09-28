"""Interval arithmetic utilities for neural network evaluation."""

from .interval import Interval
from .activations import tanh_double_prime_bounds

__all__ = ["Interval", "tanh_double_prime_bounds"]

try:
    from .pytorch import (
        IntervalAdd,
        IntervalCat,
        IntervalTensor,
        enable_interval_eval,
        interval_forward,
        interval_forward_refine,
    )
except ImportError:  # pragma: no cover - optional dependency
    pass
else:
    __all__.extend(
        [
            "IntervalTensor",
            "IntervalAdd",
            "IntervalCat",
            "enable_interval_eval",
            "interval_forward",
            "interval_forward_refine",
        ]
    )
