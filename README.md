# intervalNets

`intervalNets` is a small PyTorch-first prototype for interval evaluation of linear neural networks.
It currently focuses on affine layers (`nn.Linear`) and supports an opt-in overload so that, after
calling `enable_interval_eval()`, you can write `model.eval(interval)` to propagate interval inputs
through the network using outward-rounded arithmetic.

## Highlights

- outward rounding is applied to interval construction and arithmetic operations,
- degenerate intervals `[x, x]` are expanded outward by one floating-point step,
- interval propagation currently supports `nn.Sequential`, `nn.Flatten`, and `nn.Linear`,
- unsupported activations deliberately raise `NotImplementedError` so the extension surface is explicit.

## Quick start

```python
from intervalnets import IntervalTensor, enable_interval_eval
from torch import nn

enable_interval_eval()

model = nn.Sequential(nn.Linear(2, 1))
interval = IntervalTensor.from_bounds([1.0, 2.0], [1.5, 2.5])
output = model.eval(interval)
```

## Installation notes

- the core `Interval` type uses only the Python standard library,
- PyTorch integration is optional and requires installing the `torch` extra or PyTorch directly.

For worked examples, see `notebooks/interval_linear_networks.ipynb`.
