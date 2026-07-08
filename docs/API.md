# intervalNets API documentation

This document describes the public Python API exposed by `intervalnets` and how it maps to the implementation in `src/intervalnets`.

## Package layout

- `intervalnets.interval.Interval`: core immutable interval type with outward-rounded scalar/tuple arithmetic.
- `intervalnets.pytorch.IntervalTensor`: interval type specialized for PyTorch interoperability.
- `intervalnets.pytorch.enable_interval_eval(enclosure_mode="slope")`: monkey patch that adds interval-aware methods onto `torch.nn.Module`.
- `intervalnets.pytorch.interval_forward(module, x, enclosure_mode="box")`: interval propagation backend used by patched `model.eval(interval)`.
- `intervalnets.pytorch.interval_forward_refine(module, x, enclosure_mode="slope", splits_per_dim=2, max_cells=256)`: optional subdivision-based forward refinement.
- `intervalnets.pytorch.IntervalAdd`, `intervalnets.pytorch.IntervalCat`: helper combinators for branched interval models.

## Core interval arithmetic (`Interval`)

`Interval` uses midpoint-radius internals to support fast affine interval propagation paths used by
the PyTorch backend. The library targets **certified enclosures with good throughput**, not globally
optimal/tightest interval boxes for every operation.

### Construction

- `Interval(lower, upper)`
  - Accepts scalar values or nested tuples/lists of numeric values.
  - Validates shape consistency and enforces `lower <= upper` element-wise.
  - Stores converted data as tuples of `float`.
- `Interval.point(value)`
  - Creates a degenerate interval around a point and expands both sides by one floating-point step (`nextafter`) for sound outward rounding.
- `Interval.from_bounds(lower, upper)`
  - Constructs from explicit bounds and outward-rounds both endpoints (`lower` downward, `upper` upward).

### Introspection

- `shape` property returns tuple shape of the stored lower bounds.
- `contains(value)` checks element-wise inclusion.
- `as_tuple()` returns `(lower, upper)`.

### Arithmetic operations

- `+`, `-`, unary `-`, `*`, `/` are implemented with outward rounding.
- In affine-heavy code paths (e.g. linear layers and Jacobian composition), interval propagation is
  optimized around midpoint-radius matrix formulas (`A x_mid ± |A| x_rad`).
- Scalar division by an interval containing `0` raises `ZeroDivisionError`.
- Vector interval division is intentionally not implemented and raises `NotImplementedError`.

## PyTorch integration (`IntervalTensor` + patching)

### `IntervalTensor`

`IntervalTensor` subclasses `Interval` and adds PyTorch conversion helpers:

- `IntervalTensor.point(value)`
  - Accepts Python values or `torch.Tensor` and converts tensors via `.detach().cpu().tolist()`.
- `IntervalTensor.from_bounds(lower, upper)`
  - Same tensor-to-Python conversion behavior for both endpoints.
- `to_torch(dtype=None)`
  - Returns `(lower_tensor, upper_tensor)` using `torch.float64` by default.

### Enabling model methods

Call once at startup:

```python
from intervalnets import enable_interval_eval

enable_interval_eval(enclosure_mode="slope")
```

After this, every `torch.nn.Module` gets:

- `model.eval()`
  - Standard PyTorch eval mode behavior (unchanged).
- `model.eval(interval: IntervalTensor)`
  - Interval forward propagation.
- `model.lpnorm(domain: IntervalTensor, p: float, iterations: int = 0, theta: float = 0.5, forward_refine_splits: int = 1, forward_refine_max_cells: int = 256)`
  - Outward-rounded enclosure of the model `L^p` norm on a box domain.
- `model.eval_jacobian(domain: IntervalTensor)`
  - Interval enclosure of Jacobian matrix entries over the domain, using the same `enclosure_mode` selected when calling `enable_interval_eval(...)` for sequential pre-activation propagation.
- `model.eval_hessian(domain: IntervalTensor)`
  - Interval enclosure of Hessian tensor entries over the domain (shape `(output_dim, input_dim, input_dim)`).
- `model.sobolev_norm(domain: IntervalTensor, p: float, order: int = 1, iterations: int = 0, theta: float = 0.5, forward_refine_splits: int = 1, forward_refine_max_cells: int = 256)`
  - Enclosure of a Sobolev-style norm over the domain:
    - `order=1`: `|f|^p + |Df|^p`,
    - `order=2`: `|f|^p + |Df|^p + |D^2 f|^p`.

> Note: these methods are attached by monkey-patching `torch.nn.Module`. If patching is not desired in your application architecture, call `interval_forward(...)` directly for pure forward enclosure and avoid the norm/Jacobian helpers.

## Supported layers and modules for interval forward propagation

`interval_forward(module, x, enclosure_mode="box")` currently supports:

- `nn.Sequential`
- `nn.Flatten` (for flat vectors)
- `nn.Linear`
- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softplus`
- `nn.LeakyReLU`
- `nn.Softmax` (1D vectors, `dim in {-1, 0}`)
- `nn.Identity`
- `IntervalAdd`
- `IntervalCat` (flat vectors, concatenation along the only axis)

Unsupported modules raise `NotImplementedError` with the offending module type.

`enclosure_mode` options:

- `"box"`: midpoint-radius interval propagation throughout.
- `"slope"` (default for `enable_interval_eval`): slope-aware affine relaxation for `nn.Sequential` chains of `nn.Linear` and `nn.ReLU`; unsupported layers are conservatively concretized and then continued with `"box"` mode.

ReLU-specific note:

- For non-positive input intervals, ReLU images are kept exactly as `[0, 0]` (not artificially widened around zero).

`interval_forward_refine(...)`:

- Subdivides a flat input interval into a regular grid and hulls the per-cell `interval_forward(...)` outputs.
- Conservative by construction and typically tighter than a single forward enclosure.
- Controlled by `splits_per_dim` and `max_cells` to balance tightness vs runtime.

## Branch combinators

### `IntervalAdd(left, right)`

Runs both branches on the same input interval and adds outputs element-wise.

- Requires matching output shapes.
- Intended for residual-like structures.

### `IntervalCat(*branches, dim=-1)`

Runs each branch on the same input interval and concatenates outputs.

- Requires at least one branch.
- Current interval backend supports 1D vector outputs and `dim in {0, -1}`.


## Certified norm computation details

`model.lpnorm(..., theta=0.5)` and `model.sobolev_norm(..., theta=0.5)` use adaptive box subdivision with Dörfler-type marking:

1. Start from one domain box.
2. Compute one indicator per box
   (`integrand interval width × box volume`).
3. Mark a minimal set of boxes whose indicator sum is at least
   `theta × (sum of all indicators)` (Dörfler bulk criterion).
4. Split every marked box by bisecting its widest coordinate.
5. Repeat for `iterations` rounds.
6. Accumulate interval integral bounds over the resulting partition.
7. Clamp tiny negative roundoff artifacts to zero before taking the `1/p` power.

Sobolev refinement additionally skips boxes that are rigorously constant with an exactly zero Jacobian enclosure (and exactly zero Hessian enclosure for `order=2`), avoiding unnecessary subdivision of derivative-inactive regions.

Important constraints:

- Input domain must be an `IntervalTensor`.
- Domain must be a flat vector box (1D tuple structure).
- `p` must be finite and strictly positive.
- `iterations` must be non-negative.
- `theta` must satisfy `0 < theta <= 1` (default: `0.5`).
- `forward_refine_splits` must be an integer `>= 1` (default: `1`, meaning no extra per-box subdivision).
- `forward_refine_max_cells` limits refinement combinatorics (default: `256`).

## Jacobian enclosure details

`model.eval_jacobian(domain)` returns an `IntervalTensor` whose shape is `(output_dim, input_dim)` (stored as nested tuples).

Layer derivatives currently implemented:

- `nn.Linear`
- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softmax`
- `nn.Flatten`

For `nn.Sequential`, Jacobian enclosures are composed with interval matrix multiplication, and layer-input intervals are computed with the configured `enclosure_mode` (`"slope"` by default via `enable_interval_eval`).

## Hessian enclosure details

`model.eval_hessian(domain)` returns an `IntervalTensor` with shape `(output_dim, input_dim, input_dim)` (stored as nested tuples).

Layer Hessians currently implemented:

- `nn.Linear` (exact zero Hessian),
- `nn.ReLU` (zero enclosure),
- `nn.Sigmoid`,
- `nn.Tanh`,
- `nn.Flatten`.

For `nn.Sequential`, Hessian enclosures are composed with interval chain-rule terms:

- Jacobian-weighted propagation of previous Hessians,
- plus local layer-Hessian curvature terms weighted by interval outer products of previous Jacobian rows.

Implementation note: Jacobian composition currently favors vectorized midpoint-radius interval matrix
products for speed. This is conservative and efficient, but not necessarily the tightest enclosure
that could be achieved with more expensive symbolic or optimization-based techniques.

## Minimal examples

### Interval arithmetic only

```python
from intervalnets import Interval

a = Interval.from_bounds(1.0, 2.0)
b = Interval.point(3.0)

print(a + b)   # outward-rounded enclosure of [1,2] + [3,3]
print(a * b)   # outward-rounded enclosure of [1,2] * [3,3]
```

### Interval model evaluation and norms

```python
from intervalnets import IntervalTensor, enable_interval_eval
from torch import nn

enable_interval_eval(enclosure_mode="slope")

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1),
)

box = IntervalTensor.from_bounds([0.0, -1.0], [1.0, 2.0])
out = model.eval(box)

lp = model.lpnorm(box, p=2.0, iterations=6, forward_refine_splits=2, forward_refine_max_cells=128)
jac = model.eval_jacobian(box)
sob = model.sobolev_norm(box, p=2.0, iterations=6, forward_refine_splits=2, forward_refine_max_cells=128)
```

## Exported names

The package-level import surface in `intervalnets.__init__` is:

- Always: `Interval`, `PolynomialZonotope`, `PZTwoJet`, `TanhApproximation`, `compute_tanh_polynomial`, `certify_tanh_residual_subdivision`, `tanh_pz_scalar`
- When PyTorch is importable: `IntervalTensor`, `IntervalAdd`, `IntervalCat`, `enable_interval_eval`, `interval_forward`, `interval_forward_refine`, `pz_twojet_forward`

Prefer importing these from the top-level package for user-facing code:

```python
from intervalnets import Interval, IntervalTensor, enable_interval_eval
```
