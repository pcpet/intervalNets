# intervalNets

`intervalNets` is a robust PyTorch-first toolkit focused on three core capabilities:

1. an overloaded `model.eval(interval)` pathway (enabled via `enable_interval_eval()`) for interval
   propagation through neural networks with outward-rounded arithmetic, including roundoff-aware bounds;
2. rigorous enclosure of Lebesgue/Lp norms over interval domains via `model.lpnorm(domain, p, iterations=...)`;
3. interval Jacobian enclosure via `model.eval_jacobian(domain)` and Sobolev-style norms via
   `model.sobolev_norm(domain, p, iterations=...)`.
   The current implementation follows the same interval-enclosure + adaptive-refinement strategy outlined in
   the preprint *Certified and accurate computation of function space norms of deep neural networks*
   (arXiv:2603.06431).

## Highlights

- interval construction and arithmetic operations use outward rounding to account for floating-point roundoff errors,
- degenerate intervals `[x, x]` are supported,
- interval propagation currently supports `nn.Sequential`, `nn.Flatten`, `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softplus`, `nn.LeakyReLU`, `nn.Softmax`, `nn.Identity`, plus `IntervalAdd`/`IntervalCat` branch combinators,
- `model.eval(interval)`, `model.eval_jacobian(...)`, `model.lpnorm(...)`, and `model.sobolev_norm(...)` are attached through a single opt-in monkey patch (`enable_interval_eval()`),
- `model.lpnorm(domain, p, iterations, theta=0.5)` and
  `model.sobolev_norm(domain, p, iterations, theta=0.5)` use
  Dörfler-type bulk marking (with uncertainty indicators) and adaptive
  bisection to return outward-rounded certified norm enclosures.

## Quick start

### Option 1: install the package into your environment

From the repository root (either approach works):

```bash
# Option A: install dependencies directly
pip install -r requirements.txt

# Option B: install the package in editable mode
pip install -e .
```

Then you can import it normally:

```python
from intervalnets import IntervalTensor, enable_interval_eval
from torch import nn

enable_interval_eval()

model = nn.Sequential(nn.Linear(2, 1))
interval = IntervalTensor.from_bounds([1.0, 2.0], [1.5, 2.5])
output = model.eval(interval)

domain = IntervalTensor.from_bounds([0.0, 0.0], [1.0, 1.0])
lp_bounds = model.lpnorm(domain, p=2.0, iterations=8)
w1p_bounds = model.sobolev_norm(domain, p=2.0, iterations=8)
```

### Option 2: run directly from the repo without installing

If you are in a notebook or script inside the repository, add `src/` to `sys.path` first:

```python
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
while not (repo_root / "src" / "intervalnets").exists() and repo_root != repo_root.parent:
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root / "src"))
```

After that, `from intervalnets import ...` will work from the checkout as well.

## Installation notes

- the core `Interval` type uses only the Python standard library,
- PyTorch integration is optional and requires installing PyTorch separately,
- the included notebook now auto-detects the repo root and adds `src/` to `sys.path` for convenience.

For worked examples, see:

- `notebooks/test_suite.ipynb` for quick feature checks and sanity tests,
- `notebooks/reproduce_lp_w1p_experiments.ipynb` for reproducible certified
  `L^p` and `W^{1,p}` experiments aligned with arXiv:2603.06431 (intentionally excluding `W^{2,p}`).

## Reference

- Johannes Gründler, Moritz Maibaum, Philipp Petersen,
  *Certified and accurate computation of function space norms of deep neural networks*,
  arXiv:2603.06431 (2026). https://arxiv.org/abs/2603.06431

## Authors and acknowledgements

`intervalNets` was created by **Moritz Maibaum** and **Philipp Petersen**, with development support from **OpenAI Codex**.

## Release checklist (recommended before first public release)

- add a `LICENSE` file and corresponding metadata in `pyproject.toml`,
- add `project.urls` entries (repository, issue tracker, documentation),
- add `classifiers`/`keywords` metadata for PyPI discovery,
- ensure changelog/release notes exist (`CHANGELOG.md`),
- verify packaging artifacts with `python -m build` and `twine check dist/*`,
- run the full test suite (`pytest`) in a clean environment before tagging.
