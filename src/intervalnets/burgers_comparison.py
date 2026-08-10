"""Reproducible Burgers PINN training and pointwise certification comparison.

This module deliberately keeps the paper-reference values separate from the
same-network experiment.  The public partial-CROWN repository does not ship
the checkpoint used for the ICML 2024 tables, so comparing those published
numbers directly with bounds for a newly trained network would be invalid.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import heapq
import json
from math import pi
from pathlib import Path
import random
import sys
import time
import types
from typing import Callable, Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional project dependency
    torch = None
    nn = None


BURGERS_DOMAIN = ((0.0, -1.0), (1.0, 1.0))
BURGERS_VISCOSITY = 0.01 / pi
PARTIAL_CROWN_COMMIT = "161377048ee92b4c09faf5ce41628168c7e01556"

PAPER_REFERENCE = {
    "paper": "Eiras et al. (ICML 2024), arXiv:2305.10157",
    "checkpoint_available": False,
    "hardware": "MacBook Pro, 10-core M1 Max CPU",
    "architecture_reported": "2-20x8-1 tanh MLP",
    "residual_squared_mc_1e6": 1.80e-2,
    "residual_squared_partial_crown_ub": 1.03e-1,
    "residual_branches": 2_000_000,
    "residual_time_s": 2.8e5,
    "initial_squared_partial_crown_ub": 2.63e-6,
    "left_boundary_squared_partial_crown_ub": 6.63e-7,
    "right_boundary_squared_partial_crown_ub": 9.39e-7,
    "initial_boundary_branches": 5_000,
}


def _require_torch() -> None:
    if torch is None or nn is None:  # pragma: no cover
        raise ImportError("PyTorch is required for the Burgers comparison.")


def build_burgers_network(
    *, hidden_layers: int = 8, width: int = 20, dtype=None
) -> "nn.Sequential":
    """Build the paper-reported tanh architecture on normalized inputs.

    The returned sequential network consumes normalized coordinates
    ``(2*t-1, x)``.  Use :func:`fold_burgers_input_normalization` for a model
    consuming the physical coordinates ``(t, x)``.
    """

    _require_torch()
    if hidden_layers < 1 or width < 1:
        raise ValueError("hidden_layers and width must be positive.")
    dtype = dtype or torch.float32
    layers: list[nn.Module] = []
    in_features = 2
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_features, width, dtype=dtype))
        layers.append(nn.Tanh())
        in_features = width
    layers.append(nn.Linear(width, 1, dtype=dtype))
    return nn.Sequential(*layers)


def fold_burgers_input_normalization(network: "nn.Sequential") -> "nn.Sequential":
    """Return an equivalent network accepting physical ``(t, x)`` inputs."""

    _require_torch()
    import copy

    result = copy.deepcopy(network)
    first = result[0]
    if not isinstance(first, nn.Linear) or first.in_features != 2:
        raise ValueError("Expected a sequential network beginning with Linear(2, ...).")
    with torch.no_grad():
        original_weight = first.weight.detach().clone()
        scale = torch.tensor([2.0, 1.0], dtype=first.weight.dtype, device=first.weight.device)
        shift = torch.tensor([-1.0, 0.0], dtype=first.weight.dtype, device=first.weight.device)
        first.weight.copy_(original_weight * scale.unsqueeze(0))
        first.bias.add_(original_weight @ shift)
    result.eval()
    return result


def burgers_value_jets(
    normalized_network: "nn.Sequential", points: "torch.Tensor"
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Evaluate ``u, u_t, u_x, u_xx`` using differentiable jet propagation."""

    from .pinn import sequential_value_jacobian_laplacian

    normalized = torch.stack((2.0 * points[:, 0] - 1.0, points[:, 1]), dim=-1)
    value, jacobian_normalized, _ = sequential_value_jacobian_laplacian(
        normalized_network, normalized
    )

    # Propagate the x-Hessian diagonal directly.  The generic helper returns
    # only a Laplacian, while Burgers needs u_xx specifically.
    y = normalized
    jac_x = torch.zeros((*points.shape[:-1], 2), dtype=points.dtype, device=points.device)
    jac_x[..., 1] = 1.0
    hess_xx = torch.zeros((*points.shape[:-1], 2), dtype=points.dtype, device=points.device)
    for layer in normalized_network:
        if isinstance(layer, nn.Linear):
            y = layer(y)
            jac_x = torch.einsum("oi,...i->...o", layer.weight, jac_x)
            hess_xx = torch.einsum("oi,...i->...o", layer.weight, hess_xx)
        elif isinstance(layer, nn.Tanh):
            activated = torch.tanh(y)
            first = 1.0 - activated.square()
            second = -2.0 * activated * first
            hess_xx = second * jac_x.square() + first * hess_xx
            jac_x = first * jac_x
            y = activated
        else:  # pragma: no cover - constructor invariant
            raise TypeError(f"Unsupported layer {type(layer).__name__}.")

    u_t = 2.0 * jacobian_normalized[..., 0, 0:1]
    u_x = jacobian_normalized[..., 0, 1:2]
    return value, u_t, u_x, hess_xx[..., 0:1]


def burgers_residual(normalized_network: "nn.Sequential", points: "torch.Tensor") -> "torch.Tensor":
    value, u_t, u_x, u_xx = burgers_value_jets(normalized_network, points)
    return u_t + value * u_x - BURGERS_VISCOSITY * u_xx


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 0
    hidden_layers: int = 8
    width: int = 20
    collocation_points: int = 4096
    initial_points: int = 128
    boundary_points: int = 128
    adam_steps: int = 3000
    adam_lr: float = 1e-3
    lbfgs_steps: int = 500


def train_replacement_burgers(
    output: str | Path, config: TrainingConfig
) -> dict[str, float | int | str]:
    """Train and save a deterministic architecture-matched replacement PINN."""

    _require_torch()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))

    network = build_burgers_network(
        hidden_layers=config.hidden_layers, width=config.width, dtype=torch.float32
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    sobol = torch.quasirandom.SobolEngine(2, scramble=True, seed=config.seed)
    residual_points = sobol.draw(config.collocation_points)
    residual_points[:, 1] = 2.0 * residual_points[:, 1] - 1.0
    x0 = torch.linspace(-1.0, 1.0, config.initial_points)
    initial = torch.stack((torch.zeros_like(x0), x0), dim=-1)
    tb = torch.linspace(0.0, 1.0, config.boundary_points)
    left = torch.stack((tb, -torch.ones_like(tb)), dim=-1)
    right = torch.stack((tb, torch.ones_like(tb)), dim=-1)

    def loss_fn() -> "torch.Tensor":
        residual_loss = burgers_residual(network, residual_points).square().mean()
        initial_loss = (
            network(torch.stack((-torch.ones_like(x0), x0), dim=-1))
            + torch.sin(pi * x0).unsqueeze(-1)
        ).square().mean()
        left_normalized = torch.stack((2.0 * tb - 1.0, -torch.ones_like(tb)), dim=-1)
        right_normalized = torch.stack((2.0 * tb - 1.0, torch.ones_like(tb)), dim=-1)
        boundary_loss = network(left_normalized).square().mean() + network(right_normalized).square().mean()
        return residual_loss + initial_loss + boundary_loss

    started = time.perf_counter()
    adam = torch.optim.Adam(network.parameters(), lr=config.adam_lr)
    for _ in range(config.adam_steps):
        adam.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        adam.step()

    if config.lbfgs_steps:
        optimizer = torch.optim.LBFGS(
            network.parameters(),
            lr=1.0,
            max_iter=config.lbfgs_steps,
            max_eval=config.lbfgs_steps,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> "torch.Tensor":
            optimizer.zero_grad(set_to_none=True)
            current = loss_fn()
            current.backward()
            return current

        optimizer.step(closure)

    elapsed = time.perf_counter() - started
    final_loss = float(loss_fn().detach().item())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": network.state_dict(),
            "config": asdict(config),
            "training_time_s": elapsed,
            "final_training_loss": final_loss,
            "input_convention": "normalized (2*t-1, x)",
        },
        output,
    )
    return {"checkpoint": str(output), "training_time_s": elapsed, "final_training_loss": final_loss}


def load_replacement_burgers(path: str | Path) -> tuple["nn.Sequential", dict]:
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = payload["config"]
    network = build_burgers_network(
        hidden_layers=int(config["hidden_layers"]), width=int(config["width"])
    )
    network.load_state_dict(payload["state_dict"])
    network.eval()
    return network, payload


def sampled_residual_extrema(
    network: "nn.Sequential", *, grid_size: int = 301, batch_size: int = 65536
) -> tuple[float, float, float]:
    _require_torch()
    ts = torch.linspace(0.0, 1.0, grid_size)
    xs = torch.linspace(-1.0, 1.0, grid_size)
    grid_t, grid_x = torch.meshgrid(ts, xs, indexing="ij")
    points = torch.stack((grid_t.reshape(-1), grid_x.reshape(-1)), dim=-1)
    values: list[torch.Tensor] = []
    for chunk in points.split(batch_size):
        values.append(burgers_residual(network, chunk).detach())
    residual = torch.cat(values).reshape(-1)
    lower = float(residual.min().item())
    upper = float(residual.max().item())
    return lower, upper, max(abs(lower), abs(upper))


Box = tuple[tuple[float, float], tuple[float, float]]
Bounder = Callable[[Sequence[Box]], list[tuple[float, float]]]


def split_box_four(box: Box) -> list[Box]:
    (t0, x0), (t1, x1) = box
    tm, xm = (t0 + t1) / 2.0, (x0 + x1) / 2.0
    return [
        ((ta, xa), (tb, xb))
        for ta, tb in ((t0, tm), (tm, t1))
        for xa, xb in ((x0, xm), (xm, x1))
    ]


@dataclass(frozen=True)
class CertificationRecord:
    method: str
    cell_evaluations: int
    active_cells: int
    residual_lower: float
    residual_upper: float
    residual_abs_upper: float
    residual_squared_upper: float
    sampled_abs_max: float
    sampled_extrema_contained: bool
    upper_to_sample_ratio: float
    excess_over_sample: float
    verifier_time_s: float
    cells_per_second: float


def greedy_certification_curve(
    method: str,
    bounder: Bounder,
    budgets: Iterable[int],
    sampled_lower: float,
    sampled_upper: float,
) -> list[CertificationRecord]:
    """Run the paper's sampled-gap-driven four-way greedy subdivision."""

    targets = sorted(set(int(value) for value in budgets))
    if not targets or targets[0] < 1:
        raise ValueError("budgets must contain positive integers.")

    counter = 0
    elapsed = 0.0
    bounds_by_id: dict[int, tuple[Box, float, float]] = {}
    heap: list[tuple[float, int]] = []

    def add(cells: Sequence[Box]) -> None:
        nonlocal counter, elapsed
        started = time.perf_counter()
        cell_bounds = bounder(cells)
        elapsed += time.perf_counter() - started
        for cell, (lower, upper) in zip(cells, cell_bounds):
            if lower > upper:
                raise AssertionError(f"Invalid certified interval [{lower}, {upper}].")
            cell_id = counter
            counter += 1
            bounds_by_id[cell_id] = (cell, float(lower), float(upper))
            gap = max(sampled_lower - lower, upper - sampled_upper)
            heapq.heappush(heap, (-float(gap), cell_id))

    add([BURGERS_DOMAIN])
    evaluations = 1
    records: list[CertificationRecord] = []

    def snapshot(target: int) -> None:
        global_lower = min(value[1] for value in bounds_by_id.values())
        global_upper = max(value[2] for value in bounds_by_id.values())
        abs_upper = max(abs(global_lower), abs(global_upper))
        sample_abs = max(abs(sampled_lower), abs(sampled_upper))
        contained = global_lower <= sampled_lower and sampled_upper <= global_upper
        if not contained:
            raise AssertionError(
                "Diagnostic sampled extrema fall outside the certified global interval."
            )
        records.append(
            CertificationRecord(
                method=method,
                cell_evaluations=evaluations,
                active_cells=len(bounds_by_id),
                residual_lower=global_lower,
                residual_upper=global_upper,
                residual_abs_upper=abs_upper,
                residual_squared_upper=abs_upper * abs_upper,
                sampled_abs_max=sample_abs,
                sampled_extrema_contained=contained,
                upper_to_sample_ratio=abs_upper / sample_abs if sample_abs else float("inf"),
                excess_over_sample=abs_upper - sample_abs,
                verifier_time_s=elapsed,
                cells_per_second=evaluations / elapsed if elapsed else float("inf"),
            )
        )

    target_index = 0
    while target_index < len(targets):
        while evaluations >= targets[target_index]:
            snapshot(targets[target_index])
            target_index += 1
            if target_index == len(targets):
                return records
        _, cell_id = heapq.heappop(heap)
        cell, _, _ = bounds_by_id.pop(cell_id)
        children = split_box_four(cell)
        add(children)
        evaluations += 4
    return records


def make_intervalnets_bounder(physical_network: "nn.Sequential") -> Bounder:
    """Construct the intervalNets two-jet Burgers residual bounder."""

    from .interval import Interval
    from .pytorch import IntervalTensor, enable_interval_eval

    enable_interval_eval(enclosure_mode="box")

    def bound(cells: Sequence[Box]) -> list[tuple[float, float]]:
        output: list[tuple[float, float]] = []
        for cell in cells:
            domain = IntervalTensor.from_bounds(cell[0], cell[1])
            value = physical_network.eval(domain)
            jacobian = physical_network.eval_jacobian(domain)
            hessian = physical_network.eval_hessian(domain)
            u = Interval(float(value.lower[0]), float(value.upper[0]))
            u_t = Interval(float(jacobian.lower[0][0]), float(jacobian.upper[0][0]))
            u_x = Interval(float(jacobian.lower[0][1]), float(jacobian.upper[0][1]))
            u_xx = Interval(float(hessian.lower[0][1][1]), float(hessian.upper[0][1][1]))
            residual = u_t + u * u_x - BURGERS_VISCOSITY * u_xx
            output.append((float(residual.lower), float(residual.upper)))
        return output

    return bound


def _install_partial_crown_import_shims() -> None:
    """Provide the two normalization module names omitted from the release."""

    _require_torch()
    if "tools.custom_torch_modules" in sys.modules:
        return
    tools = types.ModuleType("tools")
    custom = types.ModuleType("tools.custom_torch_modules")

    class Add(nn.Module):
        def __init__(self, value=0.0):
            super().__init__()
            self.value = value

        def forward(self, x):
            return x + self.value

    class Mul(nn.Module):
        def __init__(self, value=1.0):
            super().__init__()
            self.value = value

        def forward(self, x):
            return x * self.value

    custom.Add = Add
    custom.Mul = Mul
    tools.custom_torch_modules = custom
    sys.modules["tools"] = tools
    sys.modules["tools.custom_torch_modules"] = custom

    # The verifier imports tqdm solely to decorate debug iterators.  Keep the
    # comparison runnable in minimal environments without changing numerical
    # behavior; the Actions environment installs the real dependency.
    try:
        import tqdm as _tqdm  # noqa: F401
    except ImportError:  # pragma: no cover - environment dependent
        progress = types.ModuleType("tqdm")
        progress.tqdm = lambda iterable, *args, **kwargs: iterable
        sys.modules["tqdm"] = progress


def make_partial_crown_bounder(
    physical_network: "nn.Sequential", partial_crown_dir: str | Path
) -> Bounder:
    """Construct a bounder using the authors' pinned partial-CROWN code."""

    root = Path(partial_crown_dir).resolve()
    if not (root / "pinn_verifier" / "burgers.py").is_file():
        raise FileNotFoundError(f"partial-CROWN source not found under {root}.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _install_partial_crown_import_shims()
    # The pinned 2024 source predates NumPy 2.0's removal of this alias.
    if not hasattr(np, "Inf"):  # pragma: no branch - version compatibility
        np.Inf = np.inf  # type: ignore[attr-defined]

    from pinn_verifier.activations.activation_relaxations import (  # type: ignore
        ActivationRelaxationType,
    )
    from pinn_verifier.activations.tanh import (  # type: ignore
        TanhDerivativeRelaxation,
        TanhRelaxation,
        TanhSecondDerivativeRelaxation,
    )
    from pinn_verifier.burgers import CROWNBurgersVerifier  # type: ignore

    layers = list(physical_network.children())

    def bound(cells: Sequence[Box]) -> list[tuple[float, float]]:
        verifier = CROWNBurgersVerifier(
            layers,
            activation_relaxation=TanhRelaxation(ActivationRelaxationType.SINGLE_LINE),
            activation_derivative_relaxation=TanhDerivativeRelaxation(
                ActivationRelaxationType.SINGLE_LINE
            ),
            activation_second_derivative_relaxation=TanhSecondDerivativeRelaxation(
                ActivationRelaxationType.SINGLE_LINE
            ),
        )
        tensor = torch.tensor(cells, dtype=torch.float32)
        upper, lower = verifier.compute_residual_bound(tensor, debug=False)
        return [
            (float(lo.item()), float(hi.item()))
            for lo, hi in zip(lower.reshape(-1), upper.reshape(-1))
        ]

    return bound


def run_comparison(
    checkpoint: str | Path,
    output_dir: str | Path,
    partial_crown_dir: str | Path,
    *,
    budgets: Sequence[int] = (1, 5, 17, 65, 257),
    sample_grid_size: int = 301,
) -> dict:
    network, payload = load_replacement_burgers(checkpoint)
    physical_network = fold_burgers_input_normalization(network)
    sampled_lower, sampled_upper, sampled_abs = sampled_residual_extrema(
        network, grid_size=sample_grid_size
    )

    methods = (
        ("intervalnets_interval_twojet", make_intervalnets_bounder(physical_network)),
        ("partial_crown", make_partial_crown_bounder(physical_network, partial_crown_dir)),
    )
    records: list[CertificationRecord] = []
    for name, bounder in methods:
        records.extend(
            greedy_certification_curve(
                name, bounder, budgets, sampled_lower, sampled_upper
            )
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with (output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "schema_version": "1.0",
        "experiment": "architecture-matched replacement Burgers PINN",
        "paper_reference": PAPER_REFERENCE,
        "partial_crown_commit": PARTIAL_CROWN_COMMIT,
        "checkpoint": str(checkpoint),
        "training_config": payload["config"],
        "training_time_s": float(payload["training_time_s"]),
        "final_training_loss": float(payload["final_training_loss"]),
        "domain": {"t": [0.0, 1.0], "x": [-1.0, 1.0]},
        "viscosity": BURGERS_VISCOSITY,
        "sample_grid_size": sample_grid_size,
        "sample_count": sample_grid_size * sample_grid_size,
        "sampled_residual_lower": sampled_lower,
        "sampled_residual_upper": sampled_upper,
        "sampled_residual_abs_max": sampled_abs,
        "budgets_requested": list(budgets),
        "budget_note": "Four-way splitting yields realizable counts 1+4k; records use the first count meeting each request.",
        "timing_scope": "verifier calls only; excludes imports, training, sampling, and output",
        "method_status": {
            "partial_crown": "executed from the pinned authors' source",
            "intervalnets_interval_twojet": "executed scalable fallback",
            "intervalnets_dependency_preserving_pz_twojet": (
                "not executed: certified two-jet support reduction is not implemented; "
                "unreduced propagation is intractable for the 8x20 network"
            ),
        },
        "records": rows,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_dir / "paper_reference.json").write_text(json.dumps(PAPER_REFERENCE, indent=2) + "\n")
    return result


def _parse_budgets(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--adam-steps", type=int, default=3000)
    train_parser.add_argument("--lbfgs-steps", type=int, default=500)
    train_parser.add_argument("--collocation-points", type=int, default=4096)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--checkpoint", required=True)
    compare_parser.add_argument("--output-dir", required=True)
    compare_parser.add_argument("--partial-crown-dir", required=True)
    compare_parser.add_argument("--budgets", default="1,5,17,65,257")
    compare_parser.add_argument("--sample-grid-size", type=int, default=301)

    args = parser.parse_args(argv)
    if args.command == "train":
        result = train_replacement_burgers(
            args.output,
            TrainingConfig(
                seed=args.seed,
                adam_steps=args.adam_steps,
                lbfgs_steps=args.lbfgs_steps,
                collocation_points=args.collocation_points,
            ),
        )
    else:
        result = run_comparison(
            args.checkpoint,
            args.output_dir,
            args.partial_crown_dir,
            budgets=_parse_budgets(args.budgets),
            sample_grid_size=args.sample_grid_size,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
