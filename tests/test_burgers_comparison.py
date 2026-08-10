from math import isclose

import pytest

torch = pytest.importorskip("torch")

from intervalnets.burgers_comparison import (
    BURGERS_DOMAIN,
    build_burgers_network,
    burgers_value_jets,
    fold_burgers_input_normalization,
    greedy_certification_curve,
    split_box_four,
)


def test_folded_input_normalization_is_exact() -> None:
    torch.manual_seed(3)
    normalized = build_burgers_network(hidden_layers=2, width=4)
    physical = fold_burgers_input_normalization(normalized)
    points = torch.tensor([[0.0, -1.0], [0.25, 0.3], [1.0, 1.0]])
    normalized_points = torch.stack((2.0 * points[:, 0] - 1.0, points[:, 1]), dim=-1)
    assert torch.allclose(normalized(normalized_points), physical(points), atol=2e-7, rtol=2e-7)


def test_direct_burgers_jets_match_autograd() -> None:
    torch.manual_seed(5)
    network = build_burgers_network(hidden_layers=2, width=4).double()
    points = torch.tensor([[0.2, -0.4], [0.8, 0.3]], dtype=torch.float64, requires_grad=True)
    value, u_t, u_x, u_xx = burgers_value_jets(network, points)
    normalized = torch.stack((2.0 * points[:, 0] - 1.0, points[:, 1]), dim=-1)
    reference = network(normalized)
    gradient = torch.autograd.grad(reference.sum(), points, create_graph=True)[0]
    reference_xx = torch.autograd.grad(gradient[:, 1].sum(), points)[0][:, 1:2]
    assert torch.allclose(value, reference, atol=1e-11, rtol=1e-11)
    assert torch.allclose(u_t, gradient[:, 0:1], atol=1e-11, rtol=1e-11)
    assert torch.allclose(u_x, gradient[:, 1:2], atol=1e-11, rtol=1e-11)
    assert torch.allclose(u_xx, reference_xx, atol=1e-11, rtol=1e-11)


def test_four_way_split_covers_parent() -> None:
    children = split_box_four(BURGERS_DOMAIN)
    assert len(children) == 4
    assert sum((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]) for b in children) == 2.0


def test_greedy_curve_uses_certified_hull_and_realizable_budgets() -> None:
    def bounder(cells):
        # Exact range of t+x on a box.
        return [(cell[0][0] + cell[0][1], cell[1][0] + cell[1][1]) for cell in cells]

    records = greedy_certification_curve("exact", bounder, [1, 5, 17], -1.0, 2.0)
    assert [row.cell_evaluations for row in records] == [1, 5, 17]
    for row in records:
        assert isclose(row.residual_lower, -1.0)
        assert isclose(row.residual_upper, 2.0)
        assert isclose(row.residual_squared_upper, 4.0)


def test_greedy_curve_rejects_invalid_bounds() -> None:
    with pytest.raises(AssertionError, match="Invalid certified interval"):
        greedy_certification_curve("bad", lambda cells: [(1.0, -1.0)] * len(cells), [1], 0.0, 0.0)
