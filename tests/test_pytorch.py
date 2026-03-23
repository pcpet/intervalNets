import pytest

torch = pytest.importorskip("torch")
from torch import nn

from intervalnets import IntervalTensor, enable_interval_eval, interval_forward


def test_linear_interval_matches_expected_affine_bounds() -> None:
    layer = nn.Linear(2, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -3.0]]))
        layer.bias.copy_(torch.tensor([0.5]))

    interval = IntervalTensor.from_bounds([1.0, 2.0], [1.5, 2.5])
    output = interval_forward(layer, interval)

    candidates = [
        2.0 * x1 - 3.0 * x2 + 0.5
        for x1 in [1.0, 1.5]
        for x2 in [2.0, 2.5]
    ]
    assert output.lower[0] <= min(candidates)
    assert output.upper[0] >= max(candidates)


def test_zero_network_contains_zero_with_rounding_margin() -> None:
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.zero_()

    interval = IntervalTensor.point([0.0, 0.0, 0.0])
    output = interval_forward(layer, interval)

    assert all(item < 0.0 for item in output.lower)
    assert all(item > 0.0 for item in output.upper)


def test_random_nonzero_network_expands_degenerate_input() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))

    with torch.no_grad():
        assert torch.count_nonzero(model[0].weight) > 0
        assert torch.count_nonzero(model[1].weight) > 0

    interval = IntervalTensor.point([1.0, 1.0, 1.0])
    output = interval_forward(model, interval)

    assert all(lower < upper for lower, upper in zip(output.lower, output.upper))


def test_eval_overload_runs_interval_propagation() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
        model[0].bias.copy_(torch.tensor([0.0, 0.0, 0.0]))
        model[1].weight.copy_(torch.tensor([[2.0, -1.0, 0.5]]))
        model[1].bias.copy_(torch.tensor([1.0]))

    interval = IntervalTensor.from_bounds([1.0, 2.0], [1.25, 2.5])
    result = model.eval(interval)
    assert len(result.lower) == 1
    assert len(result.upper) == 1
    assert result.lower[0] <= 1.0
    assert result.upper[0] >= 1.125


def test_unsupported_activation_raises_not_implemented() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    interval = IntervalTensor.point([0.0, 1.0])
    with pytest.raises(NotImplementedError):
        _ = interval_forward(model, interval)
