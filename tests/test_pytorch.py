import math
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from intervalnets import Interval, IntervalAdd, IntervalCat, IntervalTensor, enable_interval_eval, interval_forward
from intervalnets.pytorch import _interval_pow_scalar


def test_relu_negative_interval_rounds_outward_to_zero() -> None:
    relu = nn.ReLU()
    interval = IntervalTensor.from_bounds([-3.0, -0.5], [-1.0, -0.25])

    output = interval_forward(relu, interval)

    assert all(lower < 0.0 for lower in output.lower)
    assert all(upper > 0.0 for upper in output.upper)


def test_relu_positive_interval_preserves_endpoint_images() -> None:
    relu = nn.ReLU()
    interval = IntervalTensor.from_bounds([0.25, 1.5], [0.5, 3.0])

    output = interval_forward(relu, interval)

    assert output.lower[0] <= 0.25
    assert output.upper[0] >= 0.5
    assert output.lower[1] <= 1.5
    assert output.upper[1] >= 3.0


def test_relu_mixed_interval_clamps_only_the_lower_endpoint() -> None:
    relu = nn.ReLU()
    interval = IntervalTensor.from_bounds([-2.0, -1.0], [4.0, 2.5])

    output = interval_forward(relu, interval)

    assert output.lower[0] < 0.0
    assert output.upper[0] >= 4.0
    assert output.lower[1] < 0.0
    assert output.upper[1] >= 2.5


def test_relu_network_encloses_endpoint_evaluations() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1.0, -2.0], [-1.0, 0.5]]))
        model[0].bias.copy_(torch.tensor([0.25, -0.75]))
        model[2].weight.copy_(torch.tensor([[1.5, -0.5]]))
        model[2].bias.copy_(torch.tensor([0.1]))

    interval = IntervalTensor.from_bounds([-1.0, 0.5], [2.0, 1.5])
    output = interval_forward(model, interval)

    candidates = []
    for x1 in (-1.0, 2.0):
        for x2 in (0.5, 1.5):
            hidden_1 = max(0.0, x1 - 2.0 * x2 + 0.25)
            hidden_2 = max(0.0, -x1 + 0.5 * x2 - 0.75)
            candidates.append(1.5 * hidden_1 - 0.5 * hidden_2 + 0.1)

    assert output.lower[0] <= min(candidates)
    assert output.upper[0] >= max(candidates)


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



def test_softmax_bounds_match_closed_form_in_two_dimensions() -> None:
    softmax = nn.Softmax(dim=-1)
    interval = IntervalTensor.from_bounds([-1.0, 0.25], [0.5, 1.75])

    output = interval_forward(softmax, interval)

    expected_lower_0 = math.exp(-1.0) / (math.exp(-1.0) + math.exp(1.75))
    expected_upper_0 = math.exp(0.5) / (math.exp(0.5) + math.exp(0.25))
    expected_lower_1 = math.exp(0.25) / (math.exp(0.25) + math.exp(0.5))
    expected_upper_1 = math.exp(1.75) / (math.exp(1.75) + math.exp(-1.0))

    assert output.lower[0] <= expected_lower_0
    assert output.upper[0] >= expected_upper_0
    assert output.lower[1] <= expected_lower_1
    assert output.upper[1] >= expected_upper_1


def test_softmax_encloses_all_corner_evaluations_in_three_dimensions() -> None:
    softmax = nn.Softmax(dim=-1)
    interval = IntervalTensor.from_bounds([-1.5, -0.25, 0.1], [0.5, 1.0, 1.5])

    output = interval_forward(softmax, interval)

    corners = [
        (x0, x1, x2)
        for x0 in (interval.lower[0], interval.upper[0])
        for x1 in (interval.lower[1], interval.upper[1])
        for x2 in (interval.lower[2], interval.upper[2])
    ]
    for corner in corners:
        values = torch.tensor(corner, dtype=torch.float64)
        eval_softmax = torch.softmax(values, dim=-1)
        for idx, exact in enumerate(eval_softmax.tolist()):
            assert output.lower[idx] <= exact
            assert output.upper[idx] >= exact


def test_softmax_component_bounds_are_probabilities() -> None:
    softmax = nn.Softmax(dim=-1)
    interval = IntervalTensor.from_bounds([-2.0, -1.0, 0.0, 0.5], [1.0, 1.5, 2.0, 3.0])

    output = interval_forward(softmax, interval)

    assert all(0.0 <= lower <= 1.0 for lower in output.lower)
    assert all(0.0 <= upper <= 1.0 for upper in output.upper)
    assert sum(output.lower) <= 1.0 <= sum(output.upper)


def test_softmax_point_interval_is_outward_non_degenerate() -> None:
    softmax = nn.Softmax(dim=-1)
    interval = IntervalTensor.point([0.0, 1.0, -0.5])

    output = interval_forward(softmax, interval)
    exact = torch.softmax(torch.tensor([0.0, 1.0, -0.5], dtype=torch.float64), dim=-1).tolist()

    for idx, exact_item in enumerate(exact):
        assert output.lower[idx] < exact_item < output.upper[idx]


def test_softmax_accepts_dim_none_for_legacy_1d_models() -> None:
    softmax = nn.Softmax(dim=None)
    interval = IntervalTensor.from_bounds([-0.5, 0.25, 1.0], [0.5, 1.25, 2.0])

    output = interval_forward(softmax, interval)

    assert len(output.lower) == 3
    assert len(output.upper) == 3
    assert all(0.0 <= lower <= 1.0 for lower in output.lower)
    assert all(0.0 <= upper <= 1.0 for upper in output.upper)


def test_softmax_remains_finite_for_large_magnitude_logits() -> None:
    softmax = nn.Softmax(dim=-1)
    interval = IntervalTensor.from_bounds([800.0, -900.0, -950.0], [900.0, -800.0, -850.0])

    output = interval_forward(softmax, interval)

    assert all(math.isfinite(value) for value in output.lower)
    assert all(math.isfinite(value) for value in output.upper)
    assert output.lower[0] > 0.999
    assert output.upper[0] <= 1.0


@pytest.mark.parametrize(
    ("activation", "fn"),
    [
        (nn.Tanh(), lambda value: math.tanh(value)),
        (nn.Softplus(beta=1.0, threshold=20.0), lambda value: math.log1p(math.exp(value))),
        (nn.LeakyReLU(negative_slope=0.2), lambda value: value if value >= 0.0 else 0.2 * value),
    ],
)
def test_added_monotone_activations_enclose_endpoint_images(activation, fn) -> None:
    interval = IntervalTensor.from_bounds([-1.5, -0.25, 0.5], [0.2, 1.0, 2.0])

    output = interval_forward(activation, interval)

    for idx in range(len(interval.lower)):
        lower_exact = fn(interval.lower[idx])
        upper_exact = fn(interval.upper[idx])
        assert output.lower[idx] <= lower_exact
        assert output.upper[idx] >= upper_exact


def test_interval_add_encloses_branch_sum_corners() -> None:
    model = IntervalAdd(
        nn.Sequential(nn.Linear(2, 2), nn.Tanh()),
        nn.Sequential(nn.Linear(2, 2), nn.LeakyReLU(negative_slope=0.1)),
    )
    with torch.no_grad():
        model.left[0].weight.copy_(torch.tensor([[1.0, -0.5], [0.5, 1.0]]))
        model.left[0].bias.copy_(torch.tensor([0.2, -0.1]))
        model.right[0].weight.copy_(torch.tensor([[0.25, 1.5], [-1.0, 0.5]]))
        model.right[0].bias.copy_(torch.tensor([0.0, 0.3]))

    interval = IntervalTensor.from_bounds([-1.0, 0.25], [0.5, 1.0])
    output = interval_forward(model, interval)

    corners = [
        (x0, x1)
        for x0 in (interval.lower[0], interval.upper[0])
        for x1 in (interval.lower[1], interval.upper[1])
    ]
    eval_dtype = model.left[0].weight.dtype
    for corner in corners:
        value = model(torch.tensor(corner, dtype=eval_dtype)).detach().tolist()
        for idx, exact in enumerate(value):
            assert output.lower[idx] <= exact
            assert output.upper[idx] >= exact


def test_interval_cat_combines_branch_outputs() -> None:
    model = IntervalCat(
        nn.Sequential(nn.Linear(2, 1), nn.Softplus()),
        nn.Sequential(nn.Linear(2, 2), nn.Tanh()),
        dim=-1,
    )
    with torch.no_grad():
        model.branches[0][0].weight.copy_(torch.tensor([[1.0, -1.0]]))
        model.branches[0][0].bias.copy_(torch.tensor([0.1]))
        model.branches[1][0].weight.copy_(torch.tensor([[0.5, 0.25], [-0.75, 1.25]]))
        model.branches[1][0].bias.copy_(torch.tensor([0.0, -0.2]))

    interval = IntervalTensor.from_bounds([-0.5, -1.0], [1.0, 0.75])
    output = interval_forward(model, interval)

    assert len(output.lower) == 3
    assert len(output.upper) == 3

    corners = [
        (x0, x1)
        for x0 in (interval.lower[0], interval.upper[0])
        for x1 in (interval.lower[1], interval.upper[1])
    ]
    eval_dtype = model.branches[0][0].weight.dtype
    for corner in corners:
        value = model(torch.tensor(corner, dtype=eval_dtype)).detach().tolist()
        for idx, exact in enumerate(value):
            assert output.lower[idx] <= exact
            assert output.upper[idx] >= exact


def test_interval_cat_rejects_unsupported_dim() -> None:
    model = IntervalCat(nn.Identity(), nn.Identity(), dim=1)
    interval = IntervalTensor.from_bounds([0.0, 1.0], [0.5, 1.5])

    with pytest.raises(NotImplementedError):
        _ = interval_forward(model, interval)


def test_unsupported_activation_raises_not_implemented() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ELU())
    interval = IntervalTensor.point([0.0, 1.0])
    with pytest.raises(NotImplementedError):
        _ = interval_forward(model, interval)


def test_lpnorm_zero_network_returns_zero_interval() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()

    domain = IntervalTensor.from_bounds([-1.0, -2.0], [3.0, 4.0])
    bounds = model.lpnorm(domain, p=2.0, iterations=6)

    assert bounds.lower <= 0.0 <= bounds.upper
    assert bounds.upper < 1e-10


def test_interval_pow_scalar_accepts_outward_rounded_near_zero_lower_bound() -> None:
    near_zero = Interval.from_bounds(0.0, 2.0)

    result = _interval_pow_scalar(near_zero, exponent=2.0)

    assert result.lower <= 0.0 <= result.upper


def test_interval_pow_scalar_handles_fractional_exponent_near_zero() -> None:
    near_zero = Interval.from_bounds(0.0, 0.0)

    result = _interval_pow_scalar(near_zero, exponent=0.5)

    assert result.lower <= 0.0 <= result.upper


def test_lpnorm_constant_network_matches_exact_value() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(3.0)

    domain = IntervalTensor.from_bounds([0.0], [2.0])
    bounds = model.lpnorm(domain, p=2.0, iterations=4)
    exact = (18.0) ** 0.5

    assert bounds.lower <= exact <= bounds.upper
    assert (bounds.upper - bounds.lower) < 1e-10


def test_lpnorm_refinement_tightens_interval() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
    torch.manual_seed(2)
    for parameter in model.parameters():
        nn.init.uniform_(parameter, a=-1.0, b=1.0)

    domain = IntervalTensor.from_bounds([-1.0], [1.0])
    coarse = model.lpnorm(domain, p=2.0, iterations=0)
    refined = model.lpnorm(domain, p=2.0, iterations=8)

    assert refined.lower >= coarse.lower
    assert refined.upper <= coarse.upper
    assert (refined.upper - refined.lower) <= (coarse.upper - coarse.lower)


def test_lpnorm_accepts_dorfler_theta_parameter() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1))
    torch.manual_seed(17)
    for parameter in model.parameters():
        nn.init.uniform_(parameter, a=-1.0, b=1.0)

    domain = IntervalTensor.from_bounds([-1.0], [1.0])
    bounds = model.lpnorm(domain, p=2.0, iterations=4, theta=0.5)

    assert bounds.lower <= bounds.upper


def test_lpnorm_rejects_invalid_dorfler_theta() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    domain = IntervalTensor.from_bounds([0.0], [1.0])

    with pytest.raises(ValueError):
        _ = model.lpnorm(domain, p=2.0, iterations=1, theta=0.0)
    with pytest.raises(ValueError):
        _ = model.lpnorm(domain, p=2.0, iterations=1, theta=1.5)


def test_lpnorm_contains_monte_carlo_estimate() -> None:
    enable_interval_eval()
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 1))

    domain = IntervalTensor.from_bounds([-1.0, -0.5], [1.0, 1.5])
    bounds = model.lpnorm(domain, p=2.0, iterations=10)

    sample_count = 40000
    with torch.no_grad():
        samples = torch.rand(sample_count, 2, dtype=torch.float64)
        samples[:, 0] = 2.0 * samples[:, 0] - 1.0
        samples[:, 1] = 2.0 * samples[:, 1] + (-0.5)
        values = model(samples.to(dtype=torch.float32)).to(dtype=torch.float64)
        integrand = torch.abs(values.squeeze(-1)) ** 2.0
        volume = (1.0 - (-1.0)) * (1.5 - (-0.5))
        estimate = float((volume * torch.mean(integrand)).sqrt().item())

    assert bounds.lower <= estimate <= bounds.upper


def test_eval_jacobian_linear_layer_matches_exact_weight_matrix() -> None:
    enable_interval_eval()
    layer = nn.Linear(2, 2)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, -1.0], [0.5, 3.0]]))
        layer.bias.copy_(torch.tensor([1.0, -2.0]))

    domain = IntervalTensor.from_bounds([-1.0, -2.0], [2.0, 4.0])
    jacobian = layer.eval_jacobian(domain)

    expected = [[2.0, -1.0], [0.5, 3.0]]
    for row_idx, row in enumerate(expected):
        for col_idx, exact in enumerate(row):
            assert jacobian.lower[row_idx][col_idx] <= exact <= jacobian.upper[row_idx][col_idx]


def test_eval_jacobian_relu_derivative_crossing_zero_is_interval() -> None:
    enable_interval_eval()
    relu = nn.ReLU()
    domain = IntervalTensor.from_bounds([-2.0, 0.25], [1.0, 2.5])

    jacobian = relu.eval_jacobian(domain)

    assert jacobian.lower[0][0] <= 0.0
    assert jacobian.upper[0][0] >= 1.0
    assert jacobian.lower[1][1] <= 1.0 <= jacobian.upper[1][1]
    assert jacobian.lower[0][1] <= 0.0 <= jacobian.upper[0][1]
    assert jacobian.lower[1][0] <= 0.0 <= jacobian.upper[1][0]


def test_eval_jacobian_sequential_encloses_corner_gradients() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[2.0], [-1.5]]))
        model[0].bias.copy_(torch.tensor([0.25, -0.5]))
        model[2].weight.copy_(torch.tensor([[1.2, -0.7]]))
        model[2].bias.copy_(torch.tensor([0.0]))

    domain = IntervalTensor.from_bounds([-1.0], [1.0])
    jacobian = model.eval_jacobian(domain)

    slopes = []
    for endpoint in (-1.0, 1.0):
        point = torch.tensor([[endpoint]], dtype=torch.float64, requires_grad=True)
        output = model(point.to(dtype=torch.float32)).to(dtype=torch.float64)
        grad = torch.autograd.grad(output.squeeze(), point)[0].item()
        slopes.append(grad)

    assert jacobian.lower[0][0] <= min(slopes)
    assert jacobian.upper[0][0] >= max(slopes)


def test_sobolev_norm_constant_network_matches_closed_form() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(3.0)

    domain = IntervalTensor.from_bounds([0.0], [2.0])
    bounds = model.sobolev_norm(domain, p=2.0, iterations=4)
    exact = (18.0) ** 0.5

    assert bounds.lower <= exact <= bounds.upper
    assert (bounds.upper - bounds.lower) < 1e-10


def test_sobolev_norm_refinement_tightens_interval() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 6), nn.ReLU(), nn.Linear(6, 1))
    torch.manual_seed(13)
    for parameter in model.parameters():
        nn.init.uniform_(parameter, a=-1.0, b=1.0)

    domain = IntervalTensor.from_bounds([-1.0], [1.0])
    coarse = model.sobolev_norm(domain, p=2.0, iterations=0)
    refined = model.sobolev_norm(domain, p=2.0, iterations=7)

    assert refined.lower >= coarse.lower
    assert refined.upper <= coarse.upper
    assert (refined.upper - refined.lower) <= (coarse.upper - coarse.lower)


def test_sobolev_norm_accepts_dorfler_theta_parameter() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 5), nn.ReLU(), nn.Linear(5, 1))
    torch.manual_seed(29)
    for parameter in model.parameters():
        nn.init.uniform_(parameter, a=-1.0, b=1.0)

    domain = IntervalTensor.from_bounds([-1.0], [1.0])
    bounds = model.sobolev_norm(domain, p=2.0, iterations=3, theta=0.5)

    assert bounds.lower <= bounds.upper


def test_sobolev_norm_rejects_invalid_dorfler_theta() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    domain = IntervalTensor.from_bounds([0.0], [1.0])

    with pytest.raises(ValueError):
        _ = model.sobolev_norm(domain, p=2.0, iterations=1, theta=0.0)
    with pytest.raises(ValueError):
        _ = model.sobolev_norm(domain, p=2.0, iterations=1, theta=1.1)


def test_sigmoid_interval_encloses_endpoint_images() -> None:
    sigmoid = nn.Sigmoid()
    interval = IntervalTensor.from_bounds([-3.0, -0.25, 2.0], [-1.0, 0.5, 4.0])

    output = interval_forward(sigmoid, interval)
    exact_lower = torch.sigmoid(torch.tensor(interval.lower, dtype=torch.float64)).tolist()
    exact_upper = torch.sigmoid(torch.tensor(interval.upper, dtype=torch.float64)).tolist()

    for idx in range(len(exact_lower)):
        assert output.lower[idx] <= exact_lower[idx]
        assert output.upper[idx] >= exact_upper[idx]


def test_flatten_forwards_bounds_unchanged_for_flat_vectors() -> None:
    flatten = nn.Flatten()
    interval = IntervalTensor.from_bounds([-1.0, 2.0, 3.5], [0.0, 4.0, 5.5])

    output = interval_forward(flatten, interval)

    assert output.lower == interval.lower
    assert output.upper == interval.upper


def test_softmax_empty_vector_rejected() -> None:
    softmax = nn.Softmax(dim=-1)
    empty = IntervalTensor.from_bounds([], [])
    with pytest.raises(ValueError):
        _ = interval_forward(softmax, empty)


def test_softmax_invalid_dim_rejected_for_vector_input() -> None:
    softmax = nn.Softmax(dim=1)
    interval = IntervalTensor.from_bounds([-1.0, 2.0], [1.0, 3.0])
    with pytest.raises(NotImplementedError):
        _ = interval_forward(softmax, interval)


def test_enable_interval_eval_is_idempotent_and_preserves_standard_eval() -> None:
    enable_interval_eval()
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(2, 1))
    result = model.eval()
    assert result is model


def test_eval_interval_argument_requires_interval_tensor() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with pytest.raises(TypeError):
        _ = model.eval([0.0, 1.0])


def test_lpnorm_rejects_invalid_parameters() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    domain = IntervalTensor.from_bounds([0.0], [1.0])

    with pytest.raises(ValueError):
        _ = model.lpnorm(domain, p=0.0, iterations=0)
    with pytest.raises(ValueError):
        _ = model.lpnorm(domain, p=float("nan"), iterations=0)
    with pytest.raises(ValueError):
        _ = model.lpnorm(domain, p=2.0, iterations=-1)


def test_sobolev_norm_rejects_invalid_parameters() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    domain = IntervalTensor.from_bounds([0.0], [1.0])

    with pytest.raises(ValueError):
        _ = model.sobolev_norm(domain, p=0.0, iterations=0)
    with pytest.raises(ValueError):
        _ = model.sobolev_norm(domain, p=float("inf"), iterations=0)
    with pytest.raises(ValueError):
        _ = model.sobolev_norm(domain, p=2.0, iterations=-1)


def test_lpnorm_requires_interval_tensor_domain() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with pytest.raises(TypeError):
        _ = model.lpnorm([0.0, 1.0], p=2.0, iterations=0)


def test_sobolev_norm_requires_interval_tensor_domain() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with pytest.raises(TypeError):
        _ = model.sobolev_norm([0.0, 1.0], p=2.0, iterations=0)


def test_eval_jacobian_requires_interval_tensor_domain() -> None:
    enable_interval_eval()
    model = nn.Sequential(nn.Linear(1, 1))
    with pytest.raises(TypeError):
        _ = model.eval_jacobian([0.0, 1.0])


def test_softmax_jacobian_encloses_autograd_corner_gradients() -> None:
    enable_interval_eval()
    softmax = nn.Softmax(dim=-1)
    domain = IntervalTensor.from_bounds([-1.0, 0.1, 1.5], [0.5, 1.0, 2.5])

    jacobian = softmax.eval_jacobian(domain)

    corners = [
        torch.tensor(values, dtype=torch.float32)
        for values in [
            [domain.lower[0], domain.lower[1], domain.lower[2]],
            [domain.lower[0], domain.lower[1], domain.upper[2]],
            [domain.lower[0], domain.upper[1], domain.lower[2]],
            [domain.lower[0], domain.upper[1], domain.upper[2]],
            [domain.upper[0], domain.lower[1], domain.lower[2]],
            [domain.upper[0], domain.lower[1], domain.upper[2]],
            [domain.upper[0], domain.upper[1], domain.lower[2]],
            [domain.upper[0], domain.upper[1], domain.upper[2]],
        ]
    ]
    for point in corners:
        point = point.clone().detach().requires_grad_(True)
        output = softmax(point)
        for row in range(3):
            grad = torch.autograd.grad(output[row], point, retain_graph=True)[0]
            for col in range(3):
                exact = float(grad[col].item())
                assert jacobian.lower[row][col] <= exact <= jacobian.upper[row][col]


def test_sigmoid_jacobian_encloses_autograd_corner_gradients() -> None:
    enable_interval_eval()
    sigmoid = nn.Sigmoid()
    domain = IntervalTensor.from_bounds([-10.0, -0.25], [0.5, 12.0])

    jacobian = sigmoid.eval_jacobian(domain)

    for x0 in (domain.lower[0], domain.upper[0]):
        for x1 in (domain.lower[1], domain.upper[1]):
            point = torch.tensor([x0, x1], dtype=torch.float32, requires_grad=True)
            output = sigmoid(point)
            for row in range(2):
                grad = torch.autograd.grad(output[row], point, retain_graph=True)[0]
                for col in range(2):
                    exact = float(grad[col].item())
                    assert jacobian.lower[row][col] <= exact <= jacobian.upper[row][col]


def test_tanh_jacobian_encloses_autograd_corner_gradients() -> None:
    enable_interval_eval()
    tanh = nn.Tanh()
    domain = IntervalTensor.from_bounds([-2.0, -0.25], [0.5, 1.5])

    jacobian = tanh.eval_jacobian(domain)

    for x0 in (domain.lower[0], domain.upper[0]):
        for x1 in (domain.lower[1], domain.upper[1]):
            point = torch.tensor([x0, x1], dtype=torch.float32, requires_grad=True)
            output = tanh(point)
            for row in range(2):
                grad = torch.autograd.grad(output[row], point, retain_graph=True)[0]
                for col in range(2):
                    exact = float(grad[col].item())
                    assert jacobian.lower[row][col] <= exact <= jacobian.upper[row][col]
