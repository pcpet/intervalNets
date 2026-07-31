import pytest

torch = pytest.importorskip("torch")

from intervalnets import sequential_value_jacobian_laplacian


def test_sequential_value_jacobian_laplacian_matches_autograd():
    torch.manual_seed(17)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 2),
    ).double()
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

    value, jacobian, laplacian = sequential_value_jacobian_laplacian(model, x)

    expected_jacobian = []
    expected_laplacian = []
    direct = model(x)
    for output in range(direct.shape[-1]):
        gradient = torch.autograd.grad(
            direct[:, output].sum(), x, create_graph=True, retain_graph=True
        )[0]
        trace = torch.zeros(x.shape[0], dtype=x.dtype)
        for coordinate in range(x.shape[-1]):
            second = torch.autograd.grad(
                gradient[:, coordinate].sum(),
                x,
                retain_graph=True,
            )[0][:, coordinate]
            trace = trace + second
        expected_jacobian.append(gradient)
        expected_laplacian.append(trace)

    expected_jacobian = torch.stack(expected_jacobian, dim=1)
    expected_laplacian = torch.stack(expected_laplacian, dim=1)
    assert torch.allclose(value, direct, rtol=1e-12, atol=1e-12)
    assert torch.allclose(jacobian, expected_jacobian, rtol=1e-11, atol=1e-12)
    assert torch.allclose(laplacian, expected_laplacian, rtol=1e-10, atol=1e-12)


def test_laplacian_remains_differentiable_with_respect_to_parameters():
    torch.manual_seed(19)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
    ).double()
    x = torch.randn(6, 2, dtype=torch.float64)
    value, _, laplacian = sequential_value_jacobian_laplacian(model, x)
    (value.square().mean() + laplacian.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
