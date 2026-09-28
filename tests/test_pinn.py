import pytest

torch = pytest.importorskip("torch")

from intervalnets import load_tanh_mlp_checkpoint, sequential_value_jacobian_laplacian


def test_load_tanh_mlp_checkpoint_reconstructs_saved_network(tmp_path):
    torch.manual_seed(13)
    expected = torch.nn.Sequential(
        torch.nn.Linear(4, 6),
        torch.nn.Tanh(),
        torch.nn.Linear(6, 3),
        torch.nn.Tanh(),
        torch.nn.Linear(3, 1),
    ).double()
    checkpoint = tmp_path / "pinn.pt"
    torch.save({"state_dict": expected.state_dict(), "seed": 13}, checkpoint)

    loaded = load_tanh_mlp_checkpoint(checkpoint)
    x = torch.randn(7, 4, dtype=torch.float64)

    assert isinstance(loaded, torch.nn.Sequential)
    assert [type(layer) for layer in loaded] == [
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
    ]
    assert not loaded.training
    assert torch.equal(loaded(x), expected(x))


def test_load_tanh_mlp_checkpoint_requires_explicit_regeneration(tmp_path):
    with pytest.raises(FileNotFoundError, match="RETRAIN = True"):
        load_tanh_mlp_checkpoint(tmp_path / "missing.pt")


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
