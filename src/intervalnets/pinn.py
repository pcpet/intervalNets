"""Efficient differential propagation helpers for PINN training."""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - PyTorch is an optional dependency
    torch = None
    nn = None


def sequential_value_jacobian_laplacian(
    module: Any,
    x: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Evaluate a tanh MLP together with its Jacobian and Laplacian.

    The routine propagates first derivatives and the trace of the Hessian
    directly through ``Linear`` and ``Tanh`` layers.  It is differentiable
    with respect to the network parameters, so the returned Laplacian can be
    used in an ordinary PINN residual without performing one second-order
    autograd call per input coordinate.

    Parameters
    ----------
    module:
        A ``torch.nn.Sequential`` tanh MLP containing ``Linear``, ``Tanh``,
        and optional ``Identity`` modules.
    x:
        A batched tensor with shape ``(..., input_dim)``.

    Returns
    -------
    value:
        Network output with shape ``(..., output_dim)``.
    jacobian:
        Physical-input Jacobian with shape
        ``(..., output_dim, input_dim)``.
    laplacian:
        Componentwise Hessian trace with shape ``(..., output_dim)``.
    """

    if torch is None or nn is None:  # pragma: no cover - optional dependency
        raise ImportError("PyTorch is required for PINN differential propagation.")
    if not isinstance(module, nn.Sequential):
        raise TypeError("module must be a torch.nn.Sequential tanh MLP.")
    if not isinstance(x, torch.Tensor) or x.ndim < 1:
        raise TypeError("x must be a torch.Tensor with a final input dimension.")

    value = x
    jacobian = None
    laplacian = None
    input_dim = int(x.shape[-1])

    for layer in module:
        if isinstance(layer, nn.Linear):
            if int(layer.in_features) != int(value.shape[-1]):
                raise ValueError("Linear layer width does not match the propagated value.")
            value = layer(value)
            if jacobian is None:
                if int(layer.in_features) != input_dim:
                    raise ValueError("The first Linear layer must consume the physical input.")
                batch_shape = tuple(x.shape[:-1])
                jacobian = layer.weight.reshape(
                    (1,) * len(batch_shape) + tuple(layer.weight.shape)
                ).expand(batch_shape + tuple(layer.weight.shape))
                laplacian = torch.zeros_like(value)
            else:
                jacobian = torch.einsum("oi,...id->...od", layer.weight, jacobian)
                laplacian = torch.einsum("oi,...i->...o", layer.weight, laplacian)
        elif isinstance(layer, nn.Tanh):
            if jacobian is None or laplacian is None:
                raise ValueError("Tanh cannot precede the first Linear layer.")
            activated = torch.tanh(value)
            first = 1.0 - activated.square()
            second = -2.0 * activated * first
            laplacian = second * jacobian.square().sum(dim=-1) + first * laplacian
            jacobian = first.unsqueeze(-1) * jacobian
            value = activated
        elif isinstance(layer, nn.Identity):
            continue
        else:
            raise NotImplementedError(
                "PINN differential propagation supports only Sequential, Linear, "
                f"Tanh, and Identity modules; got {type(layer).__name__}."
            )

    if jacobian is None or laplacian is None:
        raise ValueError("The network must contain at least one Linear layer.")
    return value, jacobian, laplacian
