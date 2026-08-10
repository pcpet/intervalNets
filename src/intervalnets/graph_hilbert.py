"""Dependency-preserving graph moments and Hilbert norm certificates.

The scalable certificate in this module deliberately separates two concerns.
Small arithmetic circuits can be expanded by :class:`SparseReferenceMomentBackend`
to provide an exact regression oracle.  Large value circuits are compressed at
activation boundaries to their orthogonal affine projection in
``L2([-1,1]^d)`` plus a certified Hilbert remainder.  The discarded part is
never re-labelled as independent pointwise noise.

For ``W^{1,2}`` lower bounds we use Neumann-compatible polynomial witnesses.
Integration by parts then reduces the uncertain gradient pairing to a
value-only pairing, for which the Hilbert remainder is directly applicable.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from math import comb, inf, nextafter, sqrt
from typing import Any, Iterable, Literal

from .interval import Interval
from .polynomial_zonotope import PolynomialZonotope, box_monomial_moment
from .pz_integration import PZIntegrationCell
from .pz_tanh import compute_tanh_polynomial, quadratic_tanh_prime_enclosure

try:  # pragma: no cover - optional dependencies
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class GraphNode:
    """One immutable arithmetic-circuit node."""

    id: int
    operation: str
    children: tuple[int, ...]
    payload: Any
    shape: tuple[int, ...]
    degree: int
    domain_support: frozenset[int]
    residual_support: frozenset[int]


@dataclass(frozen=True)
class ArithmeticGraph:
    """Hash-consed dependency graph with one designated output node."""

    nodes: tuple[GraphNode, ...]
    output_id: int
    num_domain_noise: int
    num_noise: int

    def evaluate(self, noise: Any) -> Any:
        if torch is None:
            raise ImportError("PyTorch is required for arithmetic-graph evaluation.")
        values = torch.as_tensor(noise)
        squeeze = values.ndim == 1
        if squeeze:
            values = values.unsqueeze(0)
        if values.shape[-1] != self.num_noise:
            raise ValueError(
                f"Expected {self.num_noise} noise coordinates, got {values.shape[-1]}."
            )
        cache: dict[int, Any] = {}
        for node in self.nodes:
            children = [cache[index] for index in node.children]
            if node.operation == "noise":
                value = values
            elif node.operation == "slice":
                start, stop = node.payload
                value = children[0][..., start:stop]
            elif node.operation == "constant":
                value = node.payload.to(dtype=values.dtype, device=values.device)
            elif node.operation == "add":
                value = children[0] + children[1]
            elif node.operation == "hadamard":
                value = children[0] * children[1]
            elif node.operation == "square":
                value = children[0].square()
            elif node.operation == "scale":
                value = children[0] * node.payload.to(
                    dtype=values.dtype, device=values.device
                )
            elif node.operation == "linear_map":
                matrix = node.payload.to(dtype=values.dtype, device=values.device)
                value = children[0] @ matrix.T
            else:  # pragma: no cover - builder prevents this
                raise RuntimeError(f"Unknown graph operation {node.operation!r}.")
            cache[node.id] = value
        result = cache[self.output_id]
        return result.squeeze(0) if squeeze else result


def _payload_key(payload: Any) -> Any:
    if torch is not None and isinstance(payload, torch.Tensor):
        array = payload.detach().cpu().contiguous().numpy()
        return (tuple(array.shape), str(array.dtype), sha1(array.tobytes()).digest())
    return payload


class _GraphBuilder:
    def __init__(self, num_domain_noise: int, num_noise: int):
        self.num_domain_noise = int(num_domain_noise)
        self.num_noise = int(num_noise)
        self.nodes: list[GraphNode] = []
        self.cache: dict[Any, int] = {}

    def node(
        self,
        operation: str,
        children: Iterable[int] = (),
        payload: Any = None,
        *,
        shape: tuple[int, ...],
        degree: int,
        domain_support: Iterable[int] = (),
        residual_support: Iterable[int] = (),
    ) -> int:
        child_tuple = tuple(children)
        key = (operation, child_tuple, _payload_key(payload), shape)
        if key in self.cache:
            return self.cache[key]
        node_id = len(self.nodes)
        node = GraphNode(
            id=node_id,
            operation=operation,
            children=child_tuple,
            payload=payload,
            shape=shape,
            degree=int(degree),
            domain_support=frozenset(domain_support),
            residual_support=frozenset(residual_support),
        )
        self.nodes.append(node)
        self.cache[key] = node_id
        return node_id

    def unary(self, operation: str, child: int, payload: Any = None) -> int:
        source = self.nodes[child]
        return self.node(
            operation,
            (child,),
            payload,
            shape=source.shape,
            degree=2 * source.degree if operation == "square" else source.degree,
            domain_support=source.domain_support,
            residual_support=source.residual_support,
        )

    def binary(self, operation: str, left: int, right: int) -> int:
        lhs, rhs = self.nodes[left], self.nodes[right]
        if lhs.shape != rhs.shape:
            raise ValueError("Binary graph operations require equal shapes.")
        return self.node(
            operation,
            (left, right),
            shape=lhs.shape,
            degree=(lhs.degree + rhs.degree if operation == "hadamard" else max(lhs.degree, rhs.degree)),
            domain_support=lhs.domain_support | rhs.domain_support,
            residual_support=lhs.residual_support | rhs.residual_support,
        )


def build_factored_jacobian_graph(jacobian: Any) -> ArithmeticGraph:
    """Translate ``FactoredPolynomialJacobian`` to an exact immutable graph."""

    if torch is None:
        raise ImportError("PyTorch is required for factored graph construction.")
    builder = _GraphBuilder(jacobian.num_domain_noise, jacobian.num_noise)
    noise = builder.node(
        "noise",
        shape=(jacobian.num_noise,),
        degree=1,
        domain_support=range(jacobian.num_domain_noise),
        residual_support=range(jacobian.num_domain_noise, jacobian.num_noise),
    )

    def constant(value: Any) -> int:
        tensor = value.detach().clone()
        return builder.node(
            "constant", payload=tensor, shape=tuple(tensor.shape), degree=0
        )

    def slice_node(start: int, stop: int) -> int:
        residual = range(max(start, jacobian.num_domain_noise), stop)
        domain = range(start, min(stop, jacobian.num_domain_noise))
        return builder.node(
            "slice",
            (noise,),
            (start, stop),
            shape=(stop - start,),
            degree=1,
            domain_support=domain,
            residual_support=residual,
        )

    def linear(child: int, matrix: Any) -> int:
        source = builder.nodes[child]
        matrix = matrix.detach().clone()
        return builder.node(
            "linear_map",
            (child,),
            matrix,
            shape=(matrix.shape[0],),
            degree=source.degree,
            domain_support=source.domain_support,
            residual_support=source.residual_support,
        )

    adjoint = constant(jacobian.output_weight)
    derivative_base = jacobian.num_value_noise
    for layer_index in range(len(jacobian.factors) - 1, -1, -1):
        factor = jacobian.factors[layer_index]
        xi = slice_node(0, factor.active_value_noise)
        affine_argument = linear(xi, factor.preactivation_coefficients)
        derivative = builder.binary(
            "add", constant(factor.center), linear(xi, factor.linear_coefficients)
        )
        quadratic = builder.unary(
            "scale",
            builder.unary("square", affine_argument),
            factor.quadratic_coefficients.detach().clone(),
        )
        derivative = builder.binary("add", derivative, quadratic)
        width = factor.center.numel()
        eta = slice_node(
            derivative_base + factor.derivative_noise_offset,
            derivative_base + factor.derivative_noise_offset + width,
        )
        derivative = builder.binary(
            "add",
            derivative,
            builder.unary(
                "scale", eta, factor.approximation_radii.detach().clone()
            ),
        )
        adjoint = builder.binary("hadamard", adjoint, derivative)
        if layer_index:
            adjoint = linear(adjoint, jacobian.hidden_weights[layer_index - 1].T)
    output_id = linear(adjoint, jacobian.input_weight.T)
    return ArithmeticGraph(
        nodes=tuple(builder.nodes),
        output_id=output_id,
        num_domain_noise=jacobian.num_domain_noise,
        num_noise=jacobian.num_noise,
    )


def _add_polynomials(left: dict[tuple[int, ...], Any], right: dict[tuple[int, ...], Any]):
    result = {key: value.clone() for key, value in left.items()}
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, torch.zeros_like(coefficient)) + coefficient
        if not torch.count_nonzero(result[exponent]):
            result.pop(exponent)
    return result


class SparseReferenceMomentBackend:
    """Exact domain-only sparse expansion used as a small-case oracle."""

    def __init__(self, graph: ArithmeticGraph):
        if torch is None:
            raise ImportError("PyTorch is required for sparse reference moments.")
        self.graph = graph
        self._cache: dict[int, dict[tuple[int, ...], Any]] = {}
        self.expanded_term_pairs = 0

    def expand(self, node_id: int | None = None) -> dict[tuple[int, ...], Any]:
        node_id = self.graph.output_id if node_id is None else int(node_id)
        if node_id in self._cache:
            return self._cache[node_id]
        node = self.graph.nodes[node_id]
        children = [self.expand(index) for index in node.children]
        zero = (0,) * self.graph.num_domain_noise
        if node.operation == "noise":
            template = next(
                item.payload
                for item in self.graph.nodes
                if item.operation == "constant"
            )
            result = {}
            for index in range(self.graph.num_domain_noise):
                exponent = [0] * self.graph.num_domain_noise
                exponent[index] = 1
                coefficient = torch.zeros(
                    self.graph.num_noise,
                    dtype=template.dtype,
                    device=template.device,
                )
                coefficient[index] = 1.0
                result[tuple(exponent)] = coefficient
        elif node.operation == "slice":
            start, stop = node.payload
            result = {key: value[start:stop] for key, value in children[0].items()}
        elif node.operation == "constant":
            result = {zero: node.payload}
        elif node.operation == "add":
            result = _add_polynomials(children[0], children[1])
        elif node.operation in {"hadamard", "square"}:
            left = children[0]
            right = children[0] if node.operation == "square" else children[1]
            result: dict[tuple[int, ...], Any] = {}
            for left_exp, left_coeff in left.items():
                for right_exp, right_coeff in right.items():
                    self.expanded_term_pairs += 1
                    exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
                    coefficient = left_coeff * right_coeff
                    result[exponent] = result.get(
                        exponent, torch.zeros_like(coefficient)
                    ) + coefficient
        elif node.operation == "scale":
            result = {key: value * node.payload for key, value in children[0].items()}
        elif node.operation == "linear_map":
            result = {key: value @ node.payload.T for key, value in children[0].items()}
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported reference operation {node.operation!r}.")
        self._cache[node_id] = result
        return result

    def sum_squares(self, node_id: int | None = None) -> float:
        polynomial = self.expand(node_id)
        total = 0.0
        for left_exp, left_coeff in polynomial.items():
            for right_exp, right_coeff in polynomial.items():
                exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
                moment = box_monomial_moment(exponent) / (2.0**len(exponent))
                total += float(torch.dot(left_coeff.reshape(-1), right_coeff.reshape(-1))) * moment
        return total

    @property
    def diagnostics(self) -> dict[str, int]:
        return {
            "graph_nodes": len(self.graph.nodes),
            "expanded_nodes": len(self._cache),
            "output_terms": len(self.expand()),
            "expanded_term_pairs": self.expanded_term_pairs,
        }


@dataclass(frozen=True)
class HilbertLayerDiagnostic:
    layer: int
    width: int
    preactivation_remainder: float
    polynomial_projection_remainder: float
    uniform_approximation_remainder: float
    total_output_remainder: float
    maximum_preactivation_width: float


@dataclass(frozen=True)
class HilbertValueCertificate:
    """Affine value projection plus a certified normalized-L2 remainder."""

    center: Any
    domain_coefficients: Any
    remainder: float
    polynomial_degree: int
    residual_subdivisions: int
    layers: tuple[HilbertLayerDiagnostic, ...]
    moment_states: int
    preactivation_centers: tuple[Any, ...]
    preactivation_coefficients: tuple[Any, ...]
    preactivation_remainders: tuple[float, ...]

    @property
    def nominal_norm(self) -> float:
        value = self.center.square() + self.domain_coefficients.square().sum() / 3.0
        return sqrt(max(0.0, float(value.detach().cpu().item())))


@dataclass(frozen=True)
class DualWitnessCertificate:
    lower_bound: float
    nominal_pairing: float
    remainder_penalty: float
    witness_norm: float
    transformed_witness_norm: float
    coefficients: tuple[float, ...]
    basis: str


@dataclass(frozen=True)
class HilbertGradientCertificate:
    """Affine gradient projection plus normalized-L2 vector remainder."""

    center: Any
    domain_coefficients: Any
    remainder: float
    factor_remainders: tuple[float, ...]
    product_projection_remainders: tuple[float, ...]
    moment_states: int

    @property
    def nominal_norm(self) -> float:
        value = torch.dot(self.center, self.center) + self.domain_coefficients.square().sum() / 3.0
        return sqrt(max(0.0, float(value.detach().cpu().item())))


@dataclass(frozen=True)
class GraphNormCertificate:
    l2_squared: Interval
    w12_squared: Interval
    value: HilbertValueCertificate
    gradient: HilbertGradientCertificate
    w12_witness: DualWitnessCertificate
    previous_l2_squared: Interval
    previous_w12_squared: Interval
    normalized_diagnostics: dict[str, float]


def _require_numeric_dependencies() -> None:
    if torch is None or nn is None or np is None:
        raise ImportError("PyTorch and NumPy are required for Hilbert graph certification.")


def _tanh_scalar_linears(module: Any) -> list[Any]:
    children = list(module.children()) if isinstance(module, nn.Sequential) else []
    if len(children) < 3 or len(children) % 2 != 1:
        raise ValueError("Expected Linear/Tanh repetitions followed by scalar Linear.")
    linears: list[Any] = []
    for index, child in enumerate(children[:-1]):
        expected = nn.Linear if index % 2 == 0 else nn.Tanh
        if not isinstance(child, expected):
            raise ValueError("Expected Linear/Tanh repetitions followed by scalar Linear.")
        if index % 2 == 0:
            linears.append(child)
    output = children[-1]
    if not isinstance(output, nn.Linear) or output.out_features != 1:
        raise ValueError("The Hilbert graph certificate requires scalar output.")
    linears.append(output)
    return linears


def _affine_domain_matrix(domain: PolynomialZonotope) -> Any:
    if len(domain.shape) != 1:
        raise ValueError("The Hilbert graph certificate requires a flat affine domain.")
    columns: list[Any] = []
    for noise_index in range(domain.num_noise):
        exponent = [0] * domain.num_noise
        exponent[noise_index] = 1
        coefficient = domain.terms.get(tuple(exponent))
        if coefficient is None:
            raise ValueError("Every domain coordinate must have one affine generator.")
        columns.append(
            coefficient
            if isinstance(coefficient, torch.Tensor)
            else torch.as_tensor(coefficient, dtype=torch.float64)
        )
    if len(domain.terms) != len(columns) or any(
        kind != "domain" for kind in domain.noise_kinds
    ):
        raise ValueError("Only affine domain generators are supported.")
    return torch.stack(columns, dim=1)


def _spectral_norm_upper(matrix: Any) -> float:
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    reconstructed = (u * singular.unsqueeze(0)) @ vh
    reconstruction_error = float(
        torch.linalg.vector_norm(matrix - reconstructed).item()
    )
    identity = torch.eye(
        singular.numel(), dtype=matrix.dtype, device=matrix.device
    )
    u_error = float(torch.linalg.vector_norm(u.T @ u - identity).item())
    v_error = float(torch.linalg.vector_norm(vh @ vh.T - identity).item())
    estimate = float(singular[0].item()) if singular.numel() else 0.0
    frobenius = float(torch.linalg.vector_norm(matrix).item())
    eps = torch.finfo(matrix.dtype).eps
    padding = 4096.0 * eps * max(matrix.shape, default=1) ** 2 * max(1.0, frobenius)
    return nextafter(
        sqrt(1.0 + u_error)
        * sqrt(1.0 + v_error)
        * estimate
        + reconstruction_error
        + padding,
        inf,
    )


def _convolve_moments(left: Any, right: Any) -> Any:
    order = left.shape[-1] - 1
    output = np.zeros_like(left)
    for degree in range(order + 1):
        for right_degree in range(degree + 1):
            output[..., degree] += (
                comb(degree, right_degree)
                * left[..., degree - right_degree]
                * right[..., right_degree]
            )
    return output


def _affine_raw_and_cross_moments(center: Any, coefficients: Any, order: int):
    """Return E[q^k] and E[alpha_j q^k] in extended precision."""

    c = np.asarray(center.detach().cpu().numpy(), dtype=np.longdouble)
    a = np.asarray(coefficients.detach().cpu().numpy(), dtype=np.longdouble)
    width, dimension = a.shape
    prefix = np.zeros((dimension + 1, width, order + 1), dtype=np.longdouble)
    suffix = np.zeros_like(prefix)
    for degree in range(order + 1):
        prefix[0, :, degree] = c**degree
    suffix[dimension, :, 0] = 1.0
    for coordinate in range(dimension):
        factor = np.zeros((width, order + 1), dtype=np.longdouble)
        for degree in range(0, order + 1, 2):
            factor[:, degree] = a[:, coordinate] ** degree / (degree + 1)
        prefix[coordinate + 1] = _convolve_moments(prefix[coordinate], factor)
    for coordinate in range(dimension - 1, -1, -1):
        factor = np.zeros((width, order + 1), dtype=np.longdouble)
        for degree in range(0, order + 1, 2):
            factor[:, degree] = a[:, coordinate] ** degree / (degree + 1)
        suffix[coordinate] = _convolve_moments(factor, suffix[coordinate + 1])

    cross = np.zeros((width, dimension, order + 1), dtype=np.longdouble)
    for coordinate in range(dimension):
        leave_one_out = _convolve_moments(prefix[coordinate], suffix[coordinate + 1])
        for degree in range(1, order + 1):
            for alpha_power in range(1, degree + 1, 2):
                cross[:, coordinate, degree] += (
                    comb(degree, alpha_power)
                    * a[:, coordinate] ** alpha_power
                    * leave_one_out[:, degree - alpha_power]
                    / (alpha_power + 2)
                )
    return prefix[dimension], cross, (dimension + 1) * width * (order + 1)


def _project_polynomials_of_affine_forms(
    center: Any,
    coefficients: Any,
    polynomial_coefficients: Any,
):
    degree = polynomial_coefficients.shape[1] - 1
    moments, cross, states = _affine_raw_and_cross_moments(
        center, coefficients, 2 * degree
    )
    proposal = np.asarray(polynomial_coefficients, dtype=np.longdouble)
    mean = np.sum(proposal * moments[:, : degree + 1], axis=1)
    affine_cross = np.sum(
        proposal[:, :, None] * cross[:, :, : degree + 1].transpose(0, 2, 1),
        axis=1,
    )
    linear = 3.0 * affine_cross
    squared_coefficients = np.zeros(
        (len(center), 2 * degree + 1), dtype=np.longdouble
    )
    for left_degree in range(degree + 1):
        for right_degree in range(degree + 1):
            squared_coefficients[:, left_degree + right_degree] += (
                proposal[:, left_degree] * proposal[:, right_degree]
            )
    energy = np.sum(squared_coefficients * moments, axis=1)
    projection_energy = mean**2 + np.sum(linear**2, axis=1) / 3.0
    scale = np.maximum(1.0, np.maximum(np.abs(energy), np.abs(projection_energy)))
    rounding = (
        16384.0
        * np.finfo(np.longdouble).eps
        * (coefficients.shape[1] + 1)
        * (2 * degree + 1) ** 2
        * scale
    )
    residual_squared_upper = np.maximum(0.0, energy - projection_energy + rounding)
    dtype, device = center.dtype, center.device
    rounded_mean = np.asarray(mean, dtype=np.float64)
    rounded_linear = np.asarray(linear, dtype=np.float64)
    conversion_remainder = np.sqrt(
        (mean - rounded_mean.astype(np.longdouble)) ** 2
        + np.sum(
            (linear - rounded_linear.astype(np.longdouble)) ** 2, axis=1
        )
        / 3.0
    )
    projected_center = torch.as_tensor(rounded_mean, dtype=dtype, device=device)
    projected_linear = torch.as_tensor(rounded_linear, dtype=dtype, device=device)
    projection_remainder = torch.as_tensor(
        np.asarray(np.sqrt(residual_squared_upper) + conversion_remainder, dtype=np.float64),
        dtype=dtype,
        device=device,
    )
    projection_remainder = torch.nextafter(
        projection_remainder, torch.full_like(projection_remainder, torch.inf)
    )
    return projected_center, projected_linear, projection_remainder, states


def build_hilbert_value_certificate(
    module: Any,
    domain: PolynomialZonotope,
    *,
    polynomial_degree: int = 5,
    residual_subdivisions: int = 2048,
) -> HilbertValueCertificate:
    """Compress the exact value graph to affine projection plus L2 remainder."""

    _require_numeric_dependencies()
    if polynomial_degree < 1:
        raise ValueError("polynomial_degree must be positive.")
    if residual_subdivisions < 1:
        raise ValueError("residual_subdivisions must be positive.")
    linears = _tanh_scalar_linears(module)
    parameter = next(module.parameters())
    center = torch.as_tensor(
        domain.center, dtype=parameter.dtype, device=parameter.device
    )
    coefficients = _affine_domain_matrix(domain).to(
        dtype=parameter.dtype, device=parameter.device
    )
    remainder = 0.0
    diagnostics: list[HilbertLayerDiagnostic] = []
    moment_states = 0
    cached_preactivation_centers: list[Any] = []
    cached_preactivation_coefficients: list[Any] = []
    cached_preactivation_remainders: list[float] = []
    for layer_index, layer in enumerate(linears[:-1]):
        weight = layer.weight.detach().to(dtype=center.dtype, device=center.device)
        bias = layer.bias.detach().to(dtype=center.dtype, device=center.device)
        preactivation_center = weight @ center + bias
        preactivation_coefficients = weight @ coefficients
        preactivation_remainder = _spectral_norm_upper(weight) * remainder
        cached_preactivation_centers.append(preactivation_center)
        cached_preactivation_coefficients.append(preactivation_coefficients)
        cached_preactivation_remainders.append(preactivation_remainder)
        radius = torch.sum(torch.abs(preactivation_coefficients), dim=1)
        lower = preactivation_center - radius
        upper = preactivation_center + radius
        approximations = [
            compute_tanh_polynomial(
                (float(lo), float(hi)),
                degree=polynomial_degree,
                subdivisions=residual_subdivisions,
            )
            for lo, hi in zip(lower.detach().cpu(), upper.detach().cpu())
        ]
        proposal = np.asarray(
            [approximation.coeffs for approximation in approximations],
            dtype=np.float64,
        )
        uniform = torch.as_tensor(
            [approximation.delta for approximation in approximations],
            dtype=center.dtype,
            device=center.device,
        )
        center, coefficients, projection, states = _project_polynomials_of_affine_forms(
            preactivation_center, preactivation_coefficients, proposal
        )
        moment_states += states
        local = projection + uniform
        compression_remainder = nextafter(
            float(torch.linalg.vector_norm(local).detach().cpu().item()), inf
        )
        remainder = nextafter(preactivation_remainder + compression_remainder, inf)
        diagnostics.append(
            HilbertLayerDiagnostic(
                layer=layer_index,
                width=layer.out_features,
                preactivation_remainder=preactivation_remainder,
                polynomial_projection_remainder=float(
                    torch.linalg.vector_norm(projection).detach().cpu().item()
                ),
                uniform_approximation_remainder=float(
                    torch.linalg.vector_norm(uniform).detach().cpu().item()
                ),
                total_output_remainder=remainder,
                maximum_preactivation_width=float(
                    (2.0 * radius.max()).detach().cpu().item()
                ),
            )
        )
    output = linears[-1]
    output_weight = output.weight.detach()[0].to(
        dtype=center.dtype, device=center.device
    )
    output_bias = output.bias.detach()[0].to(dtype=center.dtype, device=center.device)
    final_center = torch.dot(output_weight, center) + output_bias
    final_coefficients = output_weight @ coefficients
    final_remainder = nextafter(
        _spectral_norm_upper(output_weight.unsqueeze(0)) * remainder, inf
    )
    return HilbertValueCertificate(
        center=final_center,
        domain_coefficients=final_coefficients,
        remainder=final_remainder,
        polynomial_degree=polynomial_degree,
        residual_subdivisions=residual_subdivisions,
        layers=tuple(diagnostics),
        moment_states=moment_states,
        preactivation_centers=tuple(cached_preactivation_centers),
        preactivation_coefficients=tuple(cached_preactivation_coefficients),
        preactivation_remainders=tuple(cached_preactivation_remainders),
    )


def _project_affine_hadamard(
    left_center: Any,
    left_coefficients: Any,
    right_center: Any,
    right_coefficients: Any,
):
    """Orthogonally project componentwise products of affine forms."""

    center = left_center * right_center + torch.sum(
        left_coefficients * right_coefficients, dim=1
    ) / 3.0
    coefficients = (
        left_center.unsqueeze(1) * right_coefficients
        + right_center.unsqueeze(1) * left_coefficients
    )
    left_norm = torch.sum(left_coefficients.square(), dim=1)
    right_norm = torch.sum(right_coefficients.square(), dim=1)
    gram = torch.sum(left_coefficients * right_coefficients, dim=1)
    coordinate_overlap = torch.sum(
        left_coefficients.square() * right_coefficients.square(), dim=1
    )
    fourth = (
        (left_norm * right_norm + 2.0 * gram.square()) / 9.0
        - (2.0 / 15.0) * coordinate_overlap
    )
    energy = (
        left_center.square() * right_center.square()
        + left_center.square() * right_norm / 3.0
        + right_center.square() * left_norm / 3.0
        + 4.0 * left_center * right_center * gram / 3.0
        + fourth
    )
    projection_energy = center.square() + coefficients.square().sum(dim=1) / 3.0
    scale = torch.maximum(
        torch.ones_like(energy), torch.maximum(torch.abs(energy), torch.abs(projection_energy))
    )
    eps = torch.finfo(energy.dtype).eps
    padding = 16384.0 * eps * (left_coefficients.shape[1] + 1) * scale
    remainder = torch.sqrt(torch.clamp(energy - projection_energy + padding, min=0.0))
    remainder = torch.nextafter(remainder, torch.full_like(remainder, torch.inf))
    return center, coefficients, remainder


def build_hilbert_gradient_certificate(
    module: Any,
    value: HilbertValueCertificate,
    *,
    derivative_certificate_subdivisions: int = 64,
) -> HilbertGradientCertificate:
    """Compress the reverse derivative graph to affine projection plus L2 error."""

    _require_numeric_dependencies()
    if derivative_certificate_subdivisions < 1:
        raise ValueError("derivative_certificate_subdivisions must be positive.")
    linears = _tanh_scalar_linears(module)
    derivative_centers: list[Any] = []
    derivative_coefficients: list[Any] = []
    derivative_remainders: list[float] = []
    moment_states = 0
    tanh_prime_lipschitz = nextafter(4.0 / (3.0 * sqrt(3.0)), inf)
    for preactivation_center, preactivation_coefficients, preactivation_remainder in zip(
        value.preactivation_centers,
        value.preactivation_coefficients,
        value.preactivation_remainders,
    ):
        radius = torch.sum(torch.abs(preactivation_coefficients), dim=1)
        approximations = [
            quadratic_tanh_prime_enclosure(
                (float(lo), float(hi)),
                certificate_subdivisions=derivative_certificate_subdivisions,
            )
            for lo, hi in zip(
                (preactivation_center - radius).detach().cpu(),
                (preactivation_center + radius).detach().cpu(),
            )
        ]
        proposal = np.asarray(
            [approximation.coeffs for approximation in approximations],
            dtype=np.float64,
        )
        uniform = torch.as_tensor(
            [approximation.delta for approximation in approximations],
            dtype=preactivation_center.dtype,
            device=preactivation_center.device,
        )
        projected_center, projected_coefficients, projection, states = (
            _project_polynomials_of_affine_forms(
                preactivation_center, preactivation_coefficients, proposal
            )
        )
        moment_states += states
        local = nextafter(
            float(torch.linalg.vector_norm(projection + uniform).detach().cpu().item()),
            inf,
        )
        total = nextafter(tanh_prime_lipschitz * preactivation_remainder + local, inf)
        derivative_centers.append(projected_center)
        derivative_coefficients.append(projected_coefficients)
        derivative_remainders.append(total)

    output = linears[-1]
    center = output.weight.detach()[0].to(
        dtype=value.center.dtype, device=value.center.device
    )
    coefficients = torch.zeros(
        (center.numel(), value.domain_coefficients.numel()),
        dtype=center.dtype,
        device=center.device,
    )
    remainder = 0.0
    product_remainders: list[float] = []
    for layer_index in range(len(derivative_centers) - 1, -1, -1):
        old_center, old_coefficients = center, coefficients
        center, coefficients, projection = _project_affine_hadamard(
            old_center,
            old_coefficients,
            derivative_centers[layer_index],
            derivative_coefficients[layer_index],
        )
        projection_remainder = nextafter(
            float(torch.linalg.vector_norm(projection).detach().cpu().item()), inf
        )
        nominal_sup = float(
            torch.max(
                torch.abs(old_center) + torch.sum(torch.abs(old_coefficients), dim=1)
            ).detach().cpu().item()
        )
        remainder = nextafter(
            remainder
            + nominal_sup * derivative_remainders[layer_index]
            + projection_remainder,
            inf,
        )
        product_remainders.append(projection_remainder)
        if layer_index:
            weight = linears[layer_index].weight.detach().to(
                dtype=center.dtype, device=center.device
            )
            center = center @ weight
            coefficients = weight.T @ coefficients
            remainder = nextafter(_spectral_norm_upper(weight) * remainder, inf)

    input_weight = linears[0].weight.detach().to(
        dtype=center.dtype, device=center.device
    )
    center = center @ input_weight
    coefficients = input_weight.T @ coefficients
    remainder = nextafter(_spectral_norm_upper(input_weight) * remainder, inf)
    return HilbertGradientCertificate(
        center=center,
        domain_coefficients=coefficients,
        remainder=remainder,
        factor_remainders=tuple(derivative_remainders),
        product_projection_remainders=tuple(reversed(product_remainders)),
        moment_states=moment_states,
    )


def _axis_aligned_radii(domain: PolynomialZonotope) -> Any:
    matrix = _affine_domain_matrix(domain)
    if matrix.shape[0] != matrix.shape[1]:
        raise NotImplementedError("Neumann witnesses currently require a full box.")
    diagonal = torch.diagonal(matrix)
    off_diagonal = matrix - torch.diag(diagonal)
    if torch.count_nonzero(off_diagonal):
        raise NotImplementedError("Neumann witnesses currently require an axis-aligned box.")
    if torch.any(diagonal == 0.0):
        raise ValueError("Neumann witnesses require positive box radii.")
    return torch.abs(diagonal)


def neumann_polynomial_witness(
    value: HilbertValueCertificate,
    domain: PolynomialZonotope,
) -> DualWitnessCertificate:
    """Certify an H1 lower bound using cubic Neumann polynomial witnesses."""

    radii = np.asarray(_axis_aligned_radii(domain).detach().cpu(), dtype=np.float64)
    affine = np.asarray(value.domain_coefficients.detach().cpu(), dtype=np.float64)
    center = float(value.center.detach().cpu().item())
    dimension = len(radii)
    # phi_i = alpha_i^3/3-alpha_i has zero physical normal derivative.
    phi_l2 = 1.0 / 63.0 - 2.0 / 15.0 + 1.0 / 3.0
    k = np.empty(dimension + 1, dtype=np.float64)
    h = np.empty_like(k)
    c = np.empty_like(k)
    k[0] = h[0] = 1.0
    c[0] = center
    k[1:] = phi_l2 + 8.0 / (15.0 * radii**2)
    beta = 1.0 + 2.0 / radii**2
    h[1:] = 1.0 / 63.0 - 2.0 * beta / 15.0 + beta**2 / 3.0
    c[1:] = affine * (1.0 / 15.0 - beta / 3.0)

    candidates: list[Any] = []
    candidates.append(c / k)
    for scale in np.logspace(-10.0, 10.0, 161):
        candidates.append(c / (k + scale * h))
    for index in range(dimension + 1):
        direction = np.zeros_like(c)
        direction[index] = 1.0
        candidates.append(direction)

    best = None
    for direction in candidates:
        witness_norm = sqrt(max(0.0, float(np.dot(k * direction, direction))))
        if witness_norm == 0.0:
            continue
        transformed = sqrt(max(0.0, float(np.dot(h * direction, direction))))
        pairing = abs(float(np.dot(c, direction)))
        penalty = value.remainder * transformed
        lower = max(0.0, pairing - penalty) / witness_norm
        if best is None or lower > best[0]:
            best = (lower, pairing, penalty, witness_norm, transformed, direction)
    assert best is not None
    lower = nextafter(max(0.0, best[0]), -inf)
    return DualWitnessCertificate(
        lower_bound=max(0.0, lower),
        nominal_pairing=best[1],
        remainder_penalty=nextafter(best[2], inf),
        witness_norm=nextafter(best[3], inf),
        transformed_witness_norm=nextafter(best[4], inf),
        coefficients=tuple(float(item) for item in best[5]),
        basis="constant_plus_coordinatewise_cubic_neumann",
    )


def _scaled_squared_interval(lower: float, upper: float, volume: float) -> Interval:
    return Interval.from_bounds(
        max(0.0, nextafter(volume * lower * lower, -inf)),
        nextafter(volume * upper * upper, inf),
    )


def certify_hybrid_graph_norms(
    module: Any,
    result: Any,
    cell: PZIntegrationCell,
    *,
    polynomial_degree: int = 5,
    residual_subdivisions: int = 2048,
    derivative_certificate_subdivisions: int = 64,
) -> GraphNormCertificate:
    """Return intersected L2/W12 certificates with positive lower mechanisms."""

    from .deep_hybrid import (
        integrate_hybrid_onejet_squared,
        integrate_hybrid_value_squared,
    )

    if not isinstance(cell.volume, (int, float)):
        raise NotImplementedError("Graph Hilbert certification requires scalar volume.")
    volume = float(cell.volume)
    value = build_hilbert_value_certificate(
        module,
        cell.domain,
        polynomial_degree=polynomial_degree,
        residual_subdivisions=residual_subdivisions,
    )
    gradient = build_hilbert_gradient_certificate(
        module,
        value,
        derivative_certificate_subdivisions=derivative_certificate_subdivisions,
    )
    previous_l2 = integrate_hybrid_value_squared(result, cell)
    previous_w12 = integrate_hybrid_onejet_squared(result, cell)
    nominal = value.nominal_norm
    reverse_lower = max(0.0, nominal - value.remainder)
    reverse_upper = nominal + value.remainder
    old_l2_lower = sqrt(max(0.0, float(previous_l2.lower) / volume))
    old_l2_upper = sqrt(max(0.0, float(previous_l2.upper) / volume))
    l2_lower = max(old_l2_lower, reverse_lower)
    l2_upper = min(old_l2_upper, reverse_upper)
    if l2_lower > l2_upper:
        raise RuntimeError("Independent sound L2 certificates have empty intersection.")

    witness = neumann_polynomial_witness(value, cell.domain)
    old_w12_lower = sqrt(max(0.0, float(previous_w12.lower) / volume))
    old_w12_upper = sqrt(max(0.0, float(previous_w12.upper) / volume))
    graph_nominal_w12 = sqrt(nominal * nominal + gradient.nominal_norm**2)
    graph_remainder_w12 = sqrt(
        value.remainder * value.remainder + gradient.remainder * gradient.remainder
    )
    graph_w12_lower = max(0.0, graph_nominal_w12 - graph_remainder_w12)
    graph_w12_upper = graph_nominal_w12 + graph_remainder_w12
    w12_lower = max(
        old_w12_lower, l2_lower, witness.lower_bound, graph_w12_lower
    )
    w12_upper = min(old_w12_upper, graph_w12_upper)
    if w12_lower > w12_upper:
        raise RuntimeError("Independent sound W12 certificates have empty intersection.")

    l2_interval = _scaled_squared_interval(l2_lower, l2_upper, volume)
    w12_interval = _scaled_squared_interval(w12_lower, w12_upper, volume)
    # Preserve monotonicity bit-for-bit as well as mathematically.  Re-scaling
    # a square root can otherwise move an endpoint by one ulp past the old
    # interval even though the real-number bounds are identical.
    l2_interval = Interval.from_bounds(
        max(float(previous_l2.lower), float(l2_interval.lower)),
        min(float(previous_l2.upper), float(l2_interval.upper)),
    )
    w12_interval = Interval.from_bounds(
        max(float(previous_w12.lower), float(w12_interval.lower)),
        min(float(previous_w12.upper), float(w12_interval.upper)),
    )
    return GraphNormCertificate(
        l2_squared=l2_interval,
        w12_squared=w12_interval,
        value=value,
        gradient=gradient,
        w12_witness=witness,
        previous_l2_squared=previous_l2,
        previous_w12_squared=previous_w12,
        normalized_diagnostics={
            "value_nominal_norm": nominal,
            "value_l2_remainder": value.remainder,
            "l2_reverse_triangle_lower": reverse_lower,
            "l2_reverse_triangle_upper": reverse_upper,
            "w12_dual_lower": witness.lower_bound,
            "gradient_nominal_norm": gradient.nominal_norm,
            "gradient_l2_remainder": gradient.remainder,
            "w12_graph_nominal_norm": graph_nominal_w12,
            "w12_graph_remainder": graph_remainder_w12,
            "w12_graph_reverse_lower": graph_w12_lower,
            "w12_graph_reverse_upper": graph_w12_upper,
            "combined_l2_lower": l2_lower,
            "combined_l2_upper": l2_upper,
            "combined_w12_lower": w12_lower,
            "combined_w12_upper": w12_upper,
        },
    )
