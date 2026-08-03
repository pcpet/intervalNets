# Codex repository instructions

## Repository state

`intervalNets` contains two certified enclosure pipelines:

- interval and derivative enclosures with adaptive norm integration;
- polynomial-zonotope (PZ) propagation of neural-network values, Jacobians, and Hessians.

The PZ core, `PZTwoJet`, affine and activation propagation, `model.eval_pz_twojet(...)`, PZ integration, and PZ norm routines already exist. A separate `model.eval_pz_value(...)` path propagates only function values and is the default for PZ \(L^2\) computation; do not reintroduce Jacobian or Hessian construction into that path. The scalable `model.eval_pz_onejet(...)` path is the default for PZ \(W^{1,2}\): it propagates a dependent Jacobian polynomial core plus a certified remainder for explicitly reduced terms, without constructing Hessians. Do not route order-one norms through `eval_pz_twojet(...)`. Do not treat the original two-jet blueprint as an unimplemented feature checklist.

## Read the relevant specification first

Inspect the existing implementation and tests before editing it. Use the document matching the task:

- `docs/polynomial_zonotope_notation_and_terminology.tex`: canonical notation and vocabulary for domain noise, approximation noise, sparse polynomial supports, network layers, and two-jets; consult this before introducing new mathematical notation or terminology;
- `docs/polynomial_zonotope_twojet_recurrence.tex`: ground-truth initialization and affine/activation recurrences for direct propagation of value, Jacobian, and Hessian enclosures through an input polynomial zonotope;
- `docs/blueprints/pz_twojet_blueprint.tex`: mathematical design and historical implementation blueprint for PZ two-jets;
- `docs/affine_tanh_enclosures.tex`: certified affine enclosures for `tanh`, `tanh'`, and `tanh''`;
- `docs/certified_polynomial_zonotope_integration.tex`: geometric PZ integration and approximation-noise semantics;
- `docs/direct_integrated_twojet_squares.tex`: direct certified integration of squared PZ two-jets without constructing the squared integrand.
- `docs/diagnostics_and_metrics_glossary.tex`: ground-truth metric definitions, aggregation rules, canonical CSV/JSON schemas, and mandatory mini- and medium-benchmark outputs; consult this before adding or changing benchmark diagnostics or output columns.

The current source code and tests define the implemented public behavior. When a design document and the implementation differ, identify the discrepancy explicitly instead of silently changing semantics.

## Important implementation invariants

- Preserve rigorous enclosure guarantees and outward-rounding behavior.
- Preserve shared polynomial dependencies; do not silently replace them by intervals unless the relevant specification explicitly permits re-enclosure.
- Keep domain noise symbols \(\alpha\) distinct from approximation noise symbols \(\eta\), and use \(\varepsilon=(\alpha,\eta)\) for their combined vector.
- Use **approximation noise symbol** as the canonical term. Reserve **residual function** for the actual difference between a function and its approximation, and **approximation-error radius** for its certified coefficient \(\rho\).
- Under pointwise integration semantics, an approximation noise symbol may represent the residual function separately at each physical point; it is not one global symbolic value over the integration domain. Follow the semantics in `pz_integration.py`.
- Canonicalize equal exponent vectors and combine their coefficients before applying absolute values or interval collapse. This is required to preserve cancellations and reproduce the existing enclosure.
- Maintain tensor-valued coefficient support and the established shapes of `Y`, `J`, and `H`.
- Exploit Hessian symmetry only where the stored Hessian convention guarantees it. For a full symmetric Hessian, off-diagonal Frobenius contributions have weight two.
- Avoid changing public APIs or numerical semantics unless the task explicitly requires it.

## PZ norm and direct-integration work

For changes to certified PZ `L^2`, `W^{1,2}`, or `W^{2,2}` integration, read `docs/direct_integrated_twojet_squares.tex` in full and inspect:

- `src/intervalnets/polynomial_zonotope.py`;
- `src/intervalnets/pz_integration.py`;
- `src/intervalnets/pz_norms.py`;
- `tests/test_polynomial_zonotope.py`;
- `tests/test_pz_integration.py`;
- `tests/test_pz_norms.py`.

The direct-integration optimization must reproduce the current certified enclosure while avoiding materialization of the squared PZ integrand. Exploit unordered monomial-pair symmetry and Hessian symmetry, but still merge all contributions with the same retained pointwise-noise exponent before taking absolute values. Keep the existing explicit-square path available at least internally for regression comparisons until equivalence is well tested.

Benchmark enclosure construction, norm-integrand construction, and integration separately. Final monomial count alone is not an adequate performance measure because sparse polynomial multiplication processes intermediate term pairs before canonicalization.

All benchmark notebooks and reusable diagnostics must follow
`docs/diagnostics_and_metrics_glossary.tex`. Preserve its schema version,
exact canonical column orders, missing-value/status conventions, deterministic
row ordering, and distinction between absolute widths, local relative radii,
familywise global normalized radii, and relative norm widths. A medium
benchmark includes every mini-benchmark output plus the per-neuron activation
approximation table and `layer_normalized_radius_Y.csv`; do not silently omit
unimplemented required quantities.

For schema version 1.2 and later, preserve both local relative radius and local
relative width in the prescribed columns, and populate the physical domain
volume plus the domain-volume-normalized lower endpoint, upper endpoint, and
width for every implemented squared and unsquared norm row. Domain-volume
normalization is a scale metric; relative norm width is the tightness metric.

For value-only \(L^2\) performance work, use
`notebooks/pz_l2_value_benchmarks.ipynb`. The affine tanh enclosure keeps the
value support degree one, with one domain symbol per input coordinate and one approximation noise symbol per hidden neuron. Preserve this
independence when batching activation enclosures.

For one-jet \(W^{1,2}\) performance work, use
`notebooks/pz_w12_polynomial_reduction_benchmarks.ipynb`. The scalable one-jet
retains a dependent Jacobian polynomial core and propagates a separate,
certified pointwise box only for terms explicitly removed by a reduction
policy. Do not replace the whole Jacobian by intervals. Available experimental
policies are top-k generator retention, degree-capped top-k retention, and a
sound coefficient-space PCA reduction with an explicitly bounded projection
remainder. Treat runtime, retained support, polynomial degree, reduction
remainder, and final enclosure width as joint diagnostics.

## Development workflow

1. Inspect the relevant source, tests, and specification.
2. Make the smallest coherent change.
3. Add focused regression tests, including cancellation and noise-kind edge cases.
4. Run targeted tests first, then the full suite:

   ```bash
   pytest -q tests/test_polynomial_zonotope.py tests/test_pz_integration.py tests/test_pz_norms.py
   pytest -q
   ```

5. For performance work, report both correctness comparisons and timings on the same input.
6. Keep experimental notebook code thin; reusable logic belongs in `src/intervalnets/` and assertions belong in `tests/`.
