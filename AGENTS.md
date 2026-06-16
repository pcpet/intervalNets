# Codex instructions

For the polynomial-zonotope two-jet enclosure task, read:

docs/blueprints/pz_twojet_blueprint.tex

Treat this LaTeX document as the mathematical and implementation specification.

Implement the feature incrementally:
1. inspect the existing interval propagation and Jacobian evaluation architecture;
2. add the core `PolynomialZonotope` and `PZTwoJet` classes;
3. add affine layer propagation;
4. add tanh approximation and certified residual error scaffolding;
5. add tanh two-jet propagation;
6. add `model.eval_pz_twojet(...)`;
7. add tests for arithmetic, shape correctness, residual certification, and comparison against PyTorch autograd samples.

Prefer small, tested changes. Do not silently replace polynomial dependencies by intervals unless the blueprint explicitly allows a reduction/re-enclosure step.
