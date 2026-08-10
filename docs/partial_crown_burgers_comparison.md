# partial-CROWN Burgers comparison protocol

This benchmark compares partial-CROWN with intervalNets on the pointwise
Burgers residual

\[
f_\theta(t,x)=u_t+u_\theta u_x-\frac{0.01}{\pi}u_{xx},
\qquad (t,x)\in[0,1]\times[-1,1].
\]

It does not compare partial-CROWN's pointwise residual certificate with the
global function-space norm certificates elsewhere in this repository.

## Reproducibility boundary

Eiras et al. report an eight-hidden-layer, width-20 tanh PINN, but their public
repository does not include the ONNX checkpoint or split history used for
Table 1. Consequently, this benchmark has two deliberately separate parts:

1. a transcription of the published Burgers reference values and hardware;
2. a same-network rerun on a deterministically trained, architecture-matched
   replacement PINN.

The second part is the only valid head-to-head method comparison. The verifier
source is pinned to partial-CROWN commit
`161377048ee92b4c09faf5ce41628168c7e01556`.

## Matched comparison rules

Both methods receive the same folded physical-input network, domain, residual,
sampled extrema used only for branch prioritization, four-way cell split, and
cell-evaluation budgets. Each method uses its own bounds to select the next
cell under the sampled-gap score used by the paper's released branching code.

The primary certified tightness quantity is

\[
U_{|f|^2}=\max\{|L_f|,|U_f|\}^2.
\]

The sampled maximum is not part of the proof. It is used to report

\[
\frac{U_{|f|}}{\widehat{\max |f|}}
\quad\text{and}\quad
U_{|f|}-\widehat{\max |f|}.
\]

Timing includes verifier calls only. It excludes imports, training, diagnostic
sampling, plotting, and serialization. Cross-hardware comparisons with the
paper's M1 Max timing are descriptive, not speedup claims.

## Workflow profiles

The manually dispatched GitHub Actions workflow offers:

- `quick`: pipeline validation with reduced training and budgets;
- `standard`: 3,000 Adam steps, 500 L-BFGS iterations, and budgets through 257
  cell evaluations.

The paper's two-million-branch residual experiment took about 280,000 seconds
on its reported hardware and cannot fit within the six-hour GitHub-hosted
runner limit. The workflow accepts custom budgets, but full-paper budgets
require a self-hosted or resumable runner.
