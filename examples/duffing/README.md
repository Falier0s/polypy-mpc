# Duffing oscillator case study

This folder is a worked example of the three controllers documented in the
[user guide](../../docs/user_guide.md#the-three-formulations) —
{class}`~polypympc.MPC`, {class}`~polypympc.RMPC`, {class}`~polypympc.RAMPC` —
applied to the same nonlinear, parametrically uncertain plant, so the three
formulations can be compared directly rather than only described in the
abstract.

## The plant and where the uncertainty comes from

The true system is the (nonlinear) Duffing oscillator

```
x1' = x2
x2' = -delta * x2 - alpha * x1 - beta * x1**3 + gamma * u
```

`delta`, `alpha`, `beta` (damping, linear stiffness, cubic stiffness) are not
assumed known exactly. An LPV model is identified offline from data
(`ss_duffing.npz`: `A`, `B` tensors of shape `(2, 2, 4)` / `(2, 1, 4)`, plus
`C`, `D`, `Ts`) together with an SVD-based reduction that extracts the
*uncertain component* of that identification along a small number of
principal directions. The resulting reduced scheduling parameter `theta`
indexes those directions, and its admissible range is the polytope `Theta`
(`uncertainty_duffing.npz`, offered as three descriptions — `H_poly`, the
tightest halfspace description; `H_rect`, an axis-aligned box; `H_hull`, the
convex hull of the sampled parameter points). `theta = 0` corresponds to the
nominal identified model `(A[:, :, 0], B[:, :, 0])`, not to the centre of
`Theta` — see `duffing_common.load_validation_point` for the Chebyshev centre
used instead where a "central" model is needed. Two held-out validation cases
("edge", "inner" in `validation_points_duffing.npz`) provide real
`(delta, alpha, beta)` triplets, never used to fit the model or the SVD basis,
against which the closed loop is actually simulated.

## The three scripts

- **`mpc_example.py`** — nominal {class}`~polypympc.MPC` on
  `(A[:, :, 0], B[:, :, 0])` alone, with no knowledge of `Theta`.
- **`rmpc_example.py`** — {class}`~polypympc.RMPC`: a tube built from `Theta`
  and the additive disturbance set `W`, with constraints tightened so the
  *whole* tube — not just the nominal prediction — stays admissible for every
  parameter the true plant could realize. The pre-stabilizing gain `K` and
  terminal weight `P` are not the ones `RMPC.initialize()` would synthesize
  internally (that LMI treats the LPV basis slices as independent vertex
  systems, which they are not); instead `synthesize_robust_gain.py` certifies
  `K`/`P` against `(A(theta_v), B(theta_v))` evaluated at the true vertices of
  `Theta` and saves them to `robust_gain.npz`, which this script loads.
- **`rampc_example.py`** — {class}`~polypympc.RAMPC`: the same tube and the
  same offline `K`/`P`, plus online Recursive-Least-Squares estimation and a
  Set-Membership update that shrinks the feasible parameter set from the
  closed-loop `(x, y)` data as the simulation runs.

`duffing_common.py` factors out what would otherwise be copy-pasted across the
three (data loading, the RK4 plant, the reference signal, CSV I/O); each
script otherwise builds its own options and stays runnable on its own.
`run_comparison.py` runs (or loads cached CSVs for) all three back to back and
overlays their closed-loop trajectories; `synthesize_robust_gain.py` is the
offline gain-synthesis step described above and only needs to be re-run after
changing `Theta`, `Q`, `R`, `lam`, or the constraint bounds.

## What the comparison shows

- **MPC** has no margin against the identification error: since it is
  designed for the nominal model alone, the real (uncertain) plant can carry
  it outside the constraints it was solved for.
- **RMPC** is the conservative-but-safe counterpart: designing against the
  *whole* of `Theta` guarantees robust constraint satisfaction, at the cost of
  a tighter admissible tube and correspondingly more cautious closed-loop
  behaviour (visible as slower tracking / a larger performance gap vs. MPC in
  the comparison plots).
- **RAMPC** starts exactly as conservative as RMPC (same tube geometry, same
  offline `K`/`P`) but is not stuck there: as closed-loop data rules out
  parameter values inconsistent with what was observed, the online
  Set-Membership update shrinks the feasible set around the estimate, so the
  tube — and the conservatism it imposes — recursively relaxes over time,
  while the same robust guarantees are preserved throughout (see the
  `theta_hat` vs. true-`theta` convergence plot drawn by
  `rampc_example.py` when run standalone).
