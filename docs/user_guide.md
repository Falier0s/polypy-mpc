# User guide

## Models and sets

**System matrices.** A controller is constructed from a dictionary
`{"A": A, "B": B, "C": C}`. Each matrix may be given either

- as a plain 2-D array — a nominal linear time-invariant model, or
- as a 3-D array of shape `(rows, cols, n_vertices)` — a
  **parameter-varying / LPV** model that is affine in a scheduling vector, one
  slice per vertex of the parameter polytope. Slice `[:, :, 0]` is the nominal
  term.

Internally a 2-D matrix is promoted to a single-vertex 3-D array, so both cases
share one code path.

**Constraint sets.** State and input bounds are passed through `options` as
`xBound = (x_min, x_max)` and `uBound = (u_min, u_max)`. They are turned into
polytopes and combined into the joint constraint set that the tube is tightened
against.

**Uncertainty sets.** All uncertainty descriptions are
[`pycvxset.Polytope`](https://github.com/merlresearch/pycvxset) objects:

- `W` — the additive disturbance set (used by {class}`~polypympc.RMPC` and
  {class}`~polypympc.RAMPC`).
- `theta`, `theta_c` — the initial feasible parameter sets for the
  state/input and output maps (used by {class}`~polypympc.RAMPC`).

## The three formulations

### `MPC` — nominal / contractive

Solves a receding-horizon quadratic OCP for a known linear model with hard state
and input constraints. A pre-stabilizing gain and a contractive terminal set
give recursive feasibility and stability of the (tracking) error. Construction
takes `Q`, `R`, `N` and the `options` dictionary; `initialize(mode=...)` selects
how the terminal ingredients are computed (`"volume"`, `"LQR"`,
`"performance"`).

### `RMPC` — robust tube

Adds a **tube**: the predicted trajectory is confined to a sequence of polytopic
cross-sections that contain every realization consistent with the disturbance
set `W` (and, for LPV models, the parameter polytope). Constraints are tightened
so that the *whole* tube — not just the nominal prediction — satisfies the
original limits. The online problem stays a single QP of fixed size.

### `RAMPC` — robust adaptive

Adds **online identification** on top of the tube:

- a projected **Recursive Least Squares** estimator with a fixed forgetting
  factor produces the point estimate of the scheduling parameter used for the
  nominal prediction (certainty equivalence);
- a **Set-Membership** update shrinks the feasible parameter set from
  closed-loop data while a bounded-complexity outer approximation keeps the
  number of half-spaces constant;
- the parameter set is dilated along the horizon to bound the admissible
  parameter drift.

The estimator and set update live in {mod}`polypympc.tools.adaptive_tools`
(`Filter`, `SetUpdater`). Recursive feasibility and robust constraint
satisfaction are preserved throughout (formal proofs omitted here).

## Artificial-reference tracking

For tracking, PolyPY-MPC optimizes an **artificial trajectory** as an extra decision
variable, constrained to satisfy the plant dynamics and to terminate in a
reachable equilibrium. The cost has a *stabilizing* part (predicted vs.
artificial trajectory) and a *planning* part (artificial trajectory vs. desired
reference). This decoupling keeps the OCP feasible when the reference itself
violates the constraints — the controller then tracks the closest admissible
trajectory instead of losing feasibility. Set `options["ref"]` to `"traj"` for a
full time-varying reference or `"ref"` for a single moving set-point.

## Offline design helpers

- {mod}`polypympc.tools.invariance_tools` — `GainSynthesis` (pre-stabilizing gain
  `K` and terminal weight `P` via LMIs) and `InvariantSet` (robust invariant /
  λ-contractive set computation).
- {mod}`polypympc.tools.create_system` — builds the internal system representation
  from the `{"A","B","C"}` dictionary.

## TurtleBot 4 case study

The `examples/turtlebot4/` directory contains an LPV pose-tracking scenario: the
differential-drive kinematics are embedded as an LPV system with scheduling
vector `[cos θ, sin θ]`, and {class}`~polypympc.RAMPC` tracks circular, harmonic
and recorded reference trajectories under real-time constraints.

```{literalinclude} ../examples/turtlebot4/rampc_example.py
:language: python
:lines: 1-60
```

The complete script (with plotting and result export) is in the repository.
