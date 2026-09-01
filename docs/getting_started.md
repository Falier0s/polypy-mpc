# Getting started

Every controller in RAMPC follows the same three-step lifecycle:

1. **Construct** — pass the system matrices, the stage-cost weights `Q`, `R`, the
   horizon `N`, and an `options` dictionary (constraints, disturbance set,
   reference mode, contraction factor, …).
2. **`initialize(...)`** — runs the offline design once: pre-stabilizing gain and
   terminal cost, terminal / contractive set, constraint tightening, and
   assembly of the QP.
3. **`solve(x0, ...)`** in the control loop — solves the tightened OCP for the
   current state and returns the input to apply.

## A minimal example

The script below builds a nominal {class}`~rampc.MPC` for a double integrator,
tracks a trajectory, and plots the resulting tube cross-sections.

```{literalinclude} ../examples/mpc_example.py
:language: python
:linenos:
```

Run it with:

```bash
python examples/mpc_example.py
```

## Where to go next

- {doc}`user_guide` — the model / set conventions, how the three formulations
  differ, and the adaptive mechanism.
- {doc}`auto_examples/index` — the full set of runnable examples
  ({class}`~rampc.MPC`, {class}`~rampc.RMPC`, {class}`~rampc.RAMPC`, the
  set-membership and filter tools).
- {doc}`api` — the complete class and function reference.
