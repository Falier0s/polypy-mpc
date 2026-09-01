# RAMPC

**Robust and Adaptive Model Predictive Control tools** — a Python toolbox for
building MPC controllers for advanced applications that require *formal
closed-loop guarantees*: recursive feasibility, robust constraint satisfaction,
and convergence of the regulation or tracking error.

RAMPC is built on [CasADi](https://web.casadi.org/) for the symbolic
optimal-control problem and on convex optimization (via
[`cvxpy`](https://www.cvxpy.org/) and
[`pycvxset`](https://github.com/merlresearch/pycvxset)) for the offline design
ingredients — the pre-stabilizing gain, terminal cost, and contractive /
invariant sets.

## What it offers

- **Two control tasks, one API** — set-point *regulation* and time-varying
  *trajectory tracking*. Tracking uses an artificial-reference formulation that
  keeps the problem feasible even when the reference is temporarily or
  permanently inadmissible.
- **Three formulations, increasing in capability:**

  | class    | formulation | additionally handles |
  |----------|-------------|----------------------|
  | {class}`~rampc.MPC`   | nominal / contractive MPC | known linear model, hard state and input constraints |
  | {class}`~rampc.RMPC`  | robust **tube** MPC | bounded additive disturbances |
  | {class}`~rampc.RAMPC` | robust **adaptive** MPC | parametric / LPV model uncertainty, estimated online (set-membership update + projected RLS with forgetting factor) |

  {class}`~rampc.RMPC` extends {class}`~rampc.MPC` and {class}`~rampc.RAMPC`
  extends {class}`~rampc.RMPC`, so the constraint, tube and terminal machinery
  is shared.
- **Polytopic representation** of the constraint sets *and* of the disturbance
  and parameter-uncertainty sets. Constraint tightening, tube cross-sections and
  the feasible-parameter-set update are all polytope operations.
- **Fixed online complexity** — the size of the QP solved at each step does not
  grow as the uncertainty description is updated online.
- **Coming soon:** distributed / interconnected-system derivations of the robust
  and robust-adaptive schemes.

## Companion paper

RAMPC is the companion software of the paper *"Design and validation of Robust
Adaptive MPC for LPV Trajectory tracking of Wheeled Differential Drive Robot with
guarantees"* (under review, *Control Engineering Practice*). The paper validates
the {class}`~rampc.RAMPC` scheme on an LPV model of a TurtleBot 4 differential
drive robot. See {doc}`citing` and the {doc}`user_guide` case study.

A **ROS 2** integration package for real-robot deployment lives at
<https://github.com/Arthasee/cRAMPC> *(link to be finalized on release).*

```{toctree}
:maxdepth: 2
:hidden:

installation
getting_started
user_guide
auto_examples/index
api
citing
changelog
```
