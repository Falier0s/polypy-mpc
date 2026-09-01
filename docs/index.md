# PolyPY-MPC

**Robust and Adaptive Model Predictive Control tools** — a Python toolbox for
building MPC controllers for advanced applications that require *formal
closed-loop guarantees*: recursive feasibility, robust constraint satisfaction,
and convergence of the regulation or tracking error.

PolyPY-MPC is built on [CasADi](https://web.casadi.org/) for the symbolic
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
  | {class}`~polypympc.MPC`   | nominal / contractive MPC | known linear model, hard state and input constraints |
  | {class}`~polypympc.RMPC`  | robust **tube** MPC | bounded additive disturbances |
  | {class}`~polypympc.RAMPC` | robust **adaptive** MPC | parametric / LPV model uncertainty, estimated online (set-membership update + projected RLS with forgetting factor) |

  {class}`~polypympc.RMPC` extends {class}`~polypympc.MPC` and {class}`~polypympc.RAMPC`
  extends {class}`~polypympc.RMPC`, so the constraint, tube and terminal machinery
  is shared.
- **Polytopic representation** of the constraint sets *and* of the disturbance
  and parameter-uncertainty sets. Constraint tightening, tube cross-sections and
  the feasible-parameter-set update are all polytope operations.
- **Fixed online complexity** — the size of the QP solved at each step does not
  grow as the uncertainty description is updated online.
- **Coming soon:** distributed / interconnected-system derivations of the robust
  and robust-adaptive schemes.

## Reference application

PolyPY-MPC demonstrates the {class}`~polypympc.RAMPC` scheme on an LPV model of a
TurtleBot 4 differential-drive robot; see the {doc}`user_guide` case study.

% Companion-paper reference — restore on acceptance (kept out of the published page):
% PolyPY-MPC is the companion software of a research paper currently under peer
% review (target journal: Control Engineering Practice). See the citing page.

As an example of robotic deployment, a **ROS 2** handler package runs the
controller as ROS 2 nodes: <https://github.com/Arthasee/cRAMPC>.

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
