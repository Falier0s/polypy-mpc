# RAMPC

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Built with CasADi](https://img.shields.io/badge/built%20with-CasADi-1f6feb.svg)](https://web.casadi.org/)
[![Paper](https://img.shields.io/badge/paper-under%20review-orange.svg)](#how-to-cite)
[![ROS 2 companion](https://img.shields.io/badge/ROS%202-companion%20package-22314e.svg)](https://github.com/Arthasee/cRAMPC)

**RAMPC** is a Python toolbox for building **Model Predictive Control (MPC)**
objects for advanced applications that require *formal closed-loop guarantees* —
recursive feasibility, robust constraint satisfaction, and convergence of the
regulation or tracking error.

It is built on [CasADi](https://web.casadi.org/) for the symbolic optimal-control
problem and on convex optimization (via [`cvxpy`](https://www.cvxpy.org/) and
[`pycvxset`](https://github.com/merlresearch/pycvxset)) for the offline design
ingredients (pre-stabilizing gain, terminal cost, contractive / invariant sets).

## What it offers

- **Two control tasks, one API** — set-point *regulation* and time-varying
  *trajectory tracking*. Tracking is handled through an artificial-reference
  formulation that keeps the problem feasible even when the reference is
  temporarily or permanently inadmissible.
- **Three formulations, increasing in capability:**

  | class    | formulation | additionally handles |
  |----------|-------------|----------------------|
  | `MPC`   | nominal / contractive MPC | known linear model, hard state and input constraints |
  | `RMPC`  | robust **tube** MPC | bounded additive disturbances |
  | `RAMPC` | robust **adaptive** MPC | parametric / LPV model uncertainty, estimated online (set-membership update + projected RLS with forgetting factor) |

  `RMPC` extends `MPC` and `RAMPC` extends `RMPC`, so the constraint,
  tube and terminal machinery is shared.
- **Polytopic representation** of the constraint sets *and* of the disturbance
  and parameter-uncertainty sets. Constraint tightening, tube cross-sections and
  the feasible-parameter-set update are all carried out as polytope operations.
- **Fixed online complexity** — the size of the QP solved at each step does not
  grow as the uncertainty description is updated online.
- **Coming soon:** distributed / interconnected-system derivations of the robust
  and robust-adaptive schemes.

## Reference application

The companion paper validates the `RAMPC` scheme on a **Linear Parameter
Varying (LPV)** model: pose tracking of a wheeled differential-drive robot
(TurtleBot 4), where the nonholonomic kinematics are embedded as an LPV system
and the scheduling parameter is estimated online while preserving the robust
guarantees. The full case study, including the offline design matrices and the
reference trajectories, is in [`examples/turtlebot4/`](examples/turtlebot4/); see
[How to cite](#how-to-cite) for the paper.

For deployment on real robots, a **ROS 2** package wraps this library (reference
generation, state observer and controller nodes, launch files):
<https://github.com/Arthasee/cRAMPC> *(repository link to be finalized on
release).*

## Controllers at a glance

```python
import rampc
print(rampc.__all__)   # ['MPC', 'RMPC', 'RAMPC']
```

| class   | description                                                                    |
|---------|-------------------------------------------------------------------------------|
| `MPC`   | nominal / contractive MPC                                                      |
| `RMPC`  | robust tube MPC                                                                |
| `RAMPC` | robust **adaptive** MPC (set-membership parameter estimation + LPV prediction) |

> **Not on PyPI.** `rampc` depends on
> [`pycvxset`](https://github.com/merlresearch/pycvxset), which is distributed
> only from source. Install everything from source as described below.

## Requirements

- Python **≥ 3.9** and `git`.
- A C toolchain for `pycddlib` (pulled in by `pycvxset`):
  - **Windows** — nothing extra, `pip` handles it. If plotting raises a backend
    error, set `MPLBACKEND=Agg`.
  - **Ubuntu / Debian** — `sudo apt-get install libgmp-dev libcdd-dev python3-dev`
  - **macOS** — `brew install gmp cddlib`
- Optional: **MOSEK** (needs a licence) for some LMI / SOCP steps. Without it the
  examples fall back to CLARABEL / OSQP / SCS.

See the [`pycvxset` installation notes](https://github.com/merlresearch/pycvxset#installation)
for the authoritative, OS-by-OS steps.

## Installation

Both options use a dedicated virtual environment. Pick the activation line for
your shell:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows cmd.exe:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
```

### Option A — quick (let `pip` fetch `pycvxset`)

```bash
git clone https://github.com/Falier0s/RAMPC.git
cd RAMPC
pip install -e ".[viz,notebook]"
```

`pip` resolves `pycvxset` straight from its GitHub repository (it is declared as a
dependency). `[viz]` adds matplotlib / pandas, required by the examples;
`[notebook]` adds `ipykernel` for the Jupyter notebooks.

### Option B — guided (install `pycvxset` yourself first)

Use this if Option A's automatic download fails, or if you want an **editable**
`pycvxset` checkout to inspect / modify.

```bash
# 1. pycvxset from source, following MERL's instructions
git clone https://github.com/merlresearch/pycvxset.git
pip install -e ./pycvxset
python pycvxset/examples/pycvxset_diag.py     # sanity check: opens a 2-subplot figure

# 2. RAMPC, without re-resolving dependencies (pycvxset is already in the venv)
git clone https://github.com/Falier0s/RAMPC.git
pip install -e ./RAMPC --no-deps
pip install numpy scipy casadi cvxpy scs osqp pycddlib matplotlib pandas ipykernel
```

`--no-deps` is what keeps `pip` from pulling a second, non-editable copy of
`pycvxset` from GitHub over the one you just installed.

## Check the installation

```bash
python -c "import rampc; print(rampc.__all__)"          # ['MPC', 'RMPC', 'RAMPC']
python examples/mpc_example.py
python examples/turtlebot4/rampc_example.py Circular --steps 300
```

Every script in `examples/` also runs directly from a plain checkout (no install)
thanks to a `src/` import fallback, as long as the dependencies are on the path.

## Repository layout

```
src/            rampc package: MPC, RMPC, RAMPC and tools/
examples/       runnable examples; examples/turtlebot4/ is the paper case study
doc/            conversion notes (MATLAB yMPC -> Python)
pyproject.toml  packaging metadata and dependencies
```

## How to cite

This library is the companion software of a research paper **currently under peer
review** (target journal: *Control Engineering Practice*, Elsevier). If you use
RAMPC in academic work, please cite the paper once it is published. In the
meantime, the following BibTeX entry can be used as a placeholder — the
bibliographic fields (`volume`, `pages`, `doi`, final `year`) will be completed on
acceptance:

```bibtex
@article{faliero_rampc_underreview,
  title   = {Design and validation of Robust Adaptive {MPC} for {LPV}
             Trajectory tracking of Wheeled Differential Drive Robot with
             guarantees},
  author  = {Faliero, Fabio and Braun-Delvoye, Baptiste and
             Iannelli, Andrea and Capello, Elisa},
  journal = {Control Engineering Practice},
  year    = {2026},
  note    = {Under review. Bibliographic details to be updated upon acceptance.},
  publisher = {Elsevier}
}
```

To cite the software implementation itself:

```bibtex
@software{rampc_software,
  title   = {{RAMPC}: Robust and Adaptive Model Predictive Control tools},
  author  = {Faliero, Fabio and Braun-Delvoye, Baptiste},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Falier0s/RAMPC}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE). Note that `rampc`
requires `pycvxset`, released under **AGPL-3.0-or-later**; keep this in mind when
redistributing a combined work.
