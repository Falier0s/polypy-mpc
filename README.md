# RAMPC

Robust and Adaptive Model Predictive Control tools, built on
[CasADi](https://web.casadi.org/) and convex optimization. Companion code for the
paper *(title TBD)*.

The `rampc` package provides three controllers:

| class    | description                                                              |
|----------|--------------------------------------------------------------------------|
| `CMPC`   | nominal / contractive MPC                                               |
| `CRMPC`  | robust tube MPC                                                         |
| `CRAMPC` | robust **adaptive** MPC (set-membership parameter estimation + LPV prediction) |

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
python -c "import rampc; print(rampc.__all__)"          # ['CMPC', 'CRMPC', 'CRAMPC']
python examples/mpc_example.py
python examples/turtlebot4/rampc_example.py Circular --steps 300
```

Every script in `examples/` also runs directly from a plain checkout (no install)
thanks to a `src/` import fallback, as long as the dependencies are on the path.

## Repository layout

```
src/            rampc package: CMPC, CRMPC, CRAMPC and tools/
examples/       runnable examples; examples/turtlebot4/ is the paper case study
doc/            conversion notes (MATLAB yMPC -> Python)
pyproject.toml  packaging metadata and dependencies
```

## License

See [LICENSE](LICENSE). Note that `rampc` requires `pycvxset`, released under
**AGPL-3.0-or-later**; keep this in mind when redistributing.
