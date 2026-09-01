# Installation

PolyPY-MPC is **not on PyPI**: it depends on
[`pycvxset`](https://github.com/merlresearch/pycvxset), which is distributed only
from source. Install everything from source.

## Requirements

- Python **≥ 3.9** and `git`.
- A C toolchain for `pycddlib` (pulled in by `pycvxset`):
  - **Windows** — nothing extra, `pip` handles it. If plotting raises a backend
    error, set `MPLBACKEND=Agg`.
  - **Ubuntu / Debian** — `sudo apt-get install libgmp-dev libcdd-dev python3-dev`
  - **macOS** — `brew install gmp cddlib`
- Optional: **MOSEK** (needs a licence) for some LMI / SOCP steps. Without it the
  examples fall back to CLARABEL / OSQP / SCS.

See the
[`pycvxset` installation notes](https://github.com/merlresearch/pycvxset#installation)
for the authoritative, OS-by-OS steps.

## Install

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux / macOS:      source .venv/bin/activate
python -m pip install --upgrade pip

git clone https://github.com/Falier0s/polypy-mpc.git
cd polypy-mpc
pip install -e ".[viz,notebook]"
```

`pip` resolves `pycvxset` straight from its GitHub repository. `[viz]` adds
matplotlib / pandas (used by the examples); `[notebook]` adds `ipykernel`.

If the automatic `pycvxset` download fails, install it yourself first and then
install PolyPY-MPC with `--no-deps`; the project README documents this guided path.

## Check the installation

```bash
python -c "import polypympc; print(polypympc.__all__, polypympc.__version__)"
python examples/rmpc_example.py
python examples/rampc_example.py
```
