from pathlib import Path

import numpy as np

from pycvxset import Polytope

NX, NU, NY = 4, 2, 2   # x = [x1, v1, x2, v2], u = [u1, u2], y = [x1, x2]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_ss(script_dir: Path):
    """Return (A, B, C, D, Ts) from ss_data.npz -- LTI model.

    A, B are (2, 2, 4) / (2, 1, 4) LPV tensors, C is (1, 2), D is (1, 1).
    """
    data = np.load(script_dir)
    return data["A"], data["B"], data["C"], data["D"], float(data["Ts"])


_THETA_KEYS = {"polytope": "H_poly", "box": "H_rect", "hull": "H_hull"}


def load_uncertainty_set(script_dir: Path, set_type: str = "polytope") -> Polytope:
    """Return Theta = Polytope(A=H, b=ones) from uncertainty_sets.npz.

    ``set_type`` selects which halfspace description of the uncertainty set
    to use: "polytope" (H_poly, the tightest description), "box" (H_rect, an
    axis-aligned outer approximation) or "hull" (H_hull, the convex hull of
    the sampled parameter points).
    """
    try:
        key = _THETA_KEYS[set_type]
    except KeyError as exc:
        raise ValueError(
            f"Unknown set_type {set_type!r}; expected one of {sorted(_THETA_KEYS)}"
        ) from exc
    H = np.load(script_dir)[key]
    return Polytope(A=H, b=np.ones(H.shape[0],))


def load_mismatch_set(script_dir: Path) -> Polytope:
    """Return W = Polytope(A=W_A, b=W_b) from uncertainty_sets.npz: the
    truncation-mismatch additive disturbance on x(k+1), in STATE space
    (dimension W_dim), not parameter space -- see the "Model-mismatch
    disturbance bound" cell in modeling-uncertainty.ipynb. General (H, h)
    form, not unit-normalized: W does not need to be scaled to contain the
    origin at its boundary, unlike the theta sets in load_uncertainty_set.
    """
    data = np.load(script_dir)
    return Polytope(A=data["W_A"], b=data["W_b"])


def load_point(script_dir: Path):
    """Return (p_real, p_basis, p_center) for ``point in {"edge", "inner"}``.

    Each is a held-out validation dataset (never used to fit the nominal
    model or the SVD parameter basis). p_real and p_basis describe the same
    physical test case but live in unrelated spaces, with no known map
    between them:

    p_real: that case's true physical coefficients, packed as
        [m1, m2, k, c] (the same as the training points in sys_param.npz).
    p_basis: the same case's coordinate in the controller's reduced theta
        space (the SVD basis the affine A(theta)/B(theta) model is built
        from) 
    p_center: the Chebyshev centre of the uncertainty ellipsoid
        (p_cheb, coincides with p_c in uncertainty_sets.npz) --
        the actual geometric centre of the identified uncertainty region
        (shared, does not depend on ``point``). NOT the same as theta = 0
        (<-> (A0, B0), the affine parametrization's origin, which need not be
        anywhere near the centre of the identified set).
    """
    data = np.load(script_dir, allow_pickle=True)
    
    return (
        data["p_real"],
        data["p_basis"],
        data["p_cheb"],
        data["n_p"]
    )


def lift_output_matrix(C: np.ndarray, n_slices: int) -> np.ndarray:
    """Stack C into a 3-axis tensor with ``n_slices`` copies (only the first
    non-zero), matching how the general rmpc/rampc examples build an output
    map with q_c = 0."""
    return np.stack([C] + [np.zeros_like(C)] * (n_slices - 1), axis=2)


def make_disturbance(W: Polytope, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a disturbance vector w uniformly distributed within a polytopic set W.

    Rejection sampling: draw uniformly in the bounding box of W and keep the
    first point that lies in W.

    For reproducible runs create the generator ONCE, before the simulation
    loop, and pass the same object at every call::

        rng = np.random.default_rng(SEED)
        for t in range(T):
            w = make_disturbance(W, rng)

    Seeding inside the function would restart the stream at each call and
    return the same w at every step.
    """
    rng = np.random.default_rng() if rng is None else rng
    lb, ub = W.minimum_volume_circumscribing_rectangle()
    while True:
        candidates = rng.uniform(lb, ub, size=(64, lb.size))
        inside = W.contains(candidates)
        if inside.any():
            return candidates[inside][0]


# ---------------------------------------------------------------------------
# Shared legend for stacked sharex subplots
# ---------------------------------------------------------------------------
def shared_legend(fig, axes, loc="outside upper center", ncol=None, **kwargs):
    """One fig-level legend replacing a repeated per-subplot one.

    Collects get_legend_handles_labels() from every axis in ``axes``,
    keeping only the FIRST handle for each distinct label (so a color/label
    reused across subplots -- e.g. the same "x bound" or "theta bounds" line
    style appearing once per state/parameter -- appears once, not once per
    subplot). ``loc="outside upper center"`` (needs a "constrained" layout
    figure) reserves its own space above the axes instead of overlapping
    them or being clipped at the figure edge.
    """
    seen = {}
    for ax in np.atleast_1d(axes):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
    if ncol is None:
        ncol = len(seen)
    fig.legend(list(seen.values()), list(seen.keys()), loc=loc, ncol=ncol, frameon=False, **kwargs)


# ---------------------------------------------------------------------------
# True nonlinear mass-spring-damper plant
# ---------------------------------------------------------------------------
def make_msd_plant(m1, m2, k, c, ts):
    """Return (rhs, step) for the true two-mass mass-spring-damper plant.

    x = [x1, v1, x2, v2], u = [u1, u2] -- linear, matching cont_ss/disc_ss in
    modeling-uncertainty.ipynb exactly. The plant IS its own Euler
    discretization by design (see that notebook): there is no separate
    "more exact" continuous-time integrator to fall back on, so ``step``
    takes a single Euler step -- not RK4 -- and is the actual plant update.
    ``rhs`` (the continuous-time dynamics) is exposed only for reference.
    """
    Ac = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-k / m1, -c / m1, k / m1, c / m1],
        [0.0, 0.0, 0.0, 1.0],
        [k / m2, c / m2, -k / m2, -c / m2],
    ])
    Bc = np.array([
        [0.0, 0.0],
        [1.0 / m1, 0.0],
        [0.0, 0.0],
        [0.0, 1.0 / m2],
    ])
    Ad, Bd = np.eye(4) + ts * Ac, ts * Bc

    def rhs(t, x, u):
        """Return dx/dt = Ac @ x + Bc @ u (reference only, not used by step)."""
        return Ac @ x + Bc @ u

    def step(x0, u0):
        """Return the next state after one sample period ts (one Euler step)."""
        return Ad @ x0 + Bd @ u0

    return rhs, step




# ---------------------------------------------------------------------------
# Reference signal + tracking-mode plumbing ("trajectory" vs "setpoint")
# ---------------------------------------------------------------------------
def make_reference(ts, kind="square", amp=1.0, n_signals=1, period=5.0, phase=0.0):
    """Return reference(k): the signal(s) tracked on y = C @ x ("square" wave
    or a "harmonic" multi-tone), evaluated at simulation step k as an
    ``(n_signals,)`` array.

    ``amp``/``period``/``phase`` are broadcast to ``n_signals`` when given as
    a scalar, or must each have exactly ``n_signals`` entries when given as a
    sequence -- a different ``period``/``phase`` per signal is what makes two
    outputs (e.g. the two masses' positions) track genuinely independent
    references instead of moving in lockstep.
    """

    def _bcast(v, name):
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.size == 1:
            return np.full(n_signals, arr[0])
        if arr.size != n_signals:
            raise ValueError(f"{name} has {arr.size} entries, expected 1 or n_signals={n_signals}")
        return arr

    amp_v = _bcast(amp, "amp")
    period_v = _bcast(period, "period")
    phase_v = _bcast(phase, "phase")

    def reference(k):
        t = k * ts
        if kind == "square":
            return amp_v * np.sign(np.sin(2.0 * np.pi * t / period_v + phase_v) + 1e-12)
        tt = t + phase_v
        return amp_v * (
             0.18 * np.sin(2.0 * np.pi * 0.15 * tt)
            + 0.08 * np.sin(2.0 * np.pi * 0.37 * tt + 0.7)
            + 0.05 * np.cos(2.0 * np.pi * 0.61 * tt)
        )

    return reference


def ref_option(tracking_mode: str) -> str:
    """Map TRACKING_MODE ("trajectory" | "setpoint") to the controller's
    options["ref"] value ("traj" | "ref")."""
    try:
        return {"trajectory": "traj", "setpoint": "ref"}[tracking_mode]
    except KeyError as exc:
        raise ValueError(
            f"Unknown TRACKING_MODE {tracking_mode!r}; expected 'trajectory' or 'setpoint'"
        ) from exc


def reference_arg(reference, t, n_horizon, tracking_mode: str):
    """Build the ``r=`` argument passed to solve() for step ``t``.

    "trajectory": the reference window [reference(t), ..., reference(t + N)]
    covering the controller's prediction horizon, under the assumption that
    the whole reference signal is precomputed and available upfront (matches
    options["ref"] = "traj").
    "setpoint": just [reference(t)] (matches options["ref"] = "ref").

    Flattened into a single flat list: solve() builds ca.vertcat(x0, r), which
    accepts a flat list/array but raises on a list of per-step reference
    arrays (e.g. one (n_signals,) array per step, since make_reference can now
    return more than one signal) -- so every per-step entry is flattened here,
    not left for the caller to do.
    """
    if tracking_mode == "trajectory":
        return np.concatenate([np.atleast_1d(reference(t + k)) for k in range(n_horizon + 1)]).tolist()
    if tracking_mode == "setpoint":
        return np.atleast_1d(reference(t)).tolist()
    raise ValueError(
        f"Unknown TRACKING_MODE {tracking_mode!r}; expected 'trajectory' or 'setpoint'"
    )


# ---------------------------------------------------------------------------
# Graceful stop on an infeasible solve()
# ---------------------------------------------------------------------------
def try_solve(t, ts, solve_call, tag="mpc"):
    """Call ``solve_call()`` (a zero-arg closure wrapping ``my_mpc.solve(...)``).

    None of MPC/RMPC/RAMPC in this repo relax constraints via a slack variable
    (``add_soft_constraints`` in src/mpc.py is an empty stub, and
    ``options["relax"]`` is parsed but never consumed), so an infeasible QP
    raises instead of degrading gracefully. Returns True on success; on
    failure it prints an explanatory message and returns False so the caller
    can stop the simulation loop cleanly (and still plot what was collected)
    instead of crashing with a raw CasADi traceback.
    """
    try:
        solve_call()
        return True
    except Exception as exc:  # noqa: BLE001 - report and let the caller stop
        print(f"[duffing {tag}] stopping at step {t} (t={t * ts:.2f} s): "
              f"solve() failed -- {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Offline cache for the (K, P) pair from GainSynthesis.synthesize_controller()
# ---------------------------------------------------------------------------
def load_offline_cache(cache_path: Path):
    """Return (K, P) if ``cache_path`` exists, else (None, None)."""
    if not cache_path.exists():
        return None, None
    cache = np.load(cache_path)
    return cache["K"], cache["P"]


def save_offline_cache(cache_path: Path, K, P):
    np.savez(cache_path, K=np.asarray(K), P=np.asarray(P))


# ---------------------------------------------------------------------------
# Closed-loop result CSV I/O (shared schema across mpc/rmpc/rampc_example.py,
# used by run_comparison.py to cache simulation results and re-plot from disk
# without re-simulating)
# ---------------------------------------------------------------------------
_RESULT_COLUMNS = ("t", "r1", "r2", "x1", "v1", "x2", "v2", "u1", "u2", "xa1", "xa2", "xa3", "xa4")


def _rows(values, n_cols: int, name: str):
    """(len, n_cols) float array from a trajectory given as a list/array of
    per-step entries, whatever their singleton dims: (T, n_cols), (T, n_cols, 1)
    -- e.g. a list of casadi ``DM`` column vectors -- or (T,) / (T, 1, 1) when
    ``n_cols == 1`` (``u_star`` is a 1x1 ``DM``). Raises if the number of
    columns does not match instead of silently dropping data."""
    arr = np.asarray(values, dtype=float)
    arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != n_cols:
        raise ValueError(f"save_results_csv: '{name}' has {arr.shape[1]} column(s) per step, expected {n_cols}")
    return arr


def save_results_csv(path: Path, t, r, x_traj, u_traj, xa_traj, extra: dict | None = None):
    """Save one closed-loop run to ``path`` with columns ``_RESULT_COLUMNS``
    plus, if given, one column per ``extra`` entry (e.g. RAMPC's per-step
    theta_hat/theta bounds -- see rampc_example.py) appended in insertion
    order.

    ``x_traj``/``xa_traj`` are (T+1, NX) / (T, NX) or (T+1, NX), ``u_traj`` is
    (T, NU), ``r`` is (T, NY) -- only the first ``len(t)`` rows of each are
    used, so callers can pass the raw trajectories (one state longer than the
    input/reference arrays) without trimming. Per-step entries may carry
    trailing singleton dims (e.g. (T, NX, 1) for x_traj, (T, NU, 1) for
    u_traj when built by appending casadi ``DM``s).
    """
    n = len(t)
    columns = list(_RESULT_COLUMNS)
    x = _rows(x_traj, NX, "x_traj")[:n]
    xa = _rows(xa_traj, NX, "xa_traj")[:n]
    r_ = _rows(r, NY, "r")[:n]
    u = _rows(u_traj, NU, "u_traj")[:n]
    arrays = [
        np.asarray(t)[:n],
        *[r_[:, i] for i in range(NY)],
        *[x[:, i] for i in range(NX)],
        *[u[:, i] for i in range(NU)],
        *[xa[:, i] for i in range(NX)],
    ]
    for name, values in (extra or {}).items():
        columns.append(name)
        arrays.append(_rows(values, 1, name)[:n, 0])
    data = np.column_stack(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header=",".join(columns), comments="")


def load_results_csv(path: Path) -> dict:
    """Return {"t": ..., "r": ..., "x1": ..., ...} from a CSV saved by
    ``save_results_csv`` -- every column actually present (including any
    ``extra`` ones), not just ``_RESULT_COLUMNS``."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}


# ---------------------------------------------------------------------------
# xBound/uBound sidecar (small .npz next to a results CSV) -- run_comparison.py
# needs the actual bounds an example script was run with to draw them as
# reference lines, both right after a fresh run and later from cache (where
# the CSV alone doesn't carry them).
# ---------------------------------------------------------------------------
def save_bounds_npz(csv_path: Path, x_bound, u_bound, **extra):
    """Save x_bound/u_bound (each a (lb, ub) pair) to a small sidecar .npz next
    to ``csv_path``. ``extra`` may carry further small (non-time-series)
    arrays that don't fit the CSV's per-step columns -- e.g. RAMPC's
    ``theta_true`` (the true theta, constant for the whole run)."""
    path = csv_path.with_name(csv_path.stem + "_bounds.npz")
    np.savez(path, x_lb=x_bound[0], x_ub=x_bound[1], u_lb=u_bound[0], u_ub=u_bound[1], **extra)


def load_bounds_npz(csv_path: Path) -> dict:
    """Return {"xBound": (x_lb, x_ub), "uBound": (u_lb, u_ub), **extra} saved
    by ``save_bounds_npz`` for the CSV at ``csv_path``."""
    path = csv_path.with_name(csv_path.stem + "_bounds.npz")
    data = np.load(path)
    result = {"xBound": (data["x_lb"], data["x_ub"]), "uBound": (data["u_lb"], data["u_ub"])}
    for key in data.files:
        if key not in ("x_lb", "x_ub", "u_lb", "u_ub"):
            result[key] = data[key]
    return result

def sys_vertex(A, B, param_vertices):
    """Return the system matrices (A, B) evaluated at the vertices of the
    parameter polytope defined by ``theta_vertices``. Each vertex is a column
    vector in ``theta_vertices``."""

    vert_aug = np.vstack([np.ones((1, param_vertices.shape[1])), param_vertices])
    A_v = np.einsum('ijk,kl->ijl', A, vert_aug)
    B_v = np.einsum('ijk,kl->ijl', B, vert_aug)

    return A_v, B_v


# ---------------------------------------------------------------------------
# RAMPC parameter-set bookkeeping
# ---------------------------------------------------------------------------
def theta_vertex_bounds(col, q):
    """Per-dimension [min, max] of the parameter polytope packed in ``col``.

    ``col`` holds the polytope vertices in homogeneous form
    [1, theta_1, ..., theta_q] (dimension-major, see RAMPC.solve()).
    """
    pts = np.asarray(col).reshape(q + 1, -1).T  # (n_vertices, q+1)
    pts = pts[:, 1:]                             # drop the homogeneous 1 -> (n_vertices, q)
    return pts.min(axis=0), pts.max(axis=0)
