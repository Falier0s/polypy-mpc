"""RAMPC turtlebot example with the reference trajectory read from a CSV file.

Usage:
    python rampc_example.py <trajectory.csv> [--steps N] [--seed S]
                            [--snapshot SEC] [--save {pdf,png,jpg,csv,all}]

The CSV is looked up inside the ``reference_trajectories`` folder next to this
script (a bare name works, with or without the ``.csv`` extension); an absolute
or relative path is accepted too. The file must have a header row containing at
least the columns ``x``, ``y`` and ``theta`` (extra columns such as ``qz``,
``qw`` are ignored).

This mirrors ``kinematic_example.py`` -- same LPV model, weights, horizon and
MPC options -- only the reference construction differs: instead of building a
circle analytically, one pose per simulation step is taken from the CSV.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import casadi as ca
from math import pi

from pycvxset import Polytope
from pycvxset.common import constants

constants.DEFAULT_LP_SOLVER_STR = "CLARABEL"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
REF_DIR = SCRIPT_DIR / "reference_trajectories"

try:
    from rampc import CRAMPC
except ModuleNotFoundError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src import CRAMPC

# The reference pose handed to the controller at step ``t`` is sample ``t + LOOKAHEAD``
# (same convention as kinematic_example.py).
LOOKAHEAD = 5

# Sampling period [s]: the discrete model uses B / 30, i.e. fs = 30 Hz.
DT = 1.0 / 30.0

# Figure style shared by every plot (and by the images exported with --save).
# Tweak these three and rerun.
cep_width_inches = 3.54 * 3
aspect_ratio = 0.618
fig_height = cep_width_inches * aspect_ratio

FIGSIZE = (cep_width_inches, fig_height)
FONTSIZE = 16      # base font size (pt) for labels, ticks, legend, titles
LINEWIDTH = 2.2    # width of the plotted curves
MARKERSIZE = 8.5        # size of every point marker (estimate triangles, snapshot x/o, star)
MARKEREDGEWIDTH = 0.6 * LINEWIDTH  # outline width of those markers
EST_MARKER_COUNT = 30   # approx. number of triangle markers along the estimate line
EST_LINEWIDTH = 0.75 * LINEWIDTH  # estimated-parameter line: a bit thinner than the rest
GRID_ALPHA = 0.5        # opacity of the ":" grid drawn on every plot

# Text rendering. USETEX=True drives a real LaTeX install (needs `latex` + `dvipng`
# on PATH) and typesets everything in Times via newtx -- matches an Elsevier paper.
# USETEX=False keeps a Times-like serif (STIX) with no external dependency.
USETEX = False

# Per-plot legend placement: (x, y) = bottom-left corner of the legend box, in
# DATA coordinates of that plot's axes. Set an entry to None for matplotlib's
# automatic "best" location.
LEGEND_ANCHOR = {
    "bounds": (68.0, -0.15),   # bottom subplot data coords -> x = t [s], y = delta_2
    "snapshot": None,       # (delta_1, delta_2) plane
    "trajectory": None,     # (X_I, Y_I) plane  (axes are swapped)
}


def place_legend(ax, key, handles=None):
    """Legend on ``ax`` at LEGEND_ANCHOR[key] (bottom-left corner, data coords)."""
    kw = {} if handles is None else {"handles": handles}
    anchor = LEGEND_ANCHOR.get(key)
    if anchor is None:
        ax.legend(loc="best", **kw)
    else:
        ax.legend(loc="lower left", bbox_to_anchor=anchor,
                  bbox_transform=ax.transData, **kw)

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "xtick.labelsize": FONTSIZE - 1,
    "ytick.labelsize": FONTSIZE - 1,
    "legend.fontsize": FONTSIZE - 1,
    "figure.titlesize": FONTSIZE,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,
    "lines.markeredgewidth": MARKEREDGEWIDTH,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": GRID_ALPHA,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # Times-like math when USETEX is False
    "text.usetex": USETEX,
})
if USETEX:
    # Times for both text and math; needs the newtx package in the LaTeX install.
    plt.rcParams["text.latex.preamble"] = r"\usepackage{newtxtext,newtxmath}"


def resolve_csv(name):
    """Resolve ``name`` to an existing CSV path (bare name -> reference_trajectories/)."""
    candidates = [Path(name), REF_DIR / name, REF_DIR / f"{name}.csv"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    available = [p.name for p in sorted(REF_DIR.glob("*.csv"))]
    raise FileNotFoundError(
        f"Reference CSV '{name}' not found (looked in {', '.join(str(c) for c in candidates)}). "
        f"Available in reference_trajectories/: {available}"
    )


def load_reference(csv_path):
    """Return an (M, 3) array of [x, y, theta] poses from a CSV with a header row."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.dtype.names is None or not {"x", "y", "theta"}.issubset(data.dtype.names):
        raise ValueError(
            f"{csv_path.name} must have 'x', 'y' and 'theta' columns; found {data.dtype.names}"
        )
    pose = np.column_stack([data["x"], data["y"], data["theta"]])
    pose[:, 2] = np.unwrap(pose[:, 2])  # continuous heading -> no 2*pi jumps for the controller
    return pose


def polytope_vertices_2d(theta_vertices_col):
    """Vertices of the parameter polytope for one dilation step l, projected onto its 2 dims.

    ``my_mpc.theta_vertices[:, l]`` packs ``n_vertices`` points in homogeneous form
    ``[1, delta_1, delta_2]`` (dimension-major); drop the constant coordinate.
    """
    pts = np.asarray(theta_vertices_col).reshape(3, -1).T  # (n_vertices, 3)
    return pts[:, 1:3]


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "csv", help="reference trajectory CSV (name inside reference_trajectories/, or a path)"
)
parser.add_argument(
    "--steps", type=int, default=None, help="cap the number of simulation steps"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="seed for the process-noise RNG, so runs are reproducible (default: 0)",
)
parser.add_argument(
    "--snapshot",
    type=float,
    default=None,
    metavar="SEC",
    help="also plot the parameter polytope at time SEC seconds (nearest sample is used); "
    "the instant is starred on the bounds plot",
)
parser.add_argument(
    "--save",
    choices=["pdf", "png", "jpg", "csv", "all"],
    default=None,
    help="export the results into a local timestamped folder: an image format saves only "
    "the figures, 'csv' saves only the per-step data sheet, 'all' saves both (figures as pdf)",
)
args = parser.parse_args()

csv_path = resolve_csv(args.csv)
ref_pose = load_reference(csv_path)
print(f"Loaded reference '{csv_path.name}' with {len(ref_pose)} poses")

# --- LPV kinematic model (identical to kinematic_example.py) -------------------
A = np.block([[[np.eye(3)]], [[np.zeros((3, 3))]], [[np.zeros((3, 3))]]])
A = A.transpose(1, 2, 0).copy()

B = np.block(
    [
        [[np.zeros((2, 2))], [0, 1]],
        [[1, 0], [np.zeros((2, 2))]],
        [[np.zeros((1, 2))], [1, 0], [np.zeros((1, 2))]],
    ]
) / 30
B = B.transpose(1, 2, 0).copy()

C = np.block([[[np.eye(3)]], [[np.zeros((3, 3))]], [[np.zeros((3, 3))]]])
C = C.transpose(1, 2, 0).copy()

Q, R = 10 * np.diag(np.array([1, 1, 0.8])), 0.5 * np.eye(2)
Theta = Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=np.ones((4, 1)))

# Define the MPC parameters
N = 5  # Prediction horizon

W = Polytope(
    A=np.block([[np.eye(3)], [-np.eye(3)]]),
    b=np.array([0.018, 0.018, 0.065, 0.018, 0.018, 0.065]),
)

E = Polytope(
    A=np.block([[np.eye(3)], [-np.eye(3)]]),
    b=np.array([0.018, 0.018, 0.065, 0.018, 0.018, 0.065]),
)

npfile = np.load(SCRIPT_DIR / "kin_offline_mat.npz")
K = npfile["arr_0"]
P = npfile["arr_1"]

options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    "xBound": (-np.array([10, 5, 1e4]), np.array([10, 5, 1e4])),
    "uBound": (-np.array([0.46, 1.90]), np.array([0.46, 1.90])),
    "name": "turtle_controller",
    "W": W,
    "E": E,
    "K": K,
    "lam": 1,
    "theta": Theta,
    "par_filter": "rls",
    "forgetting": 0.96,  # constant exponential forgetting factor (projected GF-RLS)
    "lpv_flag": True,
    "ref": "ref",
    "max_delta_th": 0.001,
    "th_hat_0": [0.0, 1.0]
}

# Create the MPC controller
my_mpc = CRAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)
# Initialize the controller
my_mpc.initialize(mode="LQR", P=P)

# Simulate the system
x0 = np.array([0, 0, pi/2])  # Initial state
T = len(ref_pose) - LOOKAHEAD
if args.steps is not None:
    T = min(T, args.steps)
x_traj = [ca.DM(x0)]
u_traj = []
polytopes = []
rng = np.random.default_rng(args.seed)  # process-noise RNG (reproducible via --seed)
print(f"process-noise seed: {args.seed}")

# Everything is logged during the loop and plotted afterwards (plotting inside the
# loop leaks one figure per iteration).
theta_vertices_hist = []  # per step: full my_mpc.theta_vertices, axes [i vertex, l dilation step]
th_hat_hist = []          # per step: estimated parameter delta_hat [delta_1, delta_2]
p_real_hist = []          # per step: real parameter delta* = [cos(theta_k), sin(theta_k)]

for t in range(T):
    polytopes.append(my_mpc.initial_tube())
    w = rng.random((3, 1)) * np.array([[0.010], [0.010], [0.03]])
    my_mpc.solve(x_traj[-1] + w, x_traj[-1], r=list(ref_pose[t + LOOKAHEAD].flatten()))
    u_traj.append(my_mpc.u_star)
    theta_k = x_traj[-1][-1].toarray()

    theta_vertices_hist.append(np.array(my_mpc.theta_vertices, copy=True))
    th_hat_k = np.asarray(my_mpc.th_hat, dtype=float).flatten()  # [1, th_hat_1, th_hat_2]
    th_hat_hist.append([th_hat_k[1], th_hat_k[2]])
    p_real_hist.append([np.cos(theta_k[0, 0]), np.sin(theta_k[0, 0])])

    x_next = np.einsum(
        "ij,jkl->kl",
        np.array([[1, np.cos(theta_k[0, 0]), np.sin(theta_k[0, 0])]]),
        A.transpose(2, 0, 1) @ x_traj[-1].toarray()
        + B.transpose(2, 0, 1) @ my_mpc.u_star.toarray(),
    )
    x_traj.append(ca.DM(x_next))

x_traj = np.array(x_traj)
u_traj = np.array(u_traj)

th_hat_hist = np.array(th_hat_hist)  # (T, 2)
p_real_hist = np.array(p_real_hist)  # (T, 2)
t_axis = np.arange(len(theta_vertices_hist)) * DT  # seconds

# Snapshot time (seconds) -> nearest available sample.
snap_idx = None
if args.snapshot is not None:
    if len(t_axis) == 0:
        parser.error("--snapshot given but the simulation produced no steps")
    snap_idx = int(np.argmin(np.abs(t_axis - args.snapshot)))
    snap_time = t_axis[snap_idx]
    if not np.isclose(snap_time, args.snapshot):
        print(f"--snapshot {args.snapshot:g} s -> nearest sample t={snap_time:.4f} s (step {snap_idx})")

# Axis-aligned bounds of the j=0 parameter polytope at every simulation step.
theta_vertices_j0 = [tv[:, 0] for tv in theta_vertices_hist]
param_lb = np.zeros((len(theta_vertices_j0), 2))
param_ub = np.zeros((len(theta_vertices_j0), 2))
for k, col in enumerate(theta_vertices_j0):
    v2d = polytope_vertices_2d(col)
    param_lb[k] = v2d.min(axis=0)
    param_ub[k] = v2d.max(axis=0)

# --- Plot: per-dimension parameter-set bounds vs real / estimated parameter ---
dim_names = [r"$\delta_1$", r"$\delta_2$"]
est_markevery = max(len(t_axis) // EST_MARKER_COUNT, 1)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=FIGSIZE)
for d, ax in enumerate(axes):
    band_color = f"C{d}"
    ax.plot(t_axis, param_ub[:, d], "--", color=band_color, linewidth=LINEWIDTH)
    ax.plot(t_axis, param_lb[:, d], "--", color=band_color, linewidth=LINEWIDTH)
    ax.fill_between(t_axis, param_lb[:, d], param_ub[:, d], color=band_color, alpha=0.2)
    ax.plot(t_axis, p_real_hist[:, d], "-", color="k", linewidth=LINEWIDTH)
    ax.plot(t_axis, th_hat_hist[:, d], "-", color="red", marker="^",
            markevery=est_markevery, markersize=MARKERSIZE, linewidth=EST_LINEWIDTH)
    if snap_idx is not None:
        ax.plot(
            snap_time,
            p_real_hist[snap_idx, d],
            "*",
            color="tab:red",
            markeredgecolor="k",
            zorder=5,
        )
    ax.set_ylabel(dim_names[d])

# explicit legend: includes the bound entry of BOTH subplots (delta_1 and delta_2)
legend_items = [
    Line2D([], [], color="C0", linestyle="--", lw=LINEWIDTH, label=r"$\delta_1$ bounds"),
    Line2D([], [], color="C1", linestyle="--", lw=LINEWIDTH, label=r"$\delta_2$ bounds"),
    Line2D([], [], color="k", linestyle="-", lw=LINEWIDTH, label=r"True $\delta^{*}_k$"),
    Line2D([], [], color="red", linestyle="-.", marker="^", markersize=MARKERSIZE, lw=EST_LINEWIDTH,
           label=r"Estimate $\hat{\delta}_k$"),
]
if snap_idx is not None:
    legend_items.append(
        Line2D([], [], color="tab:red", marker="*", linestyle="none",
               markeredgecolor="k", label=f"snapshot ($k={snap_idx}$)")
    )
place_legend(axes[1], "bounds", handles=legend_items)
# shared x-axis: label only the lower subplot
axes[-1].set_xlabel(r"$\mathrm{Time~(s)}$")
fig.suptitle(r"Parameter-set bounds ($l=0$, i.e. $\Delta_k$)")
fig.tight_layout()

# --- Optional plot: the parameter polytope at the chosen instant -------------
if snap_idx is not None:
    tv = theta_vertices_hist[snap_idx]
    n_j = tv.shape[1]
    layers = list(range(n_j))  # all dilation steps l, drawn in increasing order

    fig_snap, ax_snap = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap("viridis")
    legend_handles = []
    for layer_idx, l in enumerate(layers):
        color = cmap(layer_idx / max(len(layers) - 1, 1))
        try:
            Polytope(V=polytope_vertices_2d(tv[:, l])).plot(
                ax=ax_snap,
                patch_args={"facecolor": "none", "edgecolor": color, "linewidth": LINEWIDTH},
            )
        except Exception as exc:  # degenerate (point/segment) parameter set
            print(f"snapshot: skipping dilation step l={l} (cannot plot polytope: {exc})")
            continue
        legend_handles.append(Patch(facecolor="none", edgecolor=color, label=f"$l={l}$"))
    ax_snap.plot(*p_real_hist[snap_idx], "x", color="k", label=r"True $\delta^{*}_k$")
    ax_snap.plot(
        *th_hat_hist[snap_idx], "o", markerfacecolor="none", markeredgecolor="k",
        label=r"Estimate $\hat{\delta}_k$",
    )
    marker_handles, _ = ax_snap.get_legend_handles_labels()
    place_legend(ax_snap, "snapshot", handles=legend_handles + marker_handles)
    ax_snap.set_xlabel(r"$\delta_1$")
    ax_snap.set_ylabel(r"$\delta_2$")
    ax_snap.set_title(
        rf"Parameter set $\mathcal{{D}}_l(\Delta_k)$ at instant $k={snap_idx}$ ($t={snap_time:.2f}$ s)"
    )
    ax_snap.axis("equal")
    fig_snap.tight_layout()

# --- Plot: tracked vs reference trajectory (axes data swapped -> rotated view) --
fig_traj = plt.figure(figsize=FIGSIZE)
plt.plot(ref_pose[:, 1], ref_pose[:, 0], label=r"Reference $\mathbf{r}$", linestyle="--", color="C3", linewidth=LINEWIDTH)
plt.plot(x_traj[:, 1], x_traj[:, 0], label=r"Ground Truth $\mathbf{x}$", linestyle="-", color="C0", linewidth=LINEWIDTH)
plt.xlabel(r"$X_I$", fontsize=FONTSIZE)
plt.ylabel(r"$Y_I$", fontsize=FONTSIZE)
plt.title(f"Turtlebot Trajectory Tracking - {csv_path.stem.replace('_', ' ')}")
place_legend(fig_traj.axes[0], "trajectory")
plt.axis("equal")

# --- Optional: export results to a local timestamped folder -----------------
if args.save is not None:
    save_figs = args.save in ("pdf", "png", "jpg", "all")
    save_data = args.save in ("csv", "all")
    img_ext = "pdf" if args.save == "all" else args.save

    out_dir = Path.cwd() / f"rampc_results_{csv_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_figs:
        # exported figures carry axis labels only -> drop every title
        fig.suptitle("")
        fig_traj.axes[0].set_title("")
        if snap_idx is not None:
            ax_snap.set_title("")

        fig.savefig(out_dir / f"param_bounds.{img_ext}", bbox_inches="tight")
        if snap_idx is not None:
            fig_snap.savefig(out_dir / f"param_set_snapshot.{img_ext}", bbox_inches="tight")
        fig_traj.savefig(out_dir / f"trajectory_tracking.{img_ext}", bbox_inches="tight")

    if save_data:
        n = len(t_axis)
        X = np.asarray(x_traj, dtype=float).reshape(len(x_traj), -1)[:n]
        U = np.asarray(u_traj, dtype=float).reshape(len(u_traj), -1)[:n]
        ref_win = ref_pose[LOOKAHEAD:LOOKAHEAD + n]
        table = np.column_stack([
            t_axis, X[:, 0], X[:, 1], X[:, 2], U[:, 0], U[:, 1],
            ref_win[:, 0], ref_win[:, 1], ref_win[:, 2],
            th_hat_hist[:, 0], th_hat_hist[:, 1],
            p_real_hist[:, 0], p_real_hist[:, 1],
            param_lb[:, 0], param_ub[:, 0], param_lb[:, 1], param_ub[:, 1],
        ])
        header = ",".join([
            "t", "x", "y", "theta", "v", "omega",
            "ref_x", "ref_y", "ref_theta",
            "th_hat_1", "th_hat_2", "p_real_1", "p_real_2",
            "param_lb_1", "param_ub_1", "param_lb_2", "param_ub_2",
        ])
        np.savetxt(out_dir / "results.csv", table, delimiter=",", header=header, comments="")

    print(f"saved '{args.save}' results to: {out_dir}")

plt.show()
