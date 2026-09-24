"""
Robust Adaptive MPC (RAMPC) on the two-mass mass-spring-damper plant
=====================================================================
Closed-loop tracking of both masses' positions (y = [x1, x2]) with a
tube-based robust adaptive MPC built on the affine LPV family A(p), B(p)
(ss_data.npz) and the identified uncertainty set Theta (uncertainty_sets.npz),
with the plant driven by the TRUE physical parameters (p_real in
sys_param.npz), an additive disturbance drawn from the truncation-mismatch
set W_mismatch, and online estimation of theta (in the same reduced basis
space as p_basis) tracked/plotted against its true value and its shrinking
polytopic bound.
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from os import path
from pycvxset.common import constants
from pathlib import Path
from polypympc import RAMPC, GainSynthesis


import common as dc
constants.DEFAULT_LP_SOLVER_STR = "OSQP"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

cep_width_inches = 3.54 * 3
aspect_ratio = 0.618
fig_height = cep_width_inches * aspect_ratio

FIGSIZE = (cep_width_inches, fig_height)
CURRENT_DIR = path.dirname(path.abspath(__file__))

MODEL_DIR = "model_data"

TRACK_MODE = "trajectory"  # "setpoint"|"trajectory"
TRACK_SIGNAL = "harmonic"    # "square"|"harmonic"
FULL_PATH = path.join(CURRENT_DIR, MODEL_DIR)
SEED = 0  # seed of the disturbance generator

GAIN = 1.0
if TRACK_SIGNAL == "harmonic":
    GAIN = 4.0

UNCERTAINTY_MODEL = "polytope"  # "polytope"|"box"|"hull"

LINE_WIDTH = 2.8
MARKER_SIZE = 6
FONT_SIZE = {
    "labels": 24,
    "ticks": 20,
    "legend": 22,
}

# Affine LPV family A(p), B(p): RAMPC needs the whole family (1 + n_p
# slices), not just the nominal slice.
A, B, C, D, Ts = dc.load_ss(path.join(FULL_PATH, "ss_data.npz"))
assert A.shape[:2] == (4, 4) and B.shape[:2] == (4, 2), "ss_data.npz shapes don't match the 2-mass MSD plant"

C = dc.lift_output_matrix(C, A.shape[2])

Q, R = 10 * np.eye(A.shape[0]), 0.005 * np.eye(B.shape[1])  # State and input cost matrices

# True physical plant: the actual (m1, m2, k, c), distinct from the nominal
# model above, used to drive the closed loop. p_basis is the SAME test case's
# coordinate in theta space (the reduced basis th_hat lives in) -- used below
# as the "true theta" reference line, not p_real (a different space).
p_real, p_basis, p_cheb, n_p = dc.load_point(path.join(FULL_PATH, "sys_param.npz"))
m1, m2, k, c = p_real
print(f"True physical parameters: m1={m1:.4f}, m2={m2:.4f}, k={k:.4f}, c={c:.4f}")
plant = dc.make_msd_plant(m1, m2, k, c, Ts)

N = 10  # Prediction horizon

## Load uncertainty model

u_set_path = Path(path.join(FULL_PATH, "uncertainty_sets.npz"))
param_set = dc.load_uncertainty_set(u_set_path, UNCERTAINTY_MODEL)
W = dc.load_mismatch_set(u_set_path)

## Vertex representation of system matrices for RAMPC
a_v, b_v = dc.sys_vertex(A, B, param_set.V.T)

## Load/Compute offline ingredients for the RAMPC controller

try:
    cache_path = Path(path.join(FULL_PATH, "rampc_cache.npz"))
    K, P, controller_mode = dc.load_offline_cache(
        cache_path
    )
    print("Loaded offline ingredients from cached.\n")
    print("Controller defined in mode: ", controller_mode)
except (FileNotFoundError, OSError, ValueError, KeyError):
    gain_synt = GainSynthesis(a_v, b_v, Q=Q, R=R, mode='volume', contraction_factor=0.99)
    gain_synt.synthesize_controller()
    K = gain_synt.get_gain()
    P = gain_synt.get_terminal_weight()

    np.savez(
        path.join(FULL_PATH, "rampc_cache.npz"),
        K=K,
        P=P,
        type="volume",
    )
    print("Computed offline ingredients and saved to cache.\n")

print("Gain K: ", K)
print("Lyapunov Matrix P: ", P)

# Define the RAMPC parameters. Bounds match the MATLAB reference (Complete_5ag.m):
# umax=25, xmax=[2.22, 6] (position, velocity) per mass, symmetric across masses.
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    "xBound": (np.array([-2.22, -6.0, -2.22, -6.0]), np.array([2.22, 6.0, 2.22, 6.0])),
    "uBound": (np.array([-25.0, -25.0]), np.array([25.0, 25.0])),
    "name": "msd_rampc",
    "ref": dc.ref_option(TRACK_MODE),
    "K": K,
    "P": P,
    "W": W,
    "lam": 0.998,
    "th_hat_0": p_cheb,
    "theta": param_set,
    "par_filter": "rls",
    'lpv_flag': False,
    # Internal representation of the parameter sets in the SetUpdater:
    #   "reduced" -> vertices from fixed facet bases (cheap; can fall outside the set once the LP
    #                tightens the facets non-uniformly), "full" -> exact vertices at every step
    "set_representation": "full",
}

# Create the RAMPC controller
my_rampc = RAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# initialize() with no argument would ignore options["P"] and recompute a
# terminal weight from GainSynthesis, which (with K already given) is NOT
# positive definite -> pass the cached, valid terminal weight explicitly.
my_rampc.initialize(P=P)

# Simulate the system
x0 = np.zeros(4)  # Initial state [x1, v1, x2, v2]
T = 200  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
xa_traj = []
r_traj = []
rng = np.random.default_rng(SEED)
param_est = []
bound_vertices = []
q_par = n_p  # number of uncertain (reduced-basis) parameters, needed to decode theta_vertices

# Amplitude +/-2.2 matches the MATLAB reference (pw_ref amplitude); the two
# outputs get different periods/phases so x1, x2 track genuinely independent
# references instead of moving in lockstep.
reference = dc.make_reference(Ts, kind=TRACK_SIGNAL, amp=2.2 * GAIN,
                               n_signals=2, period=[5.0, 6.8], phase=[0.4, 0.0])

for t in range(T):
    r = dc.reference_arg(reference, t, N, TRACK_MODE)
    # w = dc.make_disturbance(W, rng)
    y0 = ca.vertcat(x_traj[-1][0], x_traj[-1][2])  # y = [x1, x2] = C @ x
    my_rampc.solve(x_traj[-1], y0, r)
    # Logged AFTER solve() (estimate/set actually used at step k, no None at
    # k=0), as copies, decoding theta_vertices (coordinate-major column) into
    # an (nv, q) array of theta vertices.
    param_est.append(my_rampc.th_hat[1:, 0].copy())                                    # (q,)
    bound_vertices.append(my_rampc.theta_vertices[:, 0].reshape(q_par + 1, -1).T[:, 1:].copy())  # (nv, q)
    u_traj.append(my_rampc.u_star)
    xa_traj.append(my_rampc.xa_star[:, -1])
    x_next = plant[1](x_traj[-1], my_rampc.u_star) #+ w
    x_traj.append(x_next)
    r_traj.append(r[:2])  # current-step [r1, r2] (the first NY entries of the flattened window)
x_traj = np.array(x_traj)
u_traj = np.array(u_traj)
xa_traj = np.array(xa_traj)
r_traj = np.array(r_traj)

# Set fontsize for each font size key
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': FONT_SIZE['labels'],
    'axes.titlesize': FONT_SIZE['labels'],
    'axes.labelsize': FONT_SIZE['labels'],
    'xtick.labelsize': FONT_SIZE['ticks'],
    'ytick.labelsize': FONT_SIZE['ticks'],
    'legend.fontsize': FONT_SIZE['legend']
})

# Output tracking (y = [x1, x2] vs. reference) + input effort (u1, u2), each
# with its own bound drawn as a reference line.
k_x = np.arange(x_traj.shape[0])  # k = 0, ..., T
k_u = np.arange(u_traj.shape[0])  # k = 0, ..., T-1
BOUND_COLOR = "#FF0000B0"

fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")

y_labels = (r"$x_1$", r"$x_2$")
y_state_idx = (0, 2)  # x1, x2 live at state indices 0, 2 (x = [x1, v1, x2, v2])
for i in range(2):
    ax[i].plot(k_x, x_traj[:, y_state_idx[i]], color="#006EFFEA", label=r'$x_k$',
               marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)
    ax[i].plot(k_u, r_traj[:, i], color="#383838B0", ls='--', label=r'$r_k$', lw=LINE_WIDTH)
    ax[i].hlines(
        [my_rampc.options.xBound[0][y_state_idx[i]], my_rampc.options.xBound[1][y_state_idx[i]]],
        k_x[0], k_x[-1], color=BOUND_COLOR, ls='--', label=r'$x$ bound', lw=LINE_WIDTH,
    )
    ax[i].set_ylabel(y_labels[i])

dc.shared_legend(fig, ax)
ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
ax[-1].set_xlim(k_x[0], k_x[-1])  # sharex: no margin beyond the simulation window
plt.show()

fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")

u_colors = ("#E28000D2", "#9B30FFD2")
for i in range(2):
    ax[i].plot(k_u, u_traj[:, i], color=u_colors[i], label=f'$u_{{{i + 1},k}}$', lw=LINE_WIDTH)
    ax[i].hlines(
        [my_rampc.options.uBound[0][i], my_rampc.options.uBound[1][i]],
        k_x[0], k_x[-1], color=BOUND_COLOR, ls='--', label=r'$u$ bound', lw=LINE_WIDTH,
    )
    ax[i].set_ylabel(f'$u_{i + 1}$')

dc.shared_legend(fig, ax)
ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
ax[-1].set_xlim(k_x[0], k_x[-1])  # sharex: no margin beyond the simulation window
plt.show()

## Plot the projection on axis of parameters evolution over time (one subplot
## per parameter): th_hat, upper/lower bound on the parameter (from the
## polytope) and the true parameter value (p_basis, the test point's own
## coordinate in the same reduced theta space).
param_est = np.array(param_est)                                  # (T, q)
up_bound = np.array([v.max(axis=0) for v in bound_vertices])      # (T, q) projection of the set on each parameter axis
low_bound = np.array([v.min(axis=0) for v in bound_vertices])     # (T, q)

fig, ax = plt.subplots(param_est.shape[1], 1, figsize=FIGSIZE, sharex=True, layout="constrained")

for i in range(param_est.shape[1]):
    ax[i].plot(k_u, param_est[:, i], color="#00FF00EA", label=r'$\hat{\theta}_k$',
               marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)

    ax[i].set_ylabel(r"$\hat{\theta}_{%d}$" % (i + 1))

    ax[i].fill_between(k_u, low_bound[:, i], up_bound[:, i],
                        color="#006EFFEA", alpha=0.2, label=r'$\theta$ bounds')

    # Upper/lower bound lines share the fill_between's label above (not
    # labeled again here). Same colors are reused across all subplots (one
    # per vector component), so a single shared legend above the figure
    # replaces a per-subplot one instead of repeating it 3 times.
    ax[i].plot(k_u, up_bound[:, i], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
    ax[i].plot(k_u, low_bound[:, i], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
    ax[i].hlines(p_basis[i], k_x[0], k_x[-1], color="#FF0000EA", ls='--',
                 label=r'$\theta$ true value', lw=LINE_WIDTH)

dc.shared_legend(fig, ax)

ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
plt.show()

# Save plot data in csv format
# Schema (common.py, _RESULT_COLUMNS): t, r1, r2, x1, v1, x2, v2, u1, u2,
# xa1..xa4, plus theta_hat_i/theta_up_i/theta_low_i (i = 1..n_p) appended as
# extra columns -- enough to rebuild the parameter plot from the CSV alone
# (theta_true itself is in the _bounds.npz sidecar, constant for the whole
# run). Rows are truncated to len(t) = T, i.e. the final state x_T is not
# stored.
RESULTS_DIR = Path(CURRENT_DIR) / "results" / TRACK_SIGNAL
csv_path = RESULTS_DIR / "rampc_example.csv"
t_axis = np.arange(T) * Ts  # [s]; step k = t / Ts
theta_extra = {}
for i in range(param_est.shape[1]):
    theta_extra[f"theta_hat_{i + 1}"] = param_est[:, i]
    theta_extra[f"theta_up_{i + 1}"] = up_bound[:, i]
    theta_extra[f"theta_low_{i + 1}"] = low_bound[:, i]
dc.save_results_csv(csv_path, t_axis, r_traj, x_traj, u_traj, xa_traj, extra=theta_extra)
# Constraint levels drawn as horizontal lines: sidecar rampc_example_bounds.npz
dc.save_bounds_npz(csv_path, my_rampc.options.xBound, my_rampc.options.uBound, theta_true=p_basis)
print("Saved plot data to:", csv_path)
