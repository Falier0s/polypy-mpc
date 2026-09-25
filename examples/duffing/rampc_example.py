"""
Nominal RAMPC on duffing oscillator
=================================
ADD TEXT

"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from os import path
from pycvxset import Polytope
from pycvxset.common import constants
from pathlib import Path
from polypympc import RAMPC, GainSynthesis, InvariantSet


import duffing_common as dc
constants.DEFAULT_LP_SOLVER_STR = "OSQP"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

cep_width_inches = 3.54 * 3
aspect_ratio = 0.618
fig_height = cep_width_inches * aspect_ratio

FIGSIZE = (cep_width_inches, fig_height)
CURRENT_DIR = path.dirname(path.abspath(__file__))
# Define the system dynamics

# [DEPRECATED] no version subfolder (which export was this?):
# MODEL_DIR = "model_data"
# [v3 FIX] export with fixed gauge (A[0,1,0] = Ts) and plant units ("physical" frame)
MODEL_DIR = "model_data\\v4"

TRACK_MODE = "trajectory" # "setpoint"|"trajectory"
TRACK_SIGNAL = "square" # "square"|"harmonic"
FULL_PATH = path.join(CURRENT_DIR, MODEL_DIR)
DATASET = "inner" # "inner"|"edge"
SEED = 0 # seed of the disturbance generator

GAIN = 1.0
if TRACK_SIGNAL == "harmonic":
    GAIN = 3.5

UNCERTAINTY_MODEL = "polytope" # "polytope"|"hyperbox|"hull"

LINE_WIDTH = 2.8
MARKER_SIZE = 6
FONT_SIZE = {
    "labels": 24,
    "ticks": 20,
    "legend": 22,
}

#import the real system dynamics
A, B, C, D, Ts = dc.load_ss_duffing(path.join(FULL_PATH, "ss_data.npz"))
# [DEPRECATED] manual sign flips (were active here, on ALL the LPV slices): with v3 the notebook
# fixes the state-2 gauge at the source (A[0,1,0] = Ts > 0, A[1,0,0] < 0, B[1,0,0] > 0), so
# nothing to flip here. Applied on top of v3 they would re-break the convention.
# A[0,1,:] = -A[0,1,:]
# A[1,0,:] = -A[1,0,:]
# B[1,0,:] = -B[1,0,:] # Flip the sign of the second column of B to match the duffing dynamics
# [v3 FIX] consistency check of the export convention (nominal slice; fails loudly on a wrong/old export)
# assert np.isclose(A[0, 1, 0], Ts) and A[1, 0, 0] < 0 and B[1, 0, 0] > 0, "ss_data.npz not in the v3 convention"
assert np.isclose(A[0, 1, 0], Ts, rtol=0.05) and A[1, 0, 0] < 0 and B[1, 0, 0] > 0

C = dc.lift_output_matrix(C, A.shape[2])

Q, R = 10*np.eye(A.shape[0]), 0.005*np.eye(B.shape[1])  # State and input cost matrices

# Real duffing oscillator dynamics

## Extract the validation points for the duffing oscillator
## duff_param = [-delta, - alpha, -beta]

duff_param, param_expected, param_center = dc.load_validation_point(path.join(FULL_PATH, "sys_param.npz"), DATASET)
# [v3 OK, unchanged] exported p_real is [-delta, -alpha, -beta]: the sign flip is already here
# (mpc_/rmpc_example.py were the ones missing it)
delta, alpha, beta = - duff_param
gamma = 1.0
print("Duffing parameters: delta = {}, alpha = {}, beta = {}, gamma = {}".format(delta, alpha, beta, gamma))
plant = dc.make_duffing_plant(delta, alpha, beta, gamma, Ts)

N = 10  # Prediction horizon

## Load Uncertainty model

u_set_path = Path(path.join(FULL_PATH, "uncertainty_sets.npz"))
param_set = dc.load_uncertainty_set(u_set_path, UNCERTAINTY_MODEL)

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

W_mismatch = dc.load_mismatch_set(u_set_path, "all")

# Define the RAMPC parameters
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    # "xBound": (np.array([-0.3, -1.0e4]), np.array([0.3, 1.0e4])),
    "xBound": (np.array([-0.3, -2.544]), np.array([0.3, 2.544])),
    "uBound": (np.array([-5]), np.array([5])),
    "name": "duffing_rampc",
    "ref": dc.ref_option(TRACK_MODE),
    "K": K,
    "P": P,
    "W": Polytope(A=W_mismatch.A, b=W_mismatch.b + 0.001),
    "lam": 0.998,
    "th_hat_0": param_center,
    "theta": param_set,
    "par_filter": "rls",
    'lpv_flag': False,
    # [NEW] internal representation of the parameter sets in the SetUpdater:
    #   "reduced" -> vertices from fixed facet bases (cheap; can fall outside the set once the LP
    #                tightens the facets non-uniformly), "full" -> exact vertices at every step
    "set_representation": "full",
}

W = Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=0.001*np.ones((4,1)))

# Create the RAMPC controller
my_rampc = RAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# [DEPRECATED] initialize() with no argument: RMPC.initialize ignores options["P"] and recomputes
# a terminal weight from GainSynthesis, which (with K already given) is NOT positive definite ->
# the QP cost is structurally non-convex, so OSQP eventually returns garbage (error_on_fail=False)
# my_rampc.initialize()
# [FIX] pass the cached, valid terminal weight explicitly
# Initialize the controller
my_rampc.initialize(P=P)
# Simulate the system
x0 = np.array([0.0, 0.0])  # Initial state
T = 200  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
xa_traj = []
r_traj = []
rng = np.random.default_rng(SEED)
param_est = []
bound_vertices = []
q_par = param_center.shape[0]  # [v3 FIX] number of uncertain parameters (needed to decode theta_vertices)



for t in range(T):
    r = dc.reference_arg(dc.make_reference(Ts, kind=TRACK_SIGNAL,amp=0.3*GAIN), t, N, TRACK_MODE)
    w = dc.make_disturbance(W, rng)
    # [DEPRECATED] logged BEFORE solve(): th_hat is None at k=0 and (being updated in place inside
    # solve) every entry pointed to the SAME array -> constant curve; theta_vertices has a different
    # layout at k=0 ((nv, q+1)) than afterwards (((q+1)*nv, 1), coordinate-major)
    # param_est.append(my_rampc.th_hat)
    # bound_vertices.append(my_rampc.theta_vertices)
    my_rampc.solve(x_traj[-1], x_traj[-1][0], r)
    # [v3 FIX] log AFTER solve() (estimate/set actually used at step k, no None at k=0), as COPIES,
    # and decode the vertices (coordinate-major column) into an (nv, q) array of theta vertices
    param_est.append(my_rampc.th_hat[1:, 0].copy())                                    # (q,)
    bound_vertices.append(my_rampc.theta_vertices[:, 0].reshape(q_par + 1, -1).T[:, 1:].copy())  # (nv, q)
    u_traj.append(my_rampc.u_star)
    xa_traj.append(my_rampc.xa_star[:,-1])
    x_next = plant[1](x_traj[-1], my_rampc.u_star) + w
    x_traj.append(x_next)
    r_traj.append(r[0])
x_traj = np.array(x_traj)
u_traj = np.array(u_traj)
xa_traj = np.array(xa_traj)
r_traj = np.array(r_traj)
# print(polytopes[0])

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


# Plot the results

k_x = np.arange(x_traj.shape[0])  # k = 0, ..., T
k_u = np.arange(u_traj.shape[0])  # k = 0, ..., T-1
BOUND_COLOR = "#FF0000B0"

fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")

ax[0].plot(k_x, x_traj[:, 0], color="#006EFFEA", label=r'$x_{1,k}$', marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)
ax[0].plot(k_u, r_traj, color="#383838B0", ls='--', label=r'$r_k$', lw=LINE_WIDTH)
# Bounds span exactly the simulated time range (one call -> one legend entry)
ax[0].hlines(
    [my_rampc.options.xBound[0][0], my_rampc.options.xBound[1][0]],
    k_x[0], k_x[-1], color=BOUND_COLOR, ls='--', label=r'$x_1$ bound', lw=LINE_WIDTH,
)
ax[0].set_ylabel(r"$y$")
ax[0].legend(loc='best')

ax[1].plot(k_u, u_traj[:, 0], color="#E28000D2", label=r'$u_k$', lw=LINE_WIDTH)
ax[1].hlines(
    [my_rampc.options.uBound[0][0], my_rampc.options.uBound[1][0]],
    k_x[0], k_x[-1], color=BOUND_COLOR, ls='--', label=r'$u$ bound', lw=LINE_WIDTH,
)
ax[1].set_xlabel(r"$\mathrm{Step}\ k$")
ax[1].set_ylabel(r"$u$")
ax[1].legend(loc='best')

ax[1].set_xlim(k_x[0], k_x[-1])  # sharex: no margin beyond the simulation window
plt.show()

## Plot the projection on axis of parameters evolution over time (one subplot per parameter)
## Plot should show th_hat, up and lower bound on the parameters (from the polytope) and the true parameter value (from the validation point -> param_expected)

# [DEPRECATED] (data were logged before solve(), with None at k=0 and aliased th_hat; wrong vertex decoding)
# remove None values from the list of parameter estimates and convert to numpy array stacking horizontally
# param_est = np.array([p[1:,0] for p in param_est if p is not None])
# bound_vertices = np.array([p.reshape(-1,param_center.shape[0]+1)[:,1:] for p in bound_vertices if p is not None])
# [v3 FIX] param_est: (T, q) estimate at step k = 0..T-1; bound_vertices: list of (nv, q) (already decoded in the loop)
param_est = np.array(param_est)
up_bound = np.array([v.max(axis=0) for v in bound_vertices])    # (T, q) projection of the set on each parameter axis
low_bound = np.array([v.min(axis=0) for v in bound_vertices])   # (T, q)
# [DEPRECATED] up_bound = np.max(bound_vertices, 1); low_bound = np.min(bound_vertices, 1)

# [DEPRECATED] param_est.shape[0] is the number of TIME STEPS (~200 subplots -> the script hung here)
# fig, ax = plt.subplots(param_est.shape[0], 1, figsize=FIGSIZE, sharex=True, layout="constrained")
# [v3 FIX] one subplot per parameter (param_est.shape[1])
fig, ax = plt.subplots(param_est.shape[1], 1, figsize=FIGSIZE, sharex=True, layout="constrained")

# [v3 FIX] all the series are now defined at k = 0..T-1 -> same time axis k_u (was k_x[2:] / k_x[1:])
for i in range(param_est.shape[1]):
    ax[i].plot(k_u, param_est[:, i], color="#00FF00EA", label=r'$\hat{\theta}_{%d,k}$' % (i+1), marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)
    # [DEPRECATED] ax[i].plot(k_x[2:], param_est.T[i,:], ...)

    ax[i].set_ylabel(r"$\hat{\theta}_{%d}$" % (i+1))
    ax[i].legend(loc='best')

    ax[i].fill_between(k_u, low_bound[:,i], up_bound[:,i],
                        color="#006EFFEA", alpha=0.2,
                        label=r'$\theta_{%d}$ bounds' % (i+1)
                        )
    # [DEPRECATED] ax[i].fill_between(k_x[1:], low_bound[:,i], up_bound[:,i], ...)

    ax[i].plot(k_u, up_bound[:,i], color="#006EFFEA", ls='--', label=r'$\theta_{%d}$ upper bound' % (i+1), lw=LINE_WIDTH)
    ax[i].hlines(param_expected[i], k_x[0], k_x[-1], color="#FF0000EA", ls='--', label=r'$\theta_{%d}$ true value' % (i+1), lw=LINE_WIDTH)
    ax[i].plot(k_u, low_bound[:,i], color="#006EFFEA", ls='--', label=r'$\theta_{%d}$ lower bound' % (i+1), lw=LINE_WIDTH)
    # [DEPRECATED] ax[i].plot(k_x[1:], up_bound[:,i], ...) / ax[i].plot(k_x[1:], low_bound[:,i], ...)

    # ax[i].legend(loc='best')

plt.show()
# Save plot data in csv format
# Shared schema (duffing_common): t, r, x1, x2, u, xa1, xa2. Covers both figures:
# phase plane (x1, x2, xa1, xa2) and time plot (t, x1, r, u). Rows are truncated
# to len(t) = T, i.e. the final state x_T is not stored.
RESULTS_DIR = Path(CURRENT_DIR) / "results" / TRACK_SIGNAL
# [DEPRECATED] same file name as mpc_example.py -> the two scripts overwrite each other's results
# csv_path = RESULTS_DIR / "mpc_example.csv"
# [FIX] own file name (as already done in rmpc_example.py)
csv_path = RESULTS_DIR / "rampc_example.csv"
t_axis = np.arange(T) * Ts  # [s]; step k = t / Ts
# Per-step parameter estimate and projected set bounds as extra columns
# (theta_hat_i, theta_up_i, theta_low_i) -- enough for run_comparison.py to rebuild the
# parameter plot from the CSV alone; the true value theta_true is constant for the
# whole run, so it goes in the _bounds.npz sidecar instead.
theta_extra = {}
for i in range(param_est.shape[1]):
    theta_extra[f"theta_hat_{i + 1}"] = param_est[:, i]
    theta_extra[f"theta_up_{i + 1}"] = up_bound[:, i]
    theta_extra[f"theta_low_{i + 1}"] = low_bound[:, i]
dc.save_results_csv(csv_path, t_axis, r_traj, x_traj, u_traj, xa_traj, extra=theta_extra)
# Constraint levels drawn as horizontal lines: sidecar rampc_example_bounds.npz
dc.save_bounds_npz(csv_path, my_rampc.options.xBound, my_rampc.options.uBound, theta_true=param_expected)
print("Saved plot data to:", csv_path)


