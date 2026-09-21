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

MODEL_DIR = "model_data"

TRACK_MODE = "trajectory" # "setpoint"|"trajectory"
TRACK_SIGNAL = "square" # "square"|"harmonic"
FULL_PATH = path.join(CURRENT_DIR, MODEL_DIR)
DATASET = "inner" # "inner"|"edge"
SEED = 0 # seed of the disturbance generator

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
A[0,1,:] = -A[0,1,:]
A[1,0,:] = -A[1,0,:]
B[1,0,:] = -B[1,0,:] # Flip the sign of the second column of B to match the duffing dynamics

C = dc.lift_output_matrix(C, A.shape[2])

Q, R = 10*np.eye(A.shape[0]), 0.01*np.eye(B.shape[1])  # State and input cost matrices

# Real duffing oscillator dynamics

## Extract the validation points for the duffing oscillator
## duff_param = [-delta, - alpha, -beta]

duff_param, param_expected, param_center = dc.load_validation_point(path.join(FULL_PATH, "sys_param.npz"), DATASET)
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
    gain_synt = GainSynthesis(a_v, b_v, Q=Q, R=R, mode='LQR', contraction_factor=0.99)
    gain_synt.synthesize_controller()
    K = gain_synt.get_gain()
    P = gain_synt.get_terminal_weight()

    np.savez(
        path.join(FULL_PATH, "rampc_cache.npz"),
        K=K,
        P=P,
        type="lqr",
    )
    print("Computed offline ingredients and saved to cache.\n")

print("Gain K: ", K)
print("Lyapunov Matrix P: ", P)

# Define the RAMPC parameters
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    "xBound": (np.array([-0.3, -1.0e4]), np.array([0.3, 1.0e4])),
    "uBound": (np.array([-5]), np.array([5])),
    "name": "duffing_rampc",
    "ref": dc.ref_option(TRACK_MODE),
    "K": K,
    "P": P,
    "W": Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=0.005*np.ones((4,1))),
    "lam": 0.998,
    "th_hat_0": param_center,
    "theta": param_set,
    "par_filter": "rls",
    'lpv_flag': False
}

W = Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=0.005*np.ones((4,1)))

# Create the RAMPC controller
my_rampc = RAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# Initialize the controller
my_rampc.initialize()
# Simulate the system
x0 = np.array([0.0, 0.0])  # Initial state
T = 200  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
xa_traj = []
r_traj = []
rng = np.random.default_rng(SEED)



for t in range(T):
    r = dc.reference_arg(dc.make_reference(Ts, kind=TRACK_SIGNAL,amp=0.3), t, N, TRACK_MODE)
    w = dc.make_disturbance(W, rng)
    my_rampc.solve(x_traj[-1], x_traj[-1][0], r)
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


# Save plot data in csv format
# Shared schema (duffing_common): t, r, x1, x2, u, xa1, xa2. Covers both figures:
# phase plane (x1, x2, xa1, xa2) and time plot (t, x1, r, u). Rows are truncated
# to len(t) = T, i.e. the final state x_T is not stored.
RESULTS_DIR = Path(CURRENT_DIR) / "results"
csv_path = RESULTS_DIR / "mpc_example.csv"
t_axis = np.arange(T) * Ts  # [s]; step k = t / Ts
dc.save_results_csv(csv_path, t_axis, r_traj, x_traj, u_traj, xa_traj)
# Constraint levels drawn as horizontal lines: sidecar mpc_example_bounds.npz
dc.save_bounds_npz(csv_path, my_rampc.options.xBound, my_rampc.options.uBound)
print("Saved plot data to:", csv_path)
