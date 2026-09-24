"""
Nominal MPC on the two-mass mass-spring-damper plant
=====================================================
Closed-loop tracking of both masses' positions (y = [x1, x2]) with a plain
(non-robust) MPC built on the nominal model (A_full[:,:,0], B_full[:,:,0])
from ss_data.npz, with the plant driven by the TRUE physical parameters
(p_real in sys_param.npz) and an additive disturbance drawn from the
truncation-mismatch set W_mismatch (uncertainty_sets.npz).
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from os import path
from pycvxset.common import constants
from pathlib import Path
from polypympc import MPC, GainSynthesis


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

LINE_WIDTH = 2.8
MARKER_SIZE = 6
FONT_SIZE = {
    "labels": 24,
    "ticks": 20,
    "legend": 22,
}

# Nominal model (p = 0 slice of the affine A(p), B(p) family): a plain MPC
# is built on this fixed pair, not on the LPV family.
A_full, B_full, C, D, Ts = dc.load_ss(path.join(FULL_PATH, "ss_data.npz"))
A = A_full[:, :, 0]
B = B_full[:, :, 0]
assert A.shape == (4, 4) and B.shape == (4, 2), "ss_data.npz shapes don't match the 2-mass MSD plant"

Q, R = 10 * np.eye(A.shape[0]), 0.01 * np.eye(B.shape[1])  # State and input cost matrices

# True physical plant: the actual (m1, m2, k, c), distinct from the nominal
# model above, used to drive the closed loop (see modeling-uncertainty.ipynb).
p_real, p_basis, p_cheb, n_p = dc.load_point(path.join(FULL_PATH, "sys_param.npz"))
m1, m2, k, c = p_real
print(f"True physical parameters: m1={m1:.4f}, m2={m2:.4f}, k={k:.4f}, c={c:.4f}")
plant = dc.make_msd_plant(m1, m2, k, c, Ts)

N = 10  # Prediction horizon

## Load/Compute offline ingredients for the MPC controller

try:
    cache_path = Path(path.join(FULL_PATH, "mpc_cache.npz"))
    K, P, controller_mode = dc.load_offline_cache(
        cache_path
    )
    print("Loaded offline ingredients from cached.\n")
    print("Controller defined in mode: ", controller_mode)
except (FileNotFoundError, OSError, ValueError, KeyError):
    gain_synt = GainSynthesis(A, B, Q=Q, R=R, mode='volume', contraction_factor=0.95)
    gain_synt.synthesize_controller()
    K = gain_synt.get_gain()
    P = gain_synt.get_terminal_weight()

    np.savez(
        path.join(FULL_PATH, "mpc_cache.npz"),
        K=K,
        P=P,
        type="volume",
    )
    print("Computed offline ingredients and saved to cache.\n")

print("Gain K: ", K)
print("Lyapunov Matrix P: ", P)

# Define the MPC parameters. Bounds match the MATLAB reference (Complete_5ag.m):
# umax=25, xmax=[2.22, 6] (position, velocity) per mass, symmetric across masses.
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": True,
    "xBound": (np.array([-2.22, -6.0, -2.22, -6.0]), np.array([2.22, 6.0, 2.22, 6.0])),
    "uBound": (np.array([-25.0, -25.0]), np.array([25.0, 25.0])),
    "name": "msd_mpc",
    "ref": dc.ref_option(TRACK_MODE),
    "K": K,
    "P": P,
}

# Create the MPC controller
my_mpc = MPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# Initialize the controller
my_mpc.initialize()

# Simulate the system
x0 = np.zeros(4)  # Initial state [x1, v1, x2, v2]
T = 200  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
xa_traj = []
r_traj = []

# Amplitude +/-2.2 matches the MATLAB reference (pw_ref amplitude); the two
# outputs get different periods/phases so x1, x2 track genuinely independent
# references instead of moving in lockstep.
reference = dc.make_reference(Ts, kind=TRACK_SIGNAL, amp=2.2 * GAIN,
                               n_signals=2, period=[5.0, 6.8], phase=[0.4, 0.0])
# W = dc.load_mismatch_set(path.join(FULL_PATH, "uncertainty_sets.npz"))
rng = np.random.default_rng(SEED)


for t in range(T):
    r = dc.reference_arg(reference, t, N, TRACK_MODE)
    # w = dc.make_disturbance(W, rng)
    my_mpc.solve(x_traj[-1], r)
    u_traj.append(my_mpc.u_star)
    xa_traj.append(my_mpc.xa_star[:, -1])
    x_next = plant[1](x_traj[-1], my_mpc.u_star) #+ w
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
        [my_mpc.options.xBound[0][y_state_idx[i]], my_mpc.options.xBound[1][y_state_idx[i]]],
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
        [my_mpc.options.uBound[0][i], my_mpc.options.uBound[1][i]],
        k_x[0], k_x[-1], color=BOUND_COLOR, ls='--', label=r'$u$ bound', lw=LINE_WIDTH,
    )
    ax[i].set_ylabel(f'$u_{i + 1}$')

dc.shared_legend(fig, ax)
ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
ax[-1].set_xlim(k_x[0], k_x[-1])  # sharex: no margin beyond the simulation window
plt.show()


# Save plot data in csv format
# Schema (common.py, _RESULT_COLUMNS): t, r1, r2, x1, v1, x2, v2, u1, u2,
# xa1..xa4. Rows are truncated to len(t) = T, i.e. the final state x_T is not
# stored.
RESULTS_DIR = Path(CURRENT_DIR) / "results" / TRACK_SIGNAL
csv_path = RESULTS_DIR / "mpc_example.csv"
t_axis = np.arange(T) * Ts  # [s]; step k = t / Ts
dc.save_results_csv(csv_path, t_axis, r_traj, x_traj, u_traj, xa_traj)
# Constraint levels drawn as horizontal lines: sidecar mpc_example_bounds.npz
dc.save_bounds_npz(csv_path, my_mpc.options.xBound, my_mpc.options.uBound)
print("Saved plot data to:", csv_path)
