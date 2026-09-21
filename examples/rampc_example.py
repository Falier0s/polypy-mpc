"""
Robust adaptive MPC on an LPV system
====================================

Builds a :class:`~polypympc.RAMPC` for a 4-vertex LPV model and runs the closed
loop with online set-membership updates of the feasible parameter set.

Set ``TRACKING = True`` below to switch the controller from plain
stabilization (regulate to the origin) to setpoint tracking of a small
square-wave reference on ``y = x1``: this is what actually configures the
controller for tracking (``options["ref"] = "ref"``, single-setpoint mode --
see duffing_common.ref_option for the "ref" vs "traj" distinction used
elsewhere in this repo), not just a different reference signal fed to the
same controller.
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from pycvxset import Polytope

from pycvxset.common import constants

try:
    from polypympc import RAMPC
except ModuleNotFoundError:
    from src import RAMPC

# Set CLARABEL as the default solver for all problem types since already installed in pycvxset
# If needed, users can change the default solver for each problem type by changing the corresponding constant in pycvxset.common.constants
# if constants.DEFAULT_LP_SOLVER_STR == 'MOSEK':
constants.DEFAULT_LP_SOLVER_STR = "OSQP"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

# ---------------------------------------------------------------------------
# Stabilization (regulate to the origin) vs setpoint tracking of a small
# square wave on y = x1. Only this flag needs to change to switch modes.
# ---------------------------------------------------------------------------
TRACKING = False

# Define the system dynamics
A0 = np.array([[0.5, 0.2], [-0.1, 0.6]])
dA1 = np.array([[0.042, 0.], [0.072, 0.03]])
dA2 = np.array([[0.0015, 0.019], [0.009, 0.035]])
dA3 = np.array([[0., 0.], [0., 0.]])

B0 = np.array([[0.], [0.5]])
dB1 = np.array([[0.], [0.]])
dB2 = np.array([[0.], [0.]])
dB3 = np.array([[0.04], [0.054]])

A = np.stack([A0, dA1, dA2, dA3], axis=2)
B = np.stack([B0, dB1, dB2, dB3], axis=2)

C = np.array([[1, 0]])
C = np.stack([C, np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))], axis=2)

Q, R = np.eye(2), np.array([[0.1]])
Theta = Polytope(A = np.block([[np.eye(3)], [-np.eye(3)]]), b=1*np.ones(6,))

# Define the MPC parameters
N = 10  # Prediction horizon
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    'xBound':(np.array([-1e4, -0.3]), np.array([1e4, 1e4])),
    'uBound':(np.array([-1e4]),np.array([1])),
    "name": "test_mpc",
    'W': Polytope(A=np.block([[np.eye(2)], [-np.eye(2)]]), b=0.05*np.ones((4,1))),
    'lam': 0.9,
    'theta': Theta,
    'K': np.array([[0.017, -0.41]]),
    'par_filter': 'lms',
    'lpv_flag': False
}
if TRACKING:
    options["ref"] = "ref"  # setpoint tracking (single value per step)

# Create the MPC controller
my_mpc = RAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# Initialize the controller
my_mpc.initialize(
    mode="volume"
)  # mode available options: "volume", "LQR" or "performance"
# my_mpc.P = np.array([[1.467, 0.207], [0.207, 1.731]])  # terminal cost matrix (optional, default is P=Q)
# True system: a fixed combination of the 4 vertex models -- NOT one of the
# vertices itself, so theta_true = [0.8, 0.2, -0.5] (the weights on dA1/dA2/dA3,
# dA3 being the zero matrix makes its own weight irrelevant to A but not to B)
# is an actual interior point of Theta, unlike A0/B0 alone. Used below only as
# the ground-truth line on the parameter-estimation plot.
THETA_TRUE = np.array([0.8, 0.2, -0.5])


def theta_vertex_bounds(col, q):
    """Per-dimension [min, max] of the parameter polytope packed in ``col``
    (homogeneous form [1, theta_1, ..., theta_q], dimension-major -- see
    RAMPC.solve())."""
    pts = np.asarray(col).reshape(q + 1, -1).T  # (n_vertices, q+1)
    pts = pts[:, 1:]                             # drop the homogeneous 1 -> (n_vertices, q)
    return pts.min(axis=0), pts.max(axis=0)


def square_wave(t, amplitude=0.02, period_steps=40):
    """Small-amplitude, zero-centred square wave -- comfortably reachable
    given this system's tight uBound (upper-capped at 1) and the tube-based
    tightening it must share with K, not a value chosen for realism."""
    return amplitude * float(np.sign(np.sin(2.0 * np.pi * t / period_steps) + 1e-12))


def step_reference(t, delta=-0.02, offset=-20):
    """Staircase: increases by ``delta`` at every single simulation step
    (r(t) = delta * t), instead of the square wave's repeated back-and-forth
    transitions or a one-off jump held constant. With this system's tight
    reachable range (see rampc_example.py's history: |r| beyond ~0.02-0.05
    was already infeasible as a plain setpoint), expect the closed loop to
    track it for a while and then stop cleanly once the ramp outgrows what's
    robustly reachable -- see the try/except around solve() below."""
    return delta * t + offset


REFERENCE_FN = step_reference  # square_wave | step_reference


# Simulate the system
# x0=[2,3] is fine for plain stabilization, but tracking mode adds the
# artificial-trajectory/terminal constraints on top of the same tube
# machinery -- with this system's tight uBound (capped at 1) that leaves far
# less room. uBound's upper cap of 1 (vs. an unbounded lower end) makes the
# reachable range for y=x1 asymmetric: the system can be driven up to about
# 0 but not meaningfully above it, so it must START at or below 0 and the
# reference must stay <= 0 too (see step_reference's negative delta above).
x0 = np.array([-20.0, 0.0]) if TRACKING else np.array([2, 3])
T = 1000  # Simulation time
SEED = 0  # process-noise RNG seed, for reproducible runs

# options["W"] declares a 0.05-bounded additive-disturbance box for the
# set-membership/tube machinery, but nothing was actually injecting any
# disturbance into the propagation below -- the closed loop was a noiseless,
# deterministic convergence to the origin, so the regressor Phi (which scales
# with z=[x,u]) shrank towards zero as the state converged, starving the
# parameter estimator of new information after the first few steps regardless
# of how long the simulation ran. Injecting noise consistent with the
# declared W keeps genuinely exciting the estimator throughout.
W_BOUND = 0.0001
rng = np.random.default_rng(SEED)

TUBE_STRIDE = max(T // 20, 1)  # only keep ~20 tube snapshots for the phase-plane plot

x_traj = [ca.DM(x0)]
u_traj = []
r_traj = []
polytopes = []
theta_vertices_hist = []  # per step: my_mpc.theta_vertices (homogeneous, dilation-indexed)
th_hat_hist = []          # per step: point estimate theta_hat (q entries, no homogeneous 1)
for t in range(T):
    if t % TUBE_STRIDE == 0:
        polytopes.append(my_mpc.initial_tube())
    r_val = REFERENCE_FN(t) if TRACKING else 0.0
    try:
        if TRACKING:
            my_mpc.solve(x_traj[-1], x_traj[-1][0], r=[r_val])
        else:
            my_mpc.solve(x_traj[-1], x_traj[-1][0])
    except RuntimeError as exc:
        print(f"[rampc] stopping at step {t}: solve() failed -- {exc}")
        break
    r_traj.append(r_val)
    u_traj.append(my_mpc.u_star)
    theta_vertices_hist.append(np.array(my_mpc.theta_vertices, copy=True))
    th_hat_hist.append(np.asarray(my_mpc.th_hat, dtype=float).flatten()[1:])
    w = rng.uniform(-W_BOUND, W_BOUND, size=(2, 1))
    x_next = np.einsum('ij,jkl->kl',np.array([[1, 0.8, 0.2, -0.5]]),
                       A.transpose(2, 0, 1) @ x_traj[-1].toarray()
                       + B.transpose(2, 0, 1) @ my_mpc.u_star.toarray()) + w
    x_traj.append(ca.DM(x_next))

T_completed = len(u_traj)  # < T if the loop above stopped early
if T_completed < T:
    print(f"[rampc] completed {T_completed}/{T} steps before stopping")

x_traj = np.array(x_traj)
u_traj = np.array(u_traj)
r_traj = np.array(r_traj)
th_hat_hist = np.array(th_hat_hist)  # (T_completed, q)
t_axis = np.arange(T_completed)

# Bounds of the *current* (dilation step l=0) shrinking parameter set, per step.
q = my_mpc.q
theta_lb = np.zeros((T_completed, q))
theta_ub = np.zeros((T_completed, q))
for k, tv in enumerate(theta_vertices_hist):
    theta_lb[k], theta_ub[k] = theta_vertex_bounds(tv[:, 0], q)

# Plot the results
ax,_,_= polytopes[0].plot(patch_args={"color": "blue", "alpha": 0.5})
num_pol = 1
for poly in polytopes[1:]:
    alpha = 1-(num_pol / len(polytopes))
    try:
        poly.plot(ax, {"color": "blue", "alpha": alpha})
    except Exception as exc:  # noqa: BLE001 - a degenerate tube can trip pycvxset's cdd-based V-rep
        print(f"[rampc] skipping one tube overlay ({type(exc).__name__}: {exc})")
    num_pol += 1

ax.plot(x_traj[:, 0], x_traj[:, 1], "ro-", label="State trajectory")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
ax.set_title("MPC State Trajectory")
ax.legend()
plt.show()

plt.figure()
plt.plot(t_axis, x_traj[:-1, 0, 0], label="x1")
plt.plot(t_axis, x_traj[:-1, 1, 0], label="x2")
plt.step(t_axis, u_traj[:, 0, 0], where="post", label="Input")
if TRACKING:
    plt.plot(t_axis, r_traj, "k--", label=f"reference ({REFERENCE_FN.__name__})")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title(f"RAMPC tracking a {REFERENCE_FN.__name__} reference on y = x1" if TRACKING
          else "RAMPC state and input trajectory (stabilization)")
plt.legend()
plt.show()

# Parameter-estimate convergence: does theta_hat -> THETA_TRUE, and does the
# estimated set (shrinking via set-membership) actually contain it?
dim_names = [rf"$\theta_{d + 1}$" for d in range(q)]
fig_p, axs_p = plt.subplots(q, 1, sharex=True, figsize=(8, 8), layout="constrained")
for d in range(q):
    axp = axs_p[d]
    axp.fill_between(t_axis, theta_lb[:, d], theta_ub[:, d],
                      color="C0", alpha=0.25, label="estimated set (l=0)")
    axp.axhline(THETA_TRUE[d], color="k", linestyle="-", linewidth=1.5,
                label="true theta" if d == 0 else None)
    axp.plot(t_axis, th_hat_hist[:, d], color="red", linestyle=":", marker=".",
             markersize=4, label="point estimate" if d == 0 else None)
    axp.set_ylabel(dim_names[d])
axs_p[0].legend(loc="best", fontsize=9)
axs_p[-1].set_xlabel("Time step")
fig_p.suptitle("RAMPC parameter estimation (true theta = [0.8, 0.2, -0.5])")
plt.show()
