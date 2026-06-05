import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from pycvxset import Polytope

from pycvxset.common import constants

from cRAMPC import CRAMPC

# Set CLARABEL as the default solver for all problem types since already installed in pycvxset
# If needed, users can change the default solver for each problem type by changing the corresponding constant in pycvxset.common.constants
# if constants.DEFAULT_LP_SOLVER_STR == 'MOSEK':
constants.DEFAULT_LP_SOLVER_STR = "OSQP"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

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
    'par_filter': 'kf',
    'lpv_flag': False
    # 'ref': 'ref'
}

# Create the MPC controller
my_mpc = CRAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# Initialize the controller
my_mpc.initialize(
    mode="volume"
)  # mode available options: "volume", "LQR" or "performance"
# my_mpc.P = np.array([[1.467, 0.207], [0.207, 1.731]])  # terminal cost matrix (optional, default is P=Q)
# Simulate the system
x0 = np.array([2, 3])  # Initial state
T = 100  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
polytopes = []
for t in range(T):
    polytopes.append(my_mpc.initial_tube())
    my_mpc.solve(x_traj[-1],x_traj[-1][0])
    u_traj.append(my_mpc.u_star)
    x_next = np.einsum('ij,jkl->kl',np.array([[1, 0.8, 0.2, -0.5]]),
                       A.transpose(2, 0, 1) @ x_traj[-1].toarray() 
                       + B.transpose(2, 0, 1) @ my_mpc.u_star.toarray())
    x_traj.append(ca.DM(x_next))
x_traj = np.array(x_traj)
u_traj = np.array(u_traj)

# Plot the results
ax,_,_= polytopes[0].plot(patch_args={"color": "blue", "alpha": 0.5})
num_pol = 1
for poly in polytopes[1:]:
    alpha = 1-(num_pol / len(polytopes))
    poly.plot(ax, {"color": "blue", "alpha": alpha})
    num_pol += 1

ax.plot(x_traj[:, 0], x_traj[:, 1], "ro-", label="State trajectory")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
ax.set_title("MPC State Trajectory")
ax.legend()
plt.show()

plt.figure()
plt.plot(x_traj[:, 0])
plt.plot(x_traj[:, 1])
plt.plot(u_traj[:, 0])
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("MPC State and Input Trajectory")
plt.legend(["x1", "x2", "Input"])
plt.show()
