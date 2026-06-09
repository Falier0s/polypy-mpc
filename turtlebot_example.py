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
file1 = np.load('system_matrices.npz')
file2 = np.load('offline_matrices.npz')
A, B = file1['arr_0'], file1['arr_1']

max_p, min_p = file1['arr_2'][:,1], file1['arr_2'][:,0]

K, P = file2['arr_0'], file2['arr_1']
C = np.block([[np.eye(2), np.zeros((2,2))]])
C = np.stack([C, np.zeros((2, 4)), np.zeros((2, 4)), np.zeros((2, 4))], axis=2)

Q, R = np.eye(4), np.eye(2)
Theta = Polytope(A = np.block([[np.eye(3)], [-np.eye(3)]]),
                 b=np.block([np.array(max_p), -np.array(min_p)]))

# Define the MPC parameters
N = 3  # Prediction horizon

W = Polytope(
                A=np.block([[np.eye(4)], [-np.eye(4)]]), b=0.001 * np.ones((8, 1))
            )

E = Polytope(
                A=np.vstack((np.eye(2), -np.eye(2))),
                b=np.concatenate((np.ones(2) * 0.01, np.ones(2) * 0.01)),
            )


options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    'xBound':(-np.array([1.9/30, 0.46, 0.1, 1.90]), np.array([1.9/30, 0.46, 0.1, 1.90])),
    'uBound':(-np.array([0.46, 1.90]), np.array([0.46, 1.90])),
    "name": "turtle_controller",
    'W': W,
    'E': E,
    'lam': 0.999,
    'theta': Theta,
    'K': K,
    'par_filter': 'kf',
    'lpv_flag': False
    # 'ref': 'ref'
}

# Create the MPC controller
my_mpc = CRAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)
# Initialize the controller
my_mpc.initialize(
    mode="LQR"
)  
my_mpc.P = P

# Simulate the system
x0 = np.array([0.1, 0.5, 0, 0])  # Initial state
T = 100  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
polytopes = []
for t in range(T):
    polytopes.append(my_mpc.initial_tube())
    my_mpc.solve(x_traj[-1], x_traj[-1][:2])
    u_traj.append(my_mpc.u_star)
    x_next = np.einsum('ij,jkl->kl',np.array([[1, 0.1, 0.2, -0.2]]),
                       A.transpose(2, 0, 1) @ x_traj[-1].toarray() 
                       + B.transpose(2, 0, 1) @ my_mpc.u_star.toarray())
    x_traj.append(ca.DM(x_next))
x_traj = np.array(x_traj)
u_traj = np.array(u_traj)

# Plot the results
# ax,_,_= polytopes[0].plot(patch_args={"color": "blue", "alpha": 0.5})
# num_pol = 1
# for poly in polytopes[1:]:
#     alpha = 1-(num_pol / len(polytopes))
#     poly.plot(ax, {"color": "blue", "alpha": alpha})
#     num_pol += 1

# plt.plot(x_traj[:, 0], x_traj[:, 1], "ro-", label="State trajectory")
# plt.xlabel("Position")
# plt.ylabel("Velocity")
# plt.title("MPC State Trajectory")
# plt.legend()
# plt.show()

plt.figure()
plt.plot(x_traj[:, 0])
plt.plot(x_traj[:, 1])
plt.plot(x_traj[:, 2])
plt.plot(x_traj[:, 3])
plt.plot(u_traj[:, 0])
plt.plot(u_traj[:, 1])
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("MPC State and Input Trajectory")
plt.legend(["x1", "x2", "x3", "x4", "u1", "u2"])
plt.show()
