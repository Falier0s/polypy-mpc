import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

from pycvxset.common import constants

from cMPC import CMPC

# Set CLARABEL as the default solver for all problem types since already installed in pycvxset
# If needed, users can change the default solver for each problem type by changing the corresponding constant in pycvxset.common.constants
if constants.DEFAULT_LP_SOLVER_STR != "CLARABEL":
    constants.DEFAULT_LP_SOLVER_STR = "CLARABEL"
    constants.DEFAULT_SOCP_SOLVER_STR = "CLARABEL"
    constants.DEFAULT_SDP_SOLVER_STR = "CLARABEL"

# Define the system dynamics
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
C = np.array([[1.0, 0.0]])
Q, R = np.eye(2), np.array([[0.1]])

# Define the MPC parameters
N = 10  # Prediction horizon
options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    "xBound": (np.array([-8, -8]), np.array([11, 8])),
    "uBound": (np.array([-1]), np.array([1])),
    "name": "test_mpc",
    'ref': 'traj'
}

# Create the MPC controller
my_mpc = CMPC({"A": A, "B": B, "C": C}, Q, R, N, options)

# Initialize the controller
my_mpc.initialize(
    mode="volume"
)  # mode available options: "volume", "LQR" or "performance"

# Simulate the system
x0 = np.array([10.0, 0.0])  # Initial state
T = 20  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
polytopes = []
for t in range(T):
    polytopes.append(my_mpc.poly_x_aug)
    my_mpc.solve(x_traj[-1],[0]*11)
    u_traj.append(my_mpc.u_star)
    x_next = A @ x_traj[-1] + B @ my_mpc.u_star
    x_traj.append(x_next)
x_traj = np.array(x_traj)
u_traj = np.array(u_traj)
print(polytopes[0])

# Plot the results
ax,_,_= polytopes[0].projection([0,1,4]).plot()
num_pol = 1
for poly in polytopes[1:]:
    alpha = 1-(num_pol / len(polytopes))
    poly.projection([0,1,4]).plot(ax, {"color": "blue", "alpha": alpha})
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
