import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import sys
from pathlib import Path

from pycvxset import Polytope
from math import pi

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

from pycvxset.common import constants

constants.DEFAULT_LP_SOLVER_STR = "CLARABEL"
constants.DEFAULT_SOCP_SOLVER_STR = "MOSEK"
constants.DEFAULT_SDP_SOLVER_STR = "MOSEK"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

try:
    from rampc import CRAMPC
except ModuleNotFoundError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src import CRAMPC

A = np.block([[[np.eye(3)]],[[np.zeros((3,3))]],[[np.zeros((3,3))]]])
A = A.transpose(1,2,0).copy()

B = np.block([[[np.zeros((2,2))],[0, 1]],[[1, 0], [np.zeros((2,2))]],[[np.zeros((1,2))],[1, 0], [np.zeros((1,2))]]])/30
B = B.transpose(1,2,0).copy()


C = np.block([[[np.eye(3)]],[[np.zeros((3,3))]],[[np.zeros((3,3))]]])
C = C.transpose(1,2,0).copy()

Q, R = 100* np.diag(np.array([1, 1, 0.8])), 0.5*np.eye(2)
Theta = Polytope(A = np.block([[np.eye(2)], [-np.eye(2)]]),
                 b=np.ones((4,1)))

# Define the MPC parameters
N = 10  # Prediction horizon

W = Polytope(
                A=np.block([[np.eye(3)], [-np.eye(3)]]), b=np.array([0.01, 0.01, 0.01*np.pi/180, 0.01, 0.01, 0.01*np.pi/180])
            )

E = Polytope(
                A=np.block([[np.eye(3)], [-np.eye(3)]]), b=np.array([0.01, 0.01, 0.01*np.pi/180, 0.01, 0.01, 0.01*np.pi/180])
            )

npfile = np.load(SCRIPT_DIR / 'kin_offline_mat.npz')
K = npfile['arr_0']
P = npfile['arr_1']

options = {
    "solver": "osqp",
    "verbose": True,
    "svd": False,
    'xBound':(-np.array([10, 5, 1e4]), np.array([10, 5, 1e4])),
    'uBound':(-np.array([0.46, 1.90]), np.array([0.46, 1.90])),
    "name": "turtle_controller",
    'W': W,
    'E': E,
    'K': K,
    'lam': 1,
    'theta': Theta,
    'par_filter': 'kf',
    'lpv_flag': True,
    'ref': 'ref',
    'max_delta_th': 0.001
}

# Create the MPC controller
my_mpc = CRAMPC({"A": A, "B": B, "C": C}, Q, R, N, options)
# Initialize the controller
my_mpc.initialize(
    mode="LQR",
    P = P
) 

circle_angle = np.arange(0.0, 2*np.pi, 2*np.pi/(30*20))

circle_pose = np.array([np.cos(circle_angle)-1, np.sin(circle_angle), circle_angle-np.pi/2]).T

# Simulate the system
x0 = np.array([0, 0, 0.1])  # Initial state
T = 500  # Simulation time
x_traj = [ca.DM(x0)]
u_traj = []
polytopes = []
# ax,_,_ = Theta.plot(patch_args={'facecolor': 'red'})
for t in range(T):
    polytopes.append(my_mpc.initial_tube())
    w = np.random.rand(3,1)*np.array([[0.01], [0.01], [0.01*np.pi/180]])
    my_mpc.solve(x_traj[-1] + w, x_traj[-1], r = list(circle_pose[t+5].flatten()))
    u_traj.append(my_mpc.u_star)
    theta_k = x_traj[-1][-1].toarray()
    x_next = np.einsum('ij,jkl->kl',
                       np.array([[1, np.cos(theta_k[0,0]), np.sin(theta_k[0,0])]]),
                       A.transpose(2, 0, 1) @ x_traj[-1].toarray() 
                       + B.transpose(2, 0, 1) @ my_mpc.u_star.toarray()
                       ) 
    x_traj.append(ca.DM(x_next))
    if t % 100 == 0:
        ax,_,_ = Polytope(V=my_mpc.theta_vertices[:,0].reshape(3,-1).T).projection(0).plot(patch_args={'facecolor': 'red', 'alpha': 0.1})
        for i in range(my_mpc.theta_vertices.shape[1]-1):
            ax,_,_ = Polytope(V=my_mpc.theta_vertices[:,i+1].reshape(3,-1).T).projection(0).plot(ax=ax, patch_args={'facecolor': 'green', 'alpha': 0.01})
        ax.scatter(my_mpc.th_hat[1], my_mpc.th_hat[2], marker='o', color='red', label='Estimated parameters')
        ax.scatter(np.cos(theta_k[0,0]), np.sin(theta_k[0,0]), marker='x', color='green', label='Real parameters')

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
plt.plot(u_traj[:, 0])
plt.plot(u_traj[:, 1])
plt.xlabel("Time step")
plt.ylabel("Value")
plt.title("MPC State and Input Trajectory")
plt.legend([r"$X$", r"$Y$", r"$\theta$", r"$v_x$", r"$\omega$"])
plt.show()


plt.figure()
plt.plot(circle_pose[:, 0], circle_pose[:, 1], label='Reference trajectory')
plt.plot(x_traj[:, 0], x_traj[:, 1], label='Actual trajectory')
plt.xlabel("X position") 
plt.ylabel("Y position")
plt.title("Turtlebot Trajectory Tracking")
plt.legend()
plt.axis('equal')
plt.show()





