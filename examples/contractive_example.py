"""
Robust invariant set computation
================================
Three test cases for InvariantSet:

  Case 1 – Autonomous system  (A_cl pre-computed externally, B and K not passed)
  Case 2 – Controlled system  (A and B passed, K computed externally but NOT passed)
  Case 3 – Autonomous via K   (A, B and K passed; class computes A_cl = A + B K)

System: 2-state, 1-input double integrator (discretised, Ts = 0.1 s)
  x+ = A x + B u
  A = [[1, 0.1], [0, 1]],  B = [[0.005], [0.1]]

Constraints:
  abs(x1) <= 5,  abs(x2) <= 2,  abs(u) <= 1

Stabilising gain (LQR-like, pre-computed for simplicity):
  K = [-1.73, -2.05]   (places closed-loop eigenvalues at ~0.85 ± 0.1j)
"""

import numpy as np
from pycvxset import Polytope

import matplotlib.pyplot as plt

try:
    from polypympc.tools.invariance_tools import InvariantSet, GainSynthesis
except ModuleNotFoundError:
    from tools.invariance_tools import InvariantSet, GainSynthesis

# -----------------------------------------------------------------------
# System matrices (deterministic, no uncertainty)
# -----------------------------------------------------------------------
# Ts = 0.1
# A = np.array([[1.0, Ts],
#               [0.0, 1.0]])

# B = np.array([[0.5 * Ts**2],
#               [Ts]])

A1 = np.array([[-0.7, 0.15], [-0.35, -0.6]])
A2 = np.array([[-0.75, -0.1], [0.15, -0.65]])
A3 = np.array([[-0.65, -0.35], [-0.1, -0.55]])

A = np.stack((A1,A2,A3),2)

B1 = np.array([[0.1], [1]])
B2 = np.array([[0.2], [1.4]])
B3 = np.array([[0.3], [0.6]])

B = np.stack((B1,B2,B3),2)

n, m = A.shape[0], B.shape[1]

# Pre-computed stabilising gain  (eigenvalues of A+BK ≈ 0.84 ± 0.12j)
# K = np.array([[-7.0, -4.5]])
# Acl = A + B @ K

K = np.array([[0.19, 0.34]])

Acl = A + np.einsum('ijk,jl->ilk',B,K)

# eigs = np.linalg.eigvals(Acl)
# print(f"Closed-loop eigenvalues: {eigs}")
# print(f"Spectral radius: {np.max(np.abs(eigs)):.4f}\n")
# assert np.max(np.abs(eigs)) < 1.0, "K is not stabilising!"

# -----------------------------------------------------------------------
# Constraint polytope Z_bnd on [x1, x2, u]
#   |x1| <= 5,  |x2| <= 2,  |u| <= 1
# -----------------------------------------------------------------------
x1_lim, x2_lim, u_lim = 5.0, 2.0, 1.0

A_bnd = np.array([
    [ 0,  0.1,  0],   #  x1 <= 5
    [ 0, -0.1,  0],   # -x1 <= 5
    [ 0,  0,  0.2],   #  u  <= 1
    [ 0,  0, -0.2],   # -u  <= 1
])
b_bnd = np.array([ 1, 1, 1, 1])

Z_bnd = Polytope(A=A_bnd, b=b_bnd)

# ======================================================================
# CASE 1: Autonomous system — Acl computed externally, only A_cl passed
#         B and K are NOT passed to InvariantSet
# ======================================================================
print("=" * 60)
print("CASE 1 – Autonomous (A_cl pre-computed, B and K not passed)")
print("=" * 60)

# The state constraint set is the projection of Z_bnd onto x
# (the class needs Z_bnd on [x; u], but with m=0 it coincides with X)

X_bnd = Polytope(A = Z_bnd.A/Z_bnd.b[:,np.newaxis] @ np.vstack((np.eye(2), K)),
                 b = np.ones((Z_bnd.A.shape[0], 1)))

inv1 = InvariantSet(
    Z_bnd=X_bnd,
    A=Acl,          # <-- closed-loop matrix passed as "autonomous" A
    B=None,         # autonomous: no input
    K=None,
    contractive_factor=0.999
)
print("Computing invariant set (Case 1)...")

result1 = inv1.compute_invariant_set()

# ======================================================================
# CASE 2: Controlled system — A and B passed, K computed externally
#         but NOT passed to the class (K = None → class uses K = 0)
#
#  NOTE: with K=None the class uses K=0, so the invariant set is for
#        the *open-loop* system; to match Case 1 the user would need to
#        pass K explicitly.  This case demonstrates the API behaviour.
# ======================================================================
print("\n" + "=" * 60)
print("CASE 2 – Controlled System")
print("=" * 60)

# We keep the full Z_bnd on [x; u] so the input constraint is active
# and the set accounts for the u channel.
inv2 = InvariantSet(
    Z_bnd=Z_bnd,
    A=A,
    B=B,
    K=None,           # class treats K = 0 (open-loop)
    contractive_factor=1.0
)
print("Computing invariant set (Case 2)...")

result2 = inv2.compute_invariant_set()

# ======================================================================
# CASE 3: Autonomous via K — A, B and K passed; class computes A_cl
# ======================================================================
print("\n" + "=" * 60)
print("CASE 3 – Autonomous system obtained via K")
print("=" * 60)

inv3 = InvariantSet(
    Z_bnd=Z_bnd,
    A=A,
    B=B,
    K=K,              # class computes A_cl = A + B @ K internally
    contractive_factor=1.0
)
print("Computing invariant set (Case 3)...")

result3 = inv3.compute_invariant_set()

# ======================================================================
# Comparison: Cases 1 and 3 should give the same set
# ======================================================================
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Case 1 == Case 3? {result1 == result3}")

# Combined plot
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
titles = ["Case 1\nAutonomous (A_cl ext.)",
          "Case 2\nControlled (K=0 in class)",
          "Case 3\nAutonomous via K"]
results = [result1, result2, result3]
colors  = ["steelblue", "tomato", "seagreen"]

for ax, poly, title, color in zip(axes, results, titles, colors):

    ax, _, _ = poly.plot(ax,{'facecolor': color,})

    # ax.set_xlim(-x1_lim * 1.1, x1_lim * 1.1)
    # ax.set_ylim(-x2_lim * 1.1, x2_lim * 1.1)
    # ax.set_xlabel("$x_1$")
    # ax.set_ylabel("$x_2$")
    ax.set_title(title)
    # ax.axhline(0, color='k', lw=0.5)
    # ax.axvline(0, color='k', lw=0.5)
    ax.grid(True, linestyle='--', alpha=0.4)

fig.suptitle("Maximal Positively Invariant Sets", fontsize=12)
fig.tight_layout()
plt.show()