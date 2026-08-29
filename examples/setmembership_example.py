import sys
from pathlib import Path

import numpy as np
from pycvxset import Polytope
from matplotlib import pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from rampc.tools import SetUpdater
except ModuleNotFoundError:
    from src.tools import SetUpdater
 
# Nominal System Info

a_nom, b_nom = 0.8, 0.5
c_nom        = 1.0
 
A_B = np.array([
    [[a_nom, b_nom]],
    [[1.0,   0.0  ]],
    [[0.0,   1.0  ]],
])  
 
C = np.array([
    [[c_nom]],
])  
 
# Initial Sets

delta_ab = 0.8
 
A_th = np.vstack([np.eye(2), -np.eye(2)])
b_th = delta_ab * np.ones(4)
Theta_init = Polytope(A=A_th, b=b_th)
 
Theta_c_init = Polytope()   # vuoto — nessun parametro incerto su C
 
A_w = np.array([[1.0], [-1.0]])
b_w = 0.05 * np.ones(2)
W = Polytope(A=A_w, b=b_w)
 
A_e = np.array([[1.0], [-1.0]])
b_e = 0.02 * np.ones(2)
E = Polytope(A=A_e, b=b_e)
 
# Initialize the SetUpdater Object

updater = SetUpdater(
    A_B=A_B, C=C,
    Theta=Theta_init, Theta_c=Theta_c_init,
    W=W, E=E,
    N=1, tol=0.0,
)
 

Theta_A_norm = (Theta_init.A / Theta_init.b[:, np.newaxis]).copy()   

Theta_b_curr   = np.ones(Theta_init.b.shape[0], dtype=float).copy()
Theta_c_b_curr = np.array([], dtype=float)   
 
# True System Parameters

a_true, b_true = 0.77, 0.45  
 
np.random.seed(42)
n_steps  = 20
w_max    = 0.05
e_max    = 0.02
max_delta_th = 0.03 
 
x = np.array([[0.5]])
u = np.array([[0.0]])
z = np.vstack([x, u]).copy()  
 
results = []
 
for k in range(n_steps):
 
    w = np.random.uniform(-w_max, w_max, (1, 1))
    e = np.random.uniform(-e_max, e_max, (1, 1))
 
    x_next = a_true * x + b_true * u + w
    y      = c_nom  * x_next + e
 
    # Unfalsified Parameter Set Construction for Visualization Purposes
    D_k_1  = (A_B[1:] @ z).squeeze(axis=1).T    # (n_state, n_theta)
    d_k    = (A_B[0] @ z - x_next)              # (n_state, 1)
    Delta_k = Polytope(
        A = -W.A @ D_k_1,
        b = (W.b[:, np.newaxis] + W.A @ d_k).squeeze()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Step {k}")
 
    Theta_curr = Polytope(A=Theta_A_norm, b=Theta_b_curr)
    ax1, _, _ = Theta_curr.plot(ax=ax1)
    ax1.set_title("Theta corrente")
 
    try:
        ax2, _, _ = Theta_curr.intersection(Delta_k).plot(ax=ax2)
    except Exception:
        ax2, _, _ = Theta_curr.plot(ax=ax2)
    ax2.set_title("Theta ∩ Delta_k")
 
    # Set Update

    NewTheta_b, NewTheta_c_b, th_vert, _ = updater.update(
        Theta        = Theta_b_curr.copy(),
        Theta_c      = Theta_c_b_curr.copy(),
        z_prev       = z.copy(),
        x_actual     = x_next.copy(),
        y_actual     = y.copy(),
        max_delta_th = max_delta_th,
    )
 
    for v in range(th_vert.shape[2]):
        ax1, _, _ = Polytope(V=th_vert[:, :, v]).plot(ax=ax1, patch_args={'facecolor':'blue', 'alpha':(0.2*v)})
    ax1.plot(a_true-a_nom, b_true-b_nom, 'ro', label='true')
    plt.tight_layout()
    plt.show()
    if NewTheta_b is not None:
        Theta_b_curr = NewTheta_b.copy()
 
    vol = Polytope(A=Theta_A_norm, b=Theta_b_curr).volume()
    results.append({'step': k, 'b_th': Theta_b_curr.copy(), 'vol': vol})
 
    print(
        f"Step {k:2d} | b: {Theta_b_curr.round(4)} | Vol≈ {vol:.5f}"
    )
 

    x = x_next.copy()   
    u = np.array([[np.sin(0.2 * k) * 0.8]])
    z = np.vstack([x, u]).copy()