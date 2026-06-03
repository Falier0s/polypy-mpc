"""
example_filter.py
Test Set-Membership Identification along with different filters (Chebyshev, LMS, RLS, KF) for parameter estimation
"""

import numpy as np
from pycvxset import Polytope
from matplotlib import pyplot as plt

from adaptive_tools import SetUpdater
from filter_tool   import Filter

# Nominal System Info

a_nom, b_nom = 0.8, 0.5
c_nom        = 1.0

A_B = np.array([
    [[a_nom, b_nom]],
    [[1.0,   0.0  ]],
    [[0.0,   1.0  ]],
])
C = np.array([[[c_nom]]])

# Initialize Sets

delta_ab = 0.8
Theta_init  = Polytope(A=np.vstack([np.eye(2), -np.eye(2)]),
                        b=delta_ab * np.ones(4))
Theta_c_init = Polytope()

W = Polytope(A=np.array([[1.],[-1.]]), b=0.05*np.ones(2))
E = Polytope(A=np.array([[1.],[-1.]]), b=0.02*np.ones(2))

# ─────────────────────────────────────────────────────────────────────────────
# 3. PARAMETRI VERI — random walk confinato dentro Theta_init
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(7)

max_delta_th = 0.03  # norm(inf, theta_k - theta_{k-1})) <= max_delta_th
n_steps      = 60

# Generate random parameter trajectories (delta_a, delta_b)
th_true = np.zeros((n_steps, 2))
th_true[0] = np.array([-0.05, -0.05]) 
for k in range(1, n_steps):
    step = np.random.uniform(-max_delta_th, max_delta_th, (2,))
    th_true[k] = np.clip(th_true[k-1] + step, -delta_ab , delta_ab)

# Initialize SetUpdater and FIlter Objects

VARYING_PARAMS = True   
USE_VFF        = True   

METHODS = ["chebyshev", "lms", "rls", "kf"]

updater = SetUpdater(
    A_B=A_B, C=C,
    Theta=Theta_init, Theta_c=Theta_c_init,
    W=W, E=E, N=3, tol=0.0,
)

filters = {
    "chebyshev": Filter(A_B, C, Theta_init, Theta_c_init,
                        method="chebyshev"),
    "lms":       Filter(A_B, C, Theta_init, Theta_c_init,
                        method="lms", mu=0.3),
    "rls":       Filter(A_B, C, Theta_init, Theta_c_init,
                        method="rls", sigma0=1.0,
                        forgetting=0.97, vff=USE_VFF,
                        vff_alpha=0.97, vff_rho=0.95),
    "kf":        Filter(A_B, C, Theta_init, Theta_c_init,
                        method="kf", sigma0=1.0, Q_scale=2e-3, R_scale=1e-2,
                        forgetting=0.97, vff=USE_VFF,
                        vff_alpha=0.97, vff_rho=0.95),
}

# Initial conditions

Theta_A_norm   = (Theta_init.A / Theta_init.b[:, np.newaxis]).copy()
Theta_b_curr   = np.ones(4, dtype=float)
Theta_c_b_curr = np.array([], dtype=float)

x = np.array([[0.5]])
u = np.array([[0.0]])
z = np.vstack([x, u]).copy()

w_max, e_max = 0.05, 0.02
  

history = {
    'vol': [],
    'th_true': [],
    'th': {m: [] for m in METHODS},
    'lam':     {'rls': [], 'kf': []},
    'margins':  [],   
    'consistent': [],
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. LOOP
# ─────────────────────────────────────────────────────────────────────────────
for k in range(n_steps):

    # True Parameters at step k
    da, db = th_true[k]
    a_true = a_nom + da
    b_true = b_nom + db

    # True System Dynamics + Noises
    w      = np.random.uniform(-w_max, w_max, (1,1))
    e      = np.random.uniform(-e_max, e_max, (1,1))
    x_next = a_true * x + b_true * u + w
    y      = c_nom  * x_next + e

    # Apply Set-Membership Identification for Set Update
    NewTheta_b, _, th_vert, _ = updater.update(
        Theta        = Theta_b_curr.copy(),
        Theta_c      = Theta_c_b_curr.copy(),
        z_prev       = z.copy(),
        x_actual     = x_next.copy(),
        y_actual     = y.copy(),
        max_delta_th = max_delta_th if VARYING_PARAMS else 0.0,
    )
    if NewTheta_b is not None:
        Theta_b_curr = NewTheta_b.copy()

    vol = Polytope(A=Theta_A_norm, b=Theta_b_curr).volume()
    history['vol'].append(vol)
    history['th_true'].append(th_true[k].copy())

    #  Obtain the Estimate from each filter
    for m, filt in filters.items():
        th_hat, _ = filt.update(
            Theta_b      = Theta_b_curr.copy(),
            Theta_c_b    = Theta_c_b_curr.copy(),
            z_prev       = z.copy(),
            x_actual     = x_next.copy(),
            y_actual     = y.copy(),
        )
        history['th'][m].append(th_hat.copy())

    # Save the Forgetting Factor (lambda) for RLS and KF if VFF is enabled
    for m in ('rls', 'kf'):
        lam = filters[m]._lam if USE_VFF else filters[m].forgetting
        history['lam'][m].append(lam)

    if th_vert is not None and th_vert.ndim == 3:
        N_tube = th_vert.shape[2]
        r_th   = Theta_A_norm.shape[0]
        h_N_reconstructed = np.zeros((r_th, N_tube))
        for t in range(N_tube):
            for i in range(r_th):
                # h[i,t] = max_{v in vertici} H[i,:] @ v
                h_N_reconstructed[i, t] = np.max(
                    Theta_A_norm[i, :] @ th_vert[:, :, t].T
                )
        consistent, margins, _, _ = filters['kf'].check_consistency(
            h_N=h_N_reconstructed
        )
        history['margins'].append(margins.copy())
        history['consistent'].append(consistent)
    else:
        history['margins'].append(None)
        history['consistent'].append(None)

    cons_str = ""
    if history['consistent'][-1] is not None:
        cons_str = f" | KF⊆tube={'OK' if history['consistent'][-1] else 'WARN'}"
    print(
        f"k={k:2d} | th_true=[{da:+.3f},{db:+.3f}] | Vol={vol:.4f}"
        + (f" | lam_rls={history['lam']['rls'][-1]:.3f}"
           f" lam_kf={history['lam']['kf'][-1]:.3f}" if USE_VFF else "")
        + cons_str
    )

    x = x_next.copy()
    u = np.array([[np.sin(0.15 * k) * 0.8]])
    z = np.vstack([x, u]).copy()

# ─────────────────────────────────────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────────────────────────────────────
steps = np.arange(n_steps)
th_true_arr = np.array(history['th_true'])   # (n_steps, 2)

markers = {'chebyshev': 'o', 'lms': 's', 'rls': '^', 'kf': 'D'}
colors  = {'chebyshev': 'k', 'lms': 'g', 'rls': 'm', 'kf': 'orange'}
param_names = ['delta_a', 'delta_b']

fig, axes = plt.subplots(3 if USE_VFF else 2, 2, figsize=(14, 10 if USE_VFF else 7))
fig.suptitle(
    f"Parameter Tracking"
    f"{' + VFF' if USE_VFF else ''}"
    f"{' (Variable Parameters)' if VARYING_PARAMS else ' (Fixed Parameters)'}"
)


for p in range(2):
    ax = axes[0, p]
    ax.plot(steps, th_true_arr[:, p], 'r--', lw=2, label='true')
    for m in METHODS:
        vals = np.array(history['th'][m])[:, p]
        ax.plot(steps, vals, color=colors[m], marker=markers[m],
                ms=3, markevery=5, label=m)
    ax.set_title(param_names[p])
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

for p in range(2):
    ax = axes[1, p]
    for m in METHODS:
        vals  = np.array(history['th'][m])[:, p]
        rmse  = np.sqrt(np.cumsum((vals - th_true_arr[:, p])**2)
                        / (steps + 1))
        ax.plot(steps, rmse, color=colors[m], label=m)
    ax.set_title(f"RMSE cumulativo — {param_names[p]}")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

if USE_VFF:
    ax = axes[2, 0]
    for m in ('rls', 'kf'):
        ax.plot(steps, history['lam'][m], color=colors[m], label=m)
    ax.axhline(filters['rls'].forgetting, color='gray', ls='--',
               label=f"lambda_init={filters['rls'].forgetting}")
    ax.set_title("Variable Forgetting Factor")
    ax.set_xlabel("Step")
    ax.set_ylabel("lambda")
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(steps, history['vol'], 'b-o', ms=3)
    ax.set_title("Volume Theta nel tempo")
    ax.set_xlabel("Step")
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)
else:
    ax = axes[1, 1]
    ax.plot(steps, history['vol'], 'b-o', ms=3)
    ax.set_title("Volume Theta nel tempo")
    ax.set_xlabel("Step")
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)

valid_margins = [(k, m) for k, m in enumerate(history['margins']) if m is not None]
if valid_margins:
    fig2, axes2 = plt.subplots(1, 1, figsize=(14, 4))
    fig2.suptitle("Margini consistenza KF vs tubo SetUpdater  (> 0 = dentro)")

    steps_m = [k for k, _ in valid_margins]
    N_tube  = valid_margins[0][1].shape[1]
    r_th    = valid_margins[0][1].shape[0]

    # Minimum margin across faces for each Uncertainty Polytope propagate
    colors_t = plt.cm.viridis(np.linspace(0.2, 0.9, N_tube))
    ax = axes2[0]
    for t in range(N_tube):
        min_margins = [m[:, t].min() for _, m in valid_margins]
        ax.plot(steps_m, min_margins, color=colors_t[t],
                label=f"t+{t+1}", marker='o', ms=3, markevery=4)
    ax.axhline(0, color='r', ls='--', lw=1.5, label='limite')
    ax.set_title("Margine minimo per passo del tubo")
    ax.set_xlabel("Step k")
    ax.set_ylabel("min_i  margin[i,t]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ax = axes2[1]
    # margin_mat = np.array([m for _, m in valid_margins])  # (n_steps, r_th, N_tube)
    # # media sui passi del tubo per ogni faccia
    # avg_margin = margin_mat.mean(axis=2)  # (n_steps, r_th)
    # im = ax.imshow(avg_margin.T, aspect='auto', origin='lower',
    #                extent=[steps_m[0], steps_m[-1], 0, r_th],
    #                cmap='RdYlGn')
    # ax.set_title("Margine medio per faccia nel tempo")
    # ax.set_xlabel("Step k")
    # ax.set_ylabel("Indice faccia i")
    # fig2.colorbar(im, ax=ax, label="margine medio")

    plt.tight_layout()
    plt.savefig("consistency_margins.png", dpi=120)
    plt.show()

plt.tight_layout()
plt.savefig("filter_results.png", dpi=120)
plt.show()