"""
**adaptive_tools.py**
This module contains the implementation of the Filter and SetUpdater classes,
which are used for parameter estimation and set-membership ID.
"""

import numpy as np

from pycvxset import Polytope

import cvxpy as cp

import casadi as ca

from warnings import warn

from itertools import combinations

class SetSymbol:

    def __init__(self, h_shape, h_w_shape, N=1):
        try:
            self.h_N = cp.Variable((h_shape, N))
            self.h_0 = cp.Parameter((h_shape, 1))
            self.lambda_1 = cp.Variable((h_shape, (2 * h_shape + h_w_shape)))
            self.lambda_N = cp.Variable((h_shape, 2 * h_shape))
        except Exception:
            warn("The set is empty")
            self.h_N = np.empty((h_shape, N))
            self.h_0 = np.empty((h_shape, 1))
            self.lambda_1 = np.empty((h_shape, (2 * h_shape + h_w_shape)))
            self.lambda_N = np.empty((h_shape, 2 * h_shape))


class SetUpdater:
    """Set-membership estimation using CVXPY."""

    def __init__(
        self,
        A_B, C,
        Theta: Polytope, Theta_c: Polytope,
        W: Polytope, E: Polytope,
        N: int = 1, tol: float = 1e-1,
    ):
        self.N = N

        # FIX: copia profonda di tutte le matrici costanti usate nella costruzione
        # del problema — evita che modifiche esterne a Theta/W cambino il grafo CVXPY
        # H_theta   = (Theta.A/Theta.b[:, np.newaxis]).copy() if Theta.dim else np.empty((0, 0))
        # H_theta_c = (Theta_c.A/Theta_c.b[:, np.newaxis]).copy() if Theta_c.dim else np.empty((0, 0))
        # h_theta   = np.ones(Theta.b[:, np.newaxis].shape).copy() if Theta.dim else np.empty((0, 1))
        # h_theta_c = np.ones(Theta_c.b[:, np.newaxis].shape).copy() if Theta_c.dim else np.empty((0, 1))

        H_theta   = (Theta.A).copy() if Theta.dim else np.empty((0, 0))
        H_theta_c = (Theta_c.A).copy() if Theta_c.dim else np.empty((0, 0))
        h_theta   = Theta.b[:, np.newaxis].copy() if Theta.dim else np.empty((0, 1))
        h_theta_c = Theta_c.b[:, np.newaxis].copy() if Theta_c.dim else np.empty((0, 1))

        Hw = W.A.copy()
        He = E.A.copy()
        hw = W.b[:, np.newaxis].copy()
        he = E.b[:, np.newaxis].copy()

        # FIX: copia profonda di A_B e C — sono array numpy usati dentro einsum
        # simbolico di CVXPY; se l'esterno li modifica in-place il grafo cambia
        self._A_B = A_B.copy()
        self._C   = C.copy()

        # cp.Parameter per le misure: vengono sovrascritti ogni step in update()
        self.z = cp.Parameter((A_B.shape[2], 1))
        self.x = cp.Parameter((C.shape[2], 1))
        self.y = cp.Parameter((C.shape[1], 1))

        self.theta_problem  = None
        self.thetaC_problem = None

        if Theta.dim:
            self.theta_problem = self._init_theta_problem(
                H_theta, h_theta, Hw, hw, self._A_B, tol
            )
            self.theta_active, self.basis_inverses = self._extract_bases(
                A=H_theta, b=h_theta
            )

        if Theta_c.dim:
            self.thetaC_problem = self._init_thetac_problem(
                H_theta_c, h_theta_c, He, he, self._C, tol
            )
            self.theta_c_active, self.basis_c_inverses = self._extract_bases(
                A=H_theta_c, b=h_theta_c
            )

    # ──────────────────────────────────────────────────────────────────────────
    def _init_theta_problem(self, H_theta, h_theta, Hw, hw, A_B, tol):

        self.thetaSym = SetSymbol(H_theta.shape[0], Hw.shape[0], N=self.N)
        self.d_theta  = cp.Parameter((1, self.N))

        # Espressioni simboliche che dipendono dai cp.Parameter self.z / self.x
        #
        # Vogliamo M_w di shape (nw, n_theta) dove:
        #   M_w[:, q] = Hw @ (A_B[q+1] @ z)   per q = 0..n_theta-1
        #
        # L'einsum '...jk,kl->j...' dava shape (n_state, n_theta) invece di (nw, n_theta).
        # Fix: costruiamo colonna per colonna con hstack.
        n_theta = A_B.shape[0] - 1
        D_k_1 = cp.hstack([
            Hw @ (A_B[q + 1] @ self.z) for q in range(n_theta)
        ])  # (nw, n_theta)

        d_k = A_B[0] @ self.z - self.x   # (n_state, 1) — residuo nominale

        # ── Complessità fissa ────────────────────────────────────────────────
        # lambda_1 (r, 2r+nw) @ [H_theta(r,n); H_theta(r,n); -D_k_1(nw,n)] == H_theta(r,n)
        contract_theta_complexity = [
            self.thetaSym.lambda_1 @ cp.vstack((
                H_theta,
                H_theta,
                -D_k_1,       # già moltiplicato per Hw
            )) == H_theta
        ]

        # ── Bound passo t=0 ──────────────────────────────────────────────────
        contract_theta_bnd = [
            self.thetaSym.lambda_1 @ cp.vstack((
                h_theta,                                                      # cap iniziale
                self.thetaSym.h_0 + self.d_theta[:, 0] * np.ones((H_theta.shape[0], 1)),
                hw + Hw @ d_k    + self.d_theta[:, 0] * np.ones((hw.shape[0], 1)),
            )) <= self.thetaSym.h_N[:, [0]]
        ]

        # ── Passi dilatati N > 1 ─────────────────────────────────────────────
        if self.N > 1:
            dilating_theta_complexity = [
                self.thetaSym.lambda_N @ np.vstack((H_theta, H_theta)) == H_theta
            ]
            dilate_theta_bnd = [
                self.thetaSym.lambda_N @ cp.vstack((
                    np.tile(h_theta, (1, self.N - 1)),                        # cap fisso
                    self.thetaSym.h_0 @ np.ones((1, self.N - 1))
                    + np.ones((H_theta.shape[0], 1)) @ self.d_theta[:, 1:],
                )) <= self.thetaSym.h_N[:, 1:]
            ]
        else:
            dilating_theta_complexity = []
            dilate_theta_bnd = []

        constraints = (
            contract_theta_complexity
            + dilating_theta_complexity
            + contract_theta_bnd
            + dilate_theta_bnd
            + [self.thetaSym.lambda_1 >= 0]
            + [self.thetaSym.lambda_N >= 0]
        )

        return cp.Problem(cp.Minimize(cp.sum(self.thetaSym.h_N)), constraints)

    # ──────────────────────────────────────────────────────────────────────────
    def _init_thetac_problem(self, H_theta_c, h_theta_c, He, he, C, tol):

        self.thetaCSym  = SetSymbol(H_theta_c.shape[0], He.shape[0], N=self.N)
        self.d_theta_c  = cp.Parameter((1, self.N))

        # Dc_k_1 shape (ne, n_theta_c): colonna q = He @ (C[q+1] @ x)
        n_theta_c = C.shape[0] - 1
        Dc_k_1 = cp.hstack([
            He @ (C[q + 1] @ self.x) for q in range(n_theta_c)
        ])  # (ne, n_theta_c)

        dc_k = C[0] @ self.x - self.y   # (p, 1) — residuo nominale

        contract_thetac_complexity = [
            self.thetaCSym.lambda_1 @ cp.vstack((
                H_theta_c,
                H_theta_c,
                -Dc_k_1,
            )) == H_theta_c
        ]

        contract_thetac_bnd = [
            self.thetaCSym.lambda_1 @ cp.vstack((
                h_theta_c,
                self.thetaCSym.h_0 + self.d_theta_c[:, [0]] * np.ones((H_theta_c.shape[0], 1)),
                he + He @ dc_k    + self.d_theta_c[:, [0]] * np.ones((he.shape[0], 1)),
            )) <= self.thetaCSym.h_N[:, [0]]
        ]

        if self.N > 1:
            dilating_thetac_complexity = [
                self.thetaCSym.lambda_N @ np.vstack((H_theta_c, H_theta_c)) == H_theta_c
            ]
            dilate_thetac_bnd = [
                self.thetaCSym.lambda_N @ cp.vstack((
                    np.tile(h_theta_c, (1, self.N - 1)),
                    self.thetaCSym.h_0 @ np.ones((1, self.N - 1))
                    + np.ones((H_theta_c.shape[0], 1)) @ self.d_theta_c[:, 1:],
                )) <= self.thetaCSym.h_N[:, 1:]
            ]
        else:
            dilating_thetac_complexity = []
            dilate_thetac_bnd = []

        constraints = (
            contract_thetac_complexity
            + dilating_thetac_complexity
            + contract_thetac_bnd
            + dilate_thetac_bnd
            + [self.thetaCSym.lambda_1 >= 0]
            + [self.thetaCSym.lambda_N >= 0]
        )

        return cp.Problem(cp.Minimize(cp.sum(self.thetaCSym.h_N)), constraints)

    # ──────────────────────────────────────────────────────────────────────────
    def update(self, Theta, Theta_c, z_prev, x_actual, y_actual, max_delta_th):

        # FIX: .copy() su tutti gli array assegnati ai cp.Parameter
        # senza copy(), se il chiamante modifica z_prev/x_actual in-place
        # il valore del parametro cambierebbe retroattivamente
        self.z.value = np.array(z_prev,   dtype=float).reshape(-1, 1).copy()
        self.x.value = np.array(x_actual, dtype=float).reshape(-1, 1).copy()
        self.y.value = np.array(y_actual, dtype=float).reshape(-1, 1).copy()

        NewTheta_b = None
        NewTheta_c_b = None
        th_vertices = None
        th_c_vertices = None

        if self.theta_problem is not None:
            self.d_theta.value = (
                max_delta_th * np.arange(1, self.N + 1, dtype=float).reshape(1, -1)
            ).copy()


            self.thetaSym.h_0.value = np.array(Theta, dtype=float).reshape(-1, 1).copy()

            self.theta_problem.solve(solver=cp.CLARABEL, verbose=False)

            if self.theta_problem.status in ("optimal", "optimal_inaccurate"):
                # FIX: .copy() su h_N.value — è un buffer interno a CVXPY
                # che viene sovrascritto al prossimo solve()
                h_val = self.thetaSym.h_N.value.copy()
                th_vertices = self._get_vertices(
                    h_val, self.basis_inverses, self.theta_active
                ).copy()
                NewTheta_b = h_val[:, 0].copy()
            else:
                h_val = Theta[:,np.newaxis] + np.matlib.repmat(
                    np.array(range(1,self.N+1))*max_delta_th,
                    Theta.shape[0],
                    1)
                th_vertices = self._get_vertices(
                    h_val, self.basis_inverses, self.theta_active
                ).copy()
                warn("theta_problem status: Parameter Set has not been updated")
                NewTheta_b = np.array(Theta, dtype=float)[:, np.newaxis].copy()
        else:
            NewTheta_b = np.empty((0, 1))
            th_vertices = np.empty((1,0,self.N))


        if self.thetaC_problem is not None:
            self.d_theta_c.value = (
                max_delta_th * np.arange(1, self.N + 1, dtype=float).reshape(1, -1)
            ).copy()

            self.thetaCSym.h_0.value = np.array(Theta_c, dtype=float).reshape(-1, 1).copy()

            self.thetaC_problem.solve(solver=cp.CLARABEL, verbose=False)

            if self.thetaC_problem.status in ("optimal", "optimal_inaccurate"):
                h_c_val = self.thetaCSym.h_N.value.copy()
                th_c_vertices = self._get_vertices(
                    h_c_val, self.basis_c_inverses, self.theta_c_active
                ).copy()
                NewTheta_c_b = h_c_val[:, 0].copy()
            else:
                h_c_val = Theta_c[:,np.newaxis] + np.matlib.repmat(
                    np.array(range(1,self.N+1))*max_delta_th,Theta_c.shape[0],
                    1)
                th_c_vertices = self._get_vertices(
                    h_c_val, self.basis_c_inverses, self.theta_c_active
                ).T.copy()
                warn("thetaC_problem status: Parameter Set has not been updated")
                NewTheta_c_b = np.array(Theta_c, dtype=float)[:, np.newaxis].copy()
        else:
            NewTheta_c_b = np.empty((0, 1))
            th_c_vertices = np.empty((0, 1, self.N))

        return NewTheta_b, NewTheta_c_b, th_vertices, th_c_vertices

    # ──────────────────────────────────────────────────────────────────────────
    def _get_vertices(self, b, basis_inverses, active_bases):
        # FIX: .copy() finale — b[active_bases] è una fancy-index view
        vertices = np.einsum("ijk,ik...->ij...", basis_inverses, b[active_bases, :])
        return vertices.copy() # if self.N == 1 else vertices.copy()

    # ──────────────────────────────────────────────────────────────────────────
    def _extract_bases(self, vertices=None, A=None, b=None, tol=1e-8):
        active_bases   = []
        basis_inverses = []

        try:
            if vertices is None:
                vertices = Polytope(A=A, b=b).V
            if A is None:
                A = Polytope(V=vertices).A / Polytope(V=vertices).b[:, np.newaxis]
            if b is None:
                b = np.ones((A.shape[0], 1))
        except Exception:
            raise ValueError("The Polytope is underdefined!")

        # FIX: copia locale di A e b per evitare aliasing con gli array passati
        A = A.copy()
        b = b.copy().reshape(-1)    # shape (m,) per il confronto residuals

        m, n = A.shape

        for v in vertices:
            residuals  = A @ v - b
            active_idx = np.where(np.abs(residuals) < tol * 100)[0]

            basis_found = False
            for indices in combinations(active_idx, n):
                A_sub = A[list(indices), :]
                if abs(np.linalg.det(A_sub)) < tol:
                    continue
                active_bases.append(list(indices))
                # FIX: copia esplicita dell'inversa — np.linalg.inv restituisce
                # un nuovo array ma lo salviamo in una lista che poi stack-iamo
                basis_inverses.append(np.linalg.inv(A_sub).copy())
                basis_found = True
                break

            if not basis_found:
                raise RuntimeWarning(f"No basis found for vertex {v}.")

        return np.array(active_bases), np.array(basis_inverses)

class Filter:
    """
    Parameter estimator for systems of the form:
        x_{k+1} = A_B[0] @ z + sum_q theta[q]   * A_B[q+1] @ z
        y_k     = C[0]   @ x + sum_q theta_c[q] * C[q+1]   @ x

    Arguments:
    ----------
    A_B        : ndarray (q+1, n, n+m)
    C          : ndarray (q_c+1, p, n)
    Theta      : Polytope — uncertainty set for theta
    Theta_c    : Polytope — uncertainty set for theta_c
    method     : 'chebyshev' | 'lms' | 'rls' | 'kf'
    mu         : LMS learning rate
    sigma0     : initial covariance scale P0 = sigma0 * I  (rls, kf)
    Q_scale    : KF process noise scale Q = Q_scale * I
    R_scale    : KF/RLS measurement noise scale
    forgetting : initial forgetting factor lambda in (0,1]
    vff        : if True, adapts lambda each step via VFF
    vff_alpha  : VFF smoothing coefficient for error variance estimate
    vff_rho    : VFF target ratio sigma_e^2 / sigma_v^2
    """

    def __init__(
        self,
        A_B:       np.ndarray,
        C:         np.ndarray,
        Theta:     Polytope,
        Theta_c:   Polytope,
        method:    str   = "chebyshev",
        mu:        float = 0.05,
        sigma0:    float = 1.0,
        Q_scale:   float = 1e-4,
        R_scale:   float = 1e-2,
        forgetting: float = 0.98,
        vff:       bool  = False,
        vff_alpha: float = 0.99,
        vff_rho:   float = 0.95,
    ):
        self.method     = method
        self.mu         = mu
        self.Q_scale    = Q_scale
        self.R_scale    = R_scale
        self.forgetting = forgetting
        self.vff        = vff and (method in ("rls", "kf"))
        self.vff_alpha  = vff_alpha
        self.vff_rho    = vff_rho

        self._A_B = A_B.copy()
        self._C   = C.copy()
        self.n_theta   = A_B.shape[0] - 1
        self.n_theta_c = C.shape[0]   - 1

        # H normalizzato — fisso per tutta la vita del filtro
        self._H_th  = (Theta.A   / Theta.b[:, np.newaxis]).copy() if Theta.dim   else None
        self._H_thc = (Theta_c.A / Theta_c.b[:, np.newaxis]).copy() if Theta_c.dim else None

        # stima corrente — inizializzata al centro Chebyshev
        self._th_hat,   _ = Theta.chebyshev_centering()   if Theta.dim   else (np.zeros(self.n_theta),   None)
        self._th_c_hat, _ = Theta_c.chebyshev_centering() if Theta_c.dim else (np.zeros(self.n_theta_c), None)
        self._th_hat   = self._th_hat.flatten().copy()
        self._th_c_hat = self._th_c_hat.flatten().copy()

        # covarianza per RLS e KF
        if method in ("rls", "kf"):
            self._P   = sigma0 * np.eye(self.n_theta)   if self.n_theta   > 0 else np.empty((0, 0))
            self._P_c = sigma0 * np.eye(self.n_theta_c) if self.n_theta_c > 0 else np.empty((0, 0))

        # VFF: stima corrente della varianza dell'errore di innovazione
        # inizializzata a R_scale (misura della varianza del rumore di misura)
        if self.vff:
            self._sigma_e   = R_scale   # varianza innovazione theta
            self._sigma_e_c = R_scale   # varianza innovazione theta_c
            self._lam       = forgetting
            self._lam_c     = forgetting

        # costruisce il problema di proiezione condiviso una volta sola
        self._build_projection_problem()

    # ──────────────────────────────────────────────────────────────────────────
    # Problema di proiezione condiviso
    # ──────────────────────────────────────────────────────────────────────────

    def _build_projection_problem(self):
        """
        Costruisce un unico QP/LP di proiezione condiviso da tutti i metodi.

        Chebyshev → LP:  max r  s.t.  H[i,:] @ th + r * ||H[i,:]|| <= b[i]
        altri      → QP: min ||th - th_tilde||^2  s.t.  H @ th <= b

        b e th_tilde sono cp.Parameter aggiornati ogni step senza ricostruire.
        """
        # ── theta ─────────────────────────────────────────────────────────────
        if self._H_th is not None and self.n_theta > 0:
            r_th = self._H_th.shape[0]
            self._th_var   = cp.Variable(self.n_theta, name="th")
            self._th_tilde = cp.Parameter(self.n_theta, name="th_tilde")
            self._b_th     = cp.Parameter(r_th, name="b_th")

            if self.method == "chebyshev":
                norms       = np.linalg.norm(self._H_th, axis=1)
                self._r_th  = cp.Variable(nonneg=True, name="r_th")
                proj_cost   = cp.Minimize(-self._r_th)
                proj_con    = [self._H_th @ self._th_var + self._r_th * norms <= self._b_th]
            else:
                self._r_th  = None
                proj_cost   = cp.Minimize(cp.sum_squares(self._th_var - self._th_tilde))
                proj_con    = [self._H_th @ self._th_var <= self._b_th]

            self._proj_th = cp.Problem(proj_cost, proj_con)
        else:
            self._proj_th = None

        # ── theta_c ───────────────────────────────────────────────────────────
        if self._H_thc is not None and self.n_theta_c > 0:
            r_thc = self._H_thc.shape[0]
            self._thc_var   = cp.Variable(self.n_theta_c, name="thc")
            self._thc_tilde = cp.Parameter(self.n_theta_c, name="thc_tilde")
            self._b_thc     = cp.Parameter(r_thc, name="b_thc")

            if self.method == "chebyshev":
                norms_c      = np.linalg.norm(self._H_thc, axis=1)
                self._r_thc  = cp.Variable(nonneg=True, name="r_thc")
                proj_cost_c  = cp.Minimize(-self._r_thc)
                proj_con_c   = [self._H_thc @ self._thc_var + self._r_thc * norms_c <= self._b_thc]
            else:
                self._r_thc  = None
                proj_cost_c  = cp.Minimize(cp.sum_squares(self._thc_var - self._thc_tilde))
                proj_con_c   = [self._H_thc @ self._thc_var <= self._b_thc]

            self._proj_thc = cp.Problem(proj_cost_c, proj_con_c)
        else:
            self._proj_thc = None

    # ──────────────────────────────────────────────────────────────────────────
    # API pubblica
    # ──────────────────────────────────────────────────────────────────────────

    def update(
        self,
        Theta_b:   np.ndarray,
        Theta_c_b: np.ndarray,
        z_prev:    np.ndarray,
        x_actual:  np.ndarray,
        y_actual:  np.ndarray,
    ):
        """
        Aggiorna la stima dei parametri al passo corrente.

        Returns:
        --------
        th_hat   : ndarray (n_theta,)   — stima corrente
        th_c_hat : ndarray (n_theta_c,) — stima corrente
        """
        z = np.array(z_prev,   dtype=float).reshape(-1, 1)
        x = np.array(x_actual, dtype=float).reshape(-1, 1)
        y = np.array(y_actual, dtype=float).reshape(-1, 1)
        b   = np.array(Theta_b,   dtype=float).flatten().copy()
        b_c = np.array(Theta_c_b, dtype=float).flatten().copy()

        th_tilde, th_c_tilde = self._compute_tilde(b, b_c, z, x, y)
        self._th_hat, self._th_c_hat = self._project(th_tilde, th_c_tilde, b, b_c)

        return self._th_hat.copy()[:,np.newaxis], self._th_c_hat.copy()[:,np.newaxis]

    def predict(self, steps: int = 1):
        """
        Predizione dei parametri per i prossimi `steps` istanti.

        Solo KF ha un modello di evoluzione esplicito (random walk):
            theta_{k+t} = theta_k   (media costante, varianza cresce con Q)

        Per gli altri metodi la predizione migliore è la stima corrente.

        Returns:
        --------
        th_pred   : ndarray (n_theta,)   — predizione (costante per tutti i metodi)
        th_c_pred : ndarray (n_theta_c,) — predizione
        P_pred    : ndarray (n_theta, n_theta) | None — covarianza predetta (solo kf)
        P_c_pred  : ndarray (n_theta_c, n_theta_c) | None
        """
        if self.method == "kf" and self.n_theta > 0:
            Q      = self.Q_scale * np.eye(self.n_theta)
            P_pred = self._P + steps * Q   # propagazione lineare della covarianza
        else:
            P_pred = None

        if self.method == "kf" and self.n_theta_c > 0:
            Q_c      = self.Q_scale * np.eye(self.n_theta_c)
            P_c_pred = self._P_c + steps * Q_c
        else:
            P_c_pred = None

        return self._th_hat.copy(), self._th_c_hat.copy(), P_pred, P_c_pred

    # ──────────────────────────────────────────────────────────────────────────
    # VFF: Variable Forgetting Factor
    # ──────────────────────────────────────────────────────────────────────────

    def _update_vff(self, e: np.ndarray, Phi: np.ndarray, P: np.ndarray,
                    lam: float, sigma_e: float, is_theta_c: bool = False):
        """
        Aggiorna il forgetting factor adattivo basato sull'errore di innovazione.

        Logica VFF (Fortescue 1981, variante semplificata):
            sigma_e_new = alpha * sigma_e + (1-alpha) * e^T e / n_obs
            sigma_v     = tr(Phi P Phi^T) / n_obs     (varianza predetta)
            lam_new     = lam_min  se sigma_e > rho * sigma_v
                        = 1.0     altrimenti

        Un errore grande rispetto alla varianza predetta → dimentica più in fretta.

        Returns: lam_new, sigma_e_new
        """
        n_obs   = e.shape[0]
        # stima della varianza dell'errore di innovazione (media mobile esponenziale)
        sigma_e_new = (self.vff_alpha * sigma_e
                       + (1 - self.vff_alpha) * np.dot(e.flatten(), e.flatten()) / n_obs)

        # varianza predetta dall'innovazione (basata sulla covarianza corrente)
        sigma_v = np.trace(Phi @ P @ Phi.T) / n_obs + self.R_scale

        # se l'errore supera rho * sigma_v, riduci lambda
        # lambda minimo = 1 - (1-rho): più rho è alto, più lentamente si dimentica
        lam_min = max(0.9, self.vff_rho)
        if sigma_e_new > self.vff_rho * sigma_v:
            lam_new = lam_min
        else:
            lam_new = 1.0

        return lam_new, sigma_e_new

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: calcolo di th_tilde
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_tilde(self, b, b_c, z, x, y):
        if self.method == "chebyshev":
            return None, None
        elif self.method == "lms":
            return self._tilde_lms(z, x, y)
        elif self.method == "rls":
            return self._tilde_rls(z, x, y)
        elif self.method == "kf":
            return self._tilde_kf(z, x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _tilde_lms(self, z, x, y):
        th_tilde = th_c_tilde = None

        if self.n_theta > 0:
            Phi      = np.hstack([self._A_B[q+1] @ z for q in range(self.n_theta)])
            x_hat    = self._A_B[0] @ z + Phi @ self._th_hat.reshape(-1, 1)
            e        = x - x_hat
            th_tilde = self._th_hat + self.mu * (Phi.T @ e).flatten()

        if self.n_theta_c > 0:
            Phi_c      = np.hstack([self._C[q+1] @ x for q in range(self.n_theta_c)])
            y_hat      = self._C[0] @ x + Phi_c @ self._th_c_hat.reshape(-1, 1)
            e_c        = y - y_hat
            th_c_tilde = self._th_c_hat + self.mu * (Phi_c.T @ e_c).flatten()

        return th_tilde, th_c_tilde

    def _tilde_rls(self, z, x, y):
        """
        RLS con forgetting factor (fisso o VFF adattivo).

        La storia passata viene pesata esponenzialmente:
            J = sum_{i=0}^{k} lambda^{k-i} ||x_i - Phi_i theta||^2
        con lambda=1 equivale a LS su tutta la storia.
        Con VFF, lambda si riduce automaticamente quando l'errore cresce.
        """
        th_tilde = th_c_tilde = None

        if self.n_theta > 0:
            lam  = self._lam if self.vff else self.forgetting
            Phi  = np.hstack([self._A_B[q+1] @ z for q in range(self.n_theta)])
            x_hat = self._A_B[0] @ z + Phi @ self._th_hat.reshape(-1, 1)
            e     = x - x_hat

            S  = lam * np.eye(Phi.shape[0]) + Phi @ self._P @ Phi.T
            K  = self._P @ Phi.T @ np.linalg.inv(S)
            th_tilde = self._th_hat + (K @ e).flatten()
            self._P  = (1/lam) * (self._P - K @ Phi @ self._P)

            if self.vff:
                self._lam, self._sigma_e = self._update_vff(
                    e, Phi, self._P, lam, self._sigma_e
                )

        if self.n_theta_c > 0:
            lam_c = self._lam_c if self.vff else self.forgetting
            Phi_c = np.hstack([self._C[q+1] @ x for q in range(self.n_theta_c)])
            y_hat = self._C[0] @ x + Phi_c @ self._th_c_hat.reshape(-1, 1)
            e_c   = y - y_hat

            S_c = lam_c * np.eye(Phi_c.shape[0]) + Phi_c @ self._P_c @ Phi_c.T
            K_c = self._P_c @ Phi_c.T @ np.linalg.inv(S_c)
            th_c_tilde = self._th_c_hat + (K_c @ e_c).flatten()
            self._P_c  = (1/lam_c) * (self._P_c - K_c @ Phi_c @ self._P_c)

            if self.vff:
                self._lam_c, self._sigma_e_c = self._update_vff(
                    e_c, Phi_c, self._P_c, lam_c, self._sigma_e_c, is_theta_c=True
                )

        return th_tilde, th_c_tilde

    def _tilde_kf(self, z, x, y):
        """
        KF per stima parametri con modello random walk:
            theta_{k+1} = theta_k + w_k,   w_k ~ N(0, Q)   ← evoluzione
            x_k = Phi_k theta_k + v_k,     v_k ~ N(0, R)   ← osservazione

        Con VFF, Q viene scalato dinamicamente: se l'errore cresce,
        Q aumenta per permettere al filtro di inseguire variazioni più rapide.
        """
        th_tilde = th_c_tilde = None

        if self.n_theta > 0:
            lam   = self._lam if self.vff else self.forgetting
            Phi   = np.hstack([self._A_B[q+1] @ z for q in range(self.n_theta)])
            x_hat = self._A_B[0] @ z + Phi @ self._th_hat.reshape(-1, 1)
            e     = x - x_hat

            # Q scalato con (1-lam)/lam per rendere VFF equivalente a RLS
            Q_eff  = (self.Q_scale + (1 - lam) / max(lam, 1e-6)) * np.eye(self.n_theta)
            R      = self.R_scale * np.eye(Phi.shape[0])

            P_pred = self._P + Q_eff                          # predict
            S      = Phi @ P_pred @ Phi.T + R
            K      = P_pred @ Phi.T @ np.linalg.inv(S)
            th_tilde = self._th_hat + (K @ e).flatten()
            self._P  = (np.eye(self.n_theta) - K @ Phi) @ P_pred   # update

            if self.vff:
                self._lam, self._sigma_e = self._update_vff(
                    e, Phi, P_pred, lam, self._sigma_e
                )

        if self.n_theta_c > 0:
            lam_c = self._lam_c if self.vff else self.forgetting
            Phi_c = np.hstack([self._C[q+1] @ x for q in range(self.n_theta_c)])
            y_hat = self._C[0] @ x + Phi_c @ self._th_c_hat.reshape(-1, 1)
            e_c   = y - y_hat

            Q_c_eff = (self.Q_scale + (1 - lam_c) / max(lam_c, 1e-6)) * np.eye(self.n_theta_c)
            R_c     = self.R_scale * np.eye(Phi_c.shape[0])

            P_c_pred   = self._P_c + Q_c_eff
            S_c        = Phi_c @ P_c_pred @ Phi_c.T + R_c
            K_c        = P_c_pred @ Phi_c.T @ np.linalg.inv(S_c)
            th_c_tilde = self._th_c_hat + (K_c @ e_c).flatten()
            self._P_c  = (np.eye(self.n_theta_c) - K_c @ Phi_c) @ P_c_pred

            if self.vff:
                self._lam_c, self._sigma_e_c = self._update_vff(
                    e_c, Phi_c, P_c_pred, lam_c, self._sigma_e_c, is_theta_c=True
                )

        return th_tilde, th_c_tilde

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: proiezione condivisa
    # ──────────────────────────────────────────────────────────────────────────

    def _project(self, th_tilde, th_c_tilde, b, b_c):
        """
        Proietta th_tilde dentro {H @ theta <= b} usando il problema pre-costruito.
        Shortcut: se th_tilde è già feasibile, salta il solver (solo QP).
        """
        th_out   = self._th_hat.copy()
        th_c_out = self._th_c_hat.copy()

        # ── theta ─────────────────────────────────────────────────────────────
        if self._proj_th is not None:
            self._b_th.value = b.copy()

            if self.method == "chebyshev":
                self._proj_th.solve(solver=cp.CLARABEL, verbose=False)
            else:
                if np.all(self._H_th @ th_tilde <= b + 1e-8):
                    th_out = th_tilde.copy()   # già dentro — nessun solve
                else:
                    self._th_tilde.value = th_tilde.copy()
                    self._proj_th.solve(solver=cp.CLARABEL, verbose=False)
                    if self._proj_th.status in ("optimal", "optimal_inaccurate"):
                        th_out = self._th_var.value.copy()
                    else:
                        warn(f"Projection theta ({self.method}): "
                             f"{self._proj_th.status} — invariata")

            if self.method == "chebyshev" and \
               self._proj_th.status in ("optimal", "optimal_inaccurate"):
                th_out = self._th_var.value.copy()

        # ── theta_c ───────────────────────────────────────────────────────────
        if self._proj_thc is not None:
            self._b_thc.value = b_c.copy()

            if self.method == "chebyshev":
                self._proj_thc.solve(solver=cp.CLARABEL, verbose=False)
            else:
                if np.all(self._H_thc @ th_c_tilde <= b_c + 1e-8):
                    th_c_out = th_c_tilde.copy()
                else:
                    self._thc_tilde.value = th_c_tilde.copy()
                    self._proj_thc.solve(solver=cp.CLARABEL, verbose=False)
                    if self._proj_thc.status in ("optimal", "optimal_inaccurate"):
                        th_c_out = self._thc_var.value.copy()
                    else:
                        warn(f"Projection theta_c ({self.method}): "
                             f"{self._proj_thc.status} — invariata")

            if self.method == "chebyshev" and \
               self._proj_thc.status in ("optimal", "optimal_inaccurate"):
                th_c_out = self._thc_var.value.copy()

        return th_out, th_c_out