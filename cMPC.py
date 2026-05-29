"""cMPC: A CasADi-based linear-quadratic Model Predictive Controller (MPC) implementation."""

import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse import block_diag
import casadi as ca

from pycvxset import Polytope
import cvxpy as cp

from symbolic import Symbolic
from options import Options
from polytopesystem import PolytopeSystem
from cartesian_product import cartesian_product
from pagemtimes import pagemtimes
from create_system import create_system

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

INF = 1e4


class CMPC:
    """A linear-quadratic Model Predictive Controller (MPC) using CasADi.

    This class implements a discrete-time MPC problem for linear systems of the
    form:

        x_{k+1} = A x_k + B u_k
        y_k     = C x_k

    where the objective is a quadratic cost on states and inputs.

    The controller supports both regulation (zero reference) and tracking modes
    (constant or trajectory reference) and allows defining state/input/output
    constraints via polyhedral sets.

    Note
    ----
    This implementation is inspired by a MATLAB/yMPC implementation, but uses
    CasADi for symbolic variables and optimization.
    """

    def __init__(self, sys, Q, R, N, options=None):
        """Initialize the MPC controller.

        Parameters
        ----------
        sys : object
            Plant model object with attributes ``A`` (state matrix), ``B`` (input
            matrix) and ``C`` (output matrix).
        Q : np.ndarray
            State cost matrix (quadratic weight on state deviations).
        R : np.ndarray
            Input cost matrix (quadratic weight on input effort).
        N : int
            Prediction horizon (number of steps).
        options : dict, optional
            Dictionary of optional settings. Supported keys:

            - ``Ts`` : Sampling time (if provided, system matrices are scaled).
            - ``solver`` : QP solver name (default: ``'quadprog'``).
            - ``verbose`` : Verbosity level (default: 0).
            - ``relax`` : Relaxation flag for constraints (default: False).
            - ``ref`` : Reference type (``'traj'``/``'trajectory'`` for full trajectory)
              or ``None`` for regulation.
            - ``name`` : Controller name.
            - ``svd`` : Whether to use SVD-based reduction for tracking.
            - ``Nc`` : Control horizon (defaults to prediction horizon).
            - ``sigma`` : Scaling factor for reachable set computation.
            - ``K`` : Stabilizing feedback gain for terminal constraints.
            - ``xBound``, ``uBound``, ``yBound`` : Tuple (lower, upper) bounds.
        """

        self.options = Options(options)

        # ---------------------------------------------------------------------
        # System matrices and sampling time handling
        # ---------------------------------------------------------------------
        # If a sampling time is provided, assume sys['A']/sys['B'] are continuoustime
        # matrices and form the (approximate) discrete-time equivalents.
        #
        # Given a LPI system or a polytopic uncertainty, the system matricies
        # could be represented as follows:
        #
        # A(p) = A_0 + A_1 @ p_1 + A_2 @ p_2 + .... A_t @ p_t
        # B(p) = B_0 + B_1 @ p_1 + B_2 @ p_2 + .... B_t @ p_t
        # C(p) = C_0 + C_1 @ p_1 + C_2 @ p_2 + .... C_t @ p_t
        #
        # To encapsulate this representation, we can store the system matrices
        # as 3D tensors, where the last axis correspond to the i-th slice which
        # depends on p_i (i.e. the first element of the first axis corresponds
        # to the nominal system)
        # As matricies could be provided as regular 2D arrays, we should reshape
        # them to 3D tensors

        # The 3D tensor representation will be also adopted to represent
        # matrices evaluated at the vertices of the uncertainty set

        self.sys = create_system(sys, self.options.ts)

        # ---------------------------------------------------------------------
        # Prediction horizon + state/input/output dimensions
        # ---------------------------------------------------------------------
        self.N = N
        self.n = self.sys.A.shape[0]
        self.m = self.sys.B.shape[1]
        self.p = self.sys.C.shape[0]

        # ---------------------------------------------------------------------
        # Decision variables (CasADi symbolic definitions)
        # ---------------------------------------------------------------------
        # x and u are horizon-length sequences of states/inputs.
        self.sym = Symbolic(self.n, self.m, self.N)

        # Cost accumulators (may be overwritten during optimization setup)
        self.cost_fun = 0

        # ---------------------------------------------------------------------
        # Cost weighting matrices
        # ---------------------------------------------------------------------
        self.Q = np.zeros((self.n, self.n))
        if Q is not None:
            self.Q = Q

        self.R = np.zeros((self.m, self.m))
        if R is not None:
            self.R = R

        # Soft-term weights (small regularization terms)
        self.S = Q * 1e-4
        self.T = R * 1e-4

        # ---------------------------------------------------------------------
        # Tracking versus regulation setup
        # ---------------------------------------------------------------------
        # If a reference is supplied, enable tracking mode.
        self.track = self.options.set_track(self.sym, self.n, self.m, self.p, self.N)

        # ---------------------------------------------------------------------
        # Controller identity
        # ---------------------------------------------------------------------
        self.name = self.options.name

        # ---------------------------------------------------------------------
        # Optional SVD-based reduction for tracking (reduces artificial variables)
        # ---------------------------------------------------------------------
        self.svd_flag = bool(self.options.svd and self.track)

        # ---------------------------------------------------------------------
        # Control horizon (Nc) defaults to prediction horizon if not provided
        # ---------------------------------------------------------------------
        self.Nc = self.options.Nc if self.options.Nc not in (None, 0) else self.N

        # ---------------------------------------------------------------------
        # Terminal stabilizing gain (K) for terminal constraint
        # ---------------------------------------------------------------------
        self.K = (
            self.options.K
            if (self.options.K is not None and not np.all(self.options.K == 0))
            else np.zeros((self.m, self.n))
        )

        if self.options.W is not None and isinstance(self.options.W, Polytope):
            self.W = self.options.W
        else:
            self.W = Polytope(
                A=np.vstack((np.eye(self.n), -np.eye(self.n))),
                b=np.concatenate((np.ones(self.n) * 0.01, np.ones(self.n) * 0.01)),
            )
        
        self.options.lam = 0.95 if self.options.lam is None else self.options.lam

        # ---------------------------------------------------------------------
        # State / input / output constraints
        # ---------------------------------------------------------------------
        # Bounds are expected as tuples: (lower_bound, upper_bound)
        lxb, uxb, lub, uub, lyb, uyb = self.options.get_bound(self.n, self.m, self.p)

        # Initialize the MX variables storing the symbolic constraints as empty symbolic MX

        self.g = []
        self.lbg = []
        self.ubg = []

        D = np.zeros((self.p, self.m))

        # Polyhedral sets for constraints (X: state, U: input, Y: output)
        self.polys = PolytopeSystem(
            self.n, self.m, self.p, (uxb, lxb, uub, lub, uyb, lyb)
        )

        self.polys.zc = Polytope(
            A=self.polys.y.A, # @ np.hstack((self.sys.C[:, :, 0], D)),
            b=self.polys.y.b,
        )

        if self.track:
            # Shrink (x,u) constraint set to account for reachability margin
            self.polys.zs = self.options.sigma * self.polys.z
            # Intersect the reachable set with the equilibrium condition
        else:
            self.polys.zs = Polytope(V=np.zeros((1, self.n + self.m)))

        self.f_const = []
        self.g_const = []
        self.qpsol = None
        self.sol = None
        self.Ak = None
        self.mn = None
        self.mx = None
        self.l_mat = None
        self.sol_partial = None
        self.u_star = None
        self.nu = None
        self.poly_x_aug = None
        self.P = None
        self.Nn = None
        self.lam = self.options.lam

    def add_hard_constraints(self, *hConstraints):
        """Add hard constraints to the MPC problem."""

        _fg = ca.MX.sym("f_g", self.n + self.m, 1)
        # lin_constr = ca.Function("lin_constr", [self.sym._z, _fg], [_fg @ self.sym._z])

        lin_constr = ca.Function(
            "lin_constr", [self.sym.get_z(), _fg], [np.matmul(_fg.T, self.sym.get_z())]
        )

        if not hConstraints:

            f_g = np.vstack(
                (
                    self.polys.z.A[self.polys.z.b < INF, :]
                    / self.polys.z.b[self.polys.z.b < INF, np.newaxis],
                )
            
            )

            fc_gc = np.vstack(
                (
                    self.polys.zc.A[self.polys.zc.b < INF, :]
                    / self.polys.zc.b[self.polys.zc.b < INF, np.newaxis],
                )
            )

            self.f_const = f_g[:, : self.n]
            self.g_const = f_g[:, self.n:]

            self.fc_const = fc_gc[:, : self.n]
            self.gc_const = fc_gc[:, self.n:]

            for i in range(f_g.shape[0]):
                self.g.append(
                    ca.reshape(lin_constr.map(self.N)(self.sym.z, f_g[i, :].T), (1, -1)).T
                )
                self.lbg.append([-ca.inf] * self.N)
                self.ubg.append([1] * self.N)


            if self.sys.C.shape[2]==1:
                for i in range(fc_gc.shape[0]):
                    self.g.append(
                        ca.reshape(lin_constr.map(self.N)(self.sym.z, (fc_gc[i, :] @ self.sys.C[:,:,0]).T), (1, -1)).T
                    )
                    self.lbg.append([-ca.inf] * self.N)
                    self.ubg.append([1] * self.N)

        else:
            self.f_const = [[None, None], [None, None]]
            self.g_const = []
            for h_constraint in hConstraints:
                if isinstance(h_constraint, Polytope):
                    poly = h_constraint
                    f = (poly.A / poly.b[:, np.newaxis])[:, : self.n]
                    g = (poly.A / poly.b[:, np.newaxis])[:, self.n:]
                    self.f_const = np.vstack((self.f_const, f))
                    self.g_const = np.vstack((self.g_const, g))
                    f = ca.DM(f)
                    g = ca.DM(g)

                elif isinstance(h_constraint, np.ndarray):
                    if h_constraint.shape[1] > (self.m + self.n):
                        raise ValueError("Wrong size of constraints matrix!")
                    f = h_constraint[:, : self.n + 1]
                    g = h_constraint[:, self.n + 1: self.n + 1 + self.m]
                    self.f_const = np.vstack((self.f_const, f))
                    self.g_const = np.vstack((self.g_const, g))
                    f = ca.DM(f)
                    g = ca.DM(g)

                elif isinstance(h_constraint, tuple):
                    f, g = h_constraint
                    f = ca.DM(f)
                    g = ca.DM(g)
                    self.f_const = np.vstack((self.f_const, np.array(f)))
                    self.g_const = np.vstack((self.g_const, np.array(g)))

                else:
                    raise ValueError(
                        "Check the constraints datatype, accepted types are: \n"
                        "    1) pycvxset.Politope()\n"
                        "    2) numpy.ndArray()\n"
                        "    3) tuple(F,G) as (numpy.ndarray/casadi.MX, numpy.ndarray/casadi.MX)"
                    )
                for i in range(f.shape[0]):
                    self.g.append(
                        ca.reshape(
                            lin_constr.map(self.N)(
                                self.sym.get_z(), np.hstack((f[i, :], g[i, :])).T
                            ),
                            (1, -1)).T
                    )
                    self.lbg.append([-ca.inf] * self.N)
                    self.ubg.append([1] * self.N)

    def add_soft_constraints(self, *sConstraints):
        """Add soft constraints (penalized violations)."""

        # TODO - Similar to the previous constraint

    def initialize(self, mode=None, constraints=None):
        """Prepare the controller (compute terminal sets, assemble QP)."""

        # Build hard constraints (user-defined)
        self.add_hard_constraints()
        if constraints is not None:
            self.add_hard_constraints(constraints)

        # If no stabilizing gain is provided, compute one via LMI solver.
        if not self.K.any():
            self._stab_gain(self.sys.A, self.sys.B, self.W.V, mode)

        # Closed-loop matrix for terminal set computations.
        self.Ak = self.sys.A + np.einsum('ij...,jl->il...', self.sys.B, self.K)
        # pagemtimes(self.sys.B, self.K[:, :, np.newaxis])

        # Optionally reduce the number of artificial variables via SVD.
        if not self.svd_flag:
            self.mn = np.eye((self.n + self.m))
        else:
            self._svd_decomposition()
            self.mx = np.hstack((np.eye(self.n), np.zeros((self.n, self.m)))) @ self.mn

        # Build augmented system for terminal set computation.
        self.l_mat = np.hstack((-self.K, np.eye(self.m))) @ self.mn
        a_aug1 = np.hstack(
            (self.Ak, np.einsum('ij...,jl->il...', self.sys.B, self.l_mat))
            #  pagemtimes(self.sys.B, self.l_mat[:, :, np.newaxis]))
        ).squeeze()
        a_aug2 = np.hstack(
            (np.zeros((np.size(self.mn, 1), self.n)), np.eye(np.size(self.mn, 1)))
        )
        a_aug = np.vstack((a_aug1, a_aug2))

        # Initial polyhedron for maximal invariant set computation.
        self.polys.z.b[self.polys.z.b == INF] = np.inf
        if self.track:
            x0_poly = Polytope(
                A=self.polys.z.A
                @ np.block(
                    [
                        [np.eye(self.n), np.zeros((self.n, np.size(self.mn, 1)))],
                        [self.K, self.l_mat],
                    ]
                ),
                b=self.polys.z.b,
            )
            # poly2 -> cartesian_product(Polytope(V=np.zeros((1,self.n))),self.Zs)

            x0_poly = x0_poly.intersection(
                cartesian_product(Polytope(V=np.zeros((1, self.n))), self.polys.zs)
            )
        else:
            x0_poly = Polytope(
                A=self.polys.z.A @ np.block([[np.eye(self.n)], [self.K]]),
                b=self.polys.z.b,
            )
            a_aug = self.Ak.squeeze()

        # Compute invariant terminal set (lambda = 1)
        self._lam_contract_set(a_aug, x0_poly, 1)

        # DMPC-related flag (kept for compatibility)
        self.sol_partial = False

        # Build cost and constraints for the underlying QP
        self._cost_fun_build()
        self._system_build()

        opt = {
            "verbose": True,
            "print_time": True,
        }  # TODO: Fill the dictionary with options for the QP solver
        self._set_controller(opt)

    def solve(self, x0, r=None):
        """Solve the MPC problem for the current state and reference."""
        if r is None:
            r = np.zeros(self.sym.r.shape)
        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))
        self.sol = self.qpsol(p=ca.vertcat(x0, r), lbg=new_lbg, ubg=new_ubg)

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n: (self.N + 1) * self.n + self.m
        ]

    def _cost_fun_build(self):
        """Build the MPC quadratic cost function."""

        # If Q , R, P ,S or T are None wont be used for cost construction
        if self.P is None:
            self.P = self.Q*100
        _q = ca.MX.sym("_Q", self.n, self.n)
        _r = ca.MX.sym("_R", self.m, self.m)

        quad_cost = ca.Function(
            "quad_cost",
            [self.sym.get_x(), self.sym.get_u(), _q, _r],
            [
                self.sym.get_x().T @ _q @ self.sym.get_x()
                + self.sym.get_u().T @ _r @ self.sym.get_u()
            ],
        )

        if not self.track:
            stage_cost = ca.sum2(
                quad_cost.map(self.N)(
                    self.sym.x[:, : self.N], self.sym.u, self.Q, self.R
                )
            )
            terminal_cost = quad_cost(
                self.sym.x[:, -1],
                np.zeros((self.m, 1)),
                self.P,
                np.zeros((self.m, 1)),
            )
            offset_cost = 0

        else:

            stage_cost = ca.sum2(
                quad_cost.map(self.N)(
                    self.sym.x[:, : self.N] - self.sym.xa[:, : max(self.sym.xa.shape[1]-1, 1)],
                    self.sym.u - self.sym.ua,
                    self.Q,
                    self.R,
                )
            )
            terminal_cost = quad_cost(
                self.sym.x[:, -1] - self.sym.xa[:, -1],
                ca.GenMX_zeros((self.m, 1)),
                self.P,
                ca.GenMX_zeros((self.m, self.m)),
            )
            offset_cost = ca.sum2(
                quad_cost.map(self.sym.r.shape[1])(
                    self.sys.C.squeeze().T @ (np.array([self.sys.C.squeeze()]) @ self.sym.xa - self.sym.r),
                    np.zeros((self.m, 1)),
                    self.Q,
                    np.zeros((self.m, self.m)),
                )
            )
        if self.options.customJ is None:
            self.options.customJ = 0

        self.cost_fun = stage_cost + terminal_cost + offset_cost + self.options.customJ

    def _system_build(self):
        """Internal function for building system related constraints such as:
        - system dynamics
        - artificial setpoints dynamics/equilibrium
        - initial conditions
        - terminal conditions
        """

        step = ca.Function(
            "step",
            [self.sym.get_x(), self.sym.get_u()],
            [
                self.sys.A.squeeze() @ self.sym.get_x()
                + self.sys.B.squeeze() @ self.sym.get_u()
            ],
        )

        linear_k = ca.Function(
            "linearK", [self.sym.get_x()], [self.K @ self.sym.get_x()]
        )

        g_dyn = self.sym.x[:, 1:] - step.map(self.N)(
            self.sym.x[:, : self.N], self.sym.u
        )  # system dynamic
        g_init = self.sym.x[:, 0] - self.sym.x_init  # Initial condition

        self.g.append(g_init) 
        self.g.append(ca.reshape(g_dyn, (1, -1)).T)

        self.lbg.append([0.0] * (self.n + (self.N * self.n) - 1))
        self.ubg.append([0.0] * (self.n + (self.N * self.n) - 1))

        # Control Horizon Constraints

        self.g.append(
            ca.reshape(
                (
                    self.sym.u[:, self.Nc: self.N]
                    - linear_k.map(self.N + 1 - self.Nc)(
                        self.sym.x[:, self.Nc: self.N + 1]
                    )
                ),
                (1, -1)).T
            )
        
        self.lbg.append(np.zeros((self.N + 1 - self.Nc)))
        self.ubg.append(np.zeros((self.N + 1 - self.Nc)))

        if self.track:
            if self.sym.r.shape[1] > 1:
                # Artificial trajectory
                self.g.append(ca.reshape(
                        self.sym.xa[:, 1:]
                        - step.map(self.N)(self.sym.xa[:, : self.N], self.sym.ua), (1, -1)).T
                    )
                self.lbg.append(np.zeros((self.N * self.n)))
                self.ubg.append(np.zeros((self.N * self.n)))
            # Artificial Setpoints
            if self.svd_flag:
                self.g.append(
                    ca.vertcat(
                        self.sym.xa[:, -1], self.sym.ua[:, -1], self.sym.r[:, -1]
                    )
                    - block_diag(self.mn, self.Nn).toarray() @ self.nu
                )
                self.lbg.append(np.zeros((self.n + self.m + self.p)))
                self.ubg.append(np.zeros((self.n + self.m + self.p)))
                self.g.append(
                    self.poly_x_aug.A @ ca.vertcat(self.sym.x[:, -1], self.nu)
                )
                self.lbg.append([-ca.inf] * (len(self.poly_x_aug.b)))
                self.ubg.append(self.poly_x_aug.b)
            else:
                self.g.append(
                    self.poly_x_aug.A
                    @ ca.vertcat(
                        self.sym.x[:, -1], self.sym.xa[:, -1], self.sym.ua[:, -1]
                    )
                )
                self.lbg.append([-ca.inf] * (len(self.poly_x_aug.b)))
                self.ubg.append(self.poly_x_aug.b)

            # Artificial Setpoints
            self.g.append(
                (
                    (self.sys.A.squeeze() - np.eye(self.n)) @ self.sym.xa[:, -1]
                    + self.sys.B.squeeze() @ self.sym.ua[:, -1]
                )
            )
            self.lbg.append(np.zeros((self.n)))
            self.ubg.append(np.zeros((self.n)))

    def _stab_gain(self, a, b, w_v, cost):
        """Compute a stabilizing state-feedback gain K via LMIs.

        Uses CVXPY to solve the LMI formulation that yields a stabilizing gain.

        The literature reference is : Linear Robust adaptive model predictive control:\
            Computational complexity and conservatism - exended version - Appendix Kohler et al.

        """
        # TODO - adding some best practice modularity - check LQR or define another one
        # Arbitrarly Small Tolerance Value for strict inequality approximation
        tol = 1e-8
        if cost is None:
            cost = "LQR"

        x_mat = cp.Variable((self.n, self.n), symmetric=True)
        y_mat = cp.Variable((self.m, self.n))

        gamma = cp.Variable(1, "gamma")
        lambd = cp.Parameter(1, "lambda")
        tau = cp.Parameter(1, "tau")

        v = a.shape[2]  # Number of vertices

        # Define the LMIs

        constraints = []
        c1 = []

        c2_1 = cp.hstack(
            (
                cp.vstack(
                    (
                        x_mat @ fractional_matrix_power(self.Q, 0.5),
                        np.zeros((self.n, self.n)),
                    )
                ),
                cp.vstack(
                    (
                        y_mat.T @ fractional_matrix_power(self.R, 0.5),
                        np.zeros((self.n, self.m)),
                    )
                ),
            )
        )
        for j in range(v):
            c1_1 = cp.hstack((x_mat, (a[:, :, j] @ x_mat + b[:, :, j] @ y_mat).T))
            c1_1 = cp.vstack(
                (c1_1, cp.hstack((a[:, :, j] @ x_mat + b[:, :, j] @ y_mat, x_mat)))
            )

            c1.append(
                cp.vstack(
                    (
                        cp.hstack((c1_1, c2_1)),
                        cp.hstack((c2_1.T, gamma * np.eye(self.n + self.m))),
                    )
                )
                >> tol
            )

        # Contractiveness

        c2 = []
        for j in range(v):
            c1_2 = cp.hstack((lambd * x_mat, a[:, :, j] @ x_mat + b[:, :, j] @ y_mat))
            c1_2 = cp.vstack(
                (
                    c1_2,
                    cp.hstack(
                        ((a[:, :, j] @ x_mat + b[:, :, j] @ y_mat).T, lambd * x_mat)
                    ),
                )
            )
            c2.append(c1_2 >> tol)

        # Constraint Satisfaction

        c3 = []

        for i in range(np.size(self.f_const, 0)):
            c1_3 = cp.hstack(
                (1, self.f_const[i, :] @ x_mat + self.g_const[i, :] @ y_mat)
            )
            c2_3 = cp.vstack(
                (
                    c1_3,
                    cp.hstack(
                        (
                            (self.f_const[i, :] @ x_mat + self.g_const[i, :] @ y_mat)[
                                :, np.newaxis
                            ],
                            x_mat,
                        )
                    ),
                )
            )
            c3.append(c2_3 >> tol)

        # Noise Attenuation

        c4 = []

        w_v = np.array(w_v)
        if w_v.any():
            for j in range(v):
                for k in range(np.size(w_v, 0)):
                    c1_4 = cp.hstack(
                        (
                            tau * x_mat,
                            np.zeros((self.n, 1)),
                            (a[:, :, j] @ x_mat + b[:, :, j] @ y_mat).T,
                        )
                    )
                    c2_4 = cp.hstack((np.zeros((1, self.n)), (1 - tau)[np.newaxis], w_v[k,:,np.newaxis].T))
                    c3_4 = cp.vstack(
                        (
                            c1_4,
                            c2_4,
                            cp.hstack(
                                (
                                    a[:, :, j] @ x_mat + b[:, :, j] @ y_mat,
                                    w_v[k, :, np.newaxis],
                                    x_mat,
                                )
                            ),
                        )
                    )

                    c4.append(c3_4 >> tol)

        constraints.append(x_mat >> tol * np.eye(self.n))
        constraints = constraints + c1 + c2 + c3 + c4 + [lambd >= self.options.lam]
        j_cost = 0
        if cost == "volume":
            j_cost = cp.Minimize(-cp.log_det(x_mat))
        elif cost == "performance":
            j_cost = cp.Minimize(gamma[0])
        elif cost == "LQR":
            j_cost = cp.Minimize(cp.trace(x_mat))

        # LMI Problem

        lmip = cp.Problem(j_cost, constraints)

        tau.value = [0.95]
        lambd.value = [self.lam] #[self.options.lam +0.05]  # Start with a slightly larger lambda to ensure feasibility

        satisfied = False

        error = 0.0
        new_lambd = 0.0
        alpha = 0.5

        rho = np.zeros(v)  # spectral radius for each vertex

        while not satisfied:

            lmip.solve(solver='CLARABEL')
            self.P = np.linalg.inv(x_mat.value)
            self.K = y_mat.value @ self.P
            for j in range(v):

                rho[j] = np.linalg.norm(
                    np.linalg.eig(a[:, :, j] + b[:, :, j] @ self.K).eigenvalues, np.inf
                )
            max_rho = np.max(rho)
            target = np.max([self.lam, max_rho + 0.7*(1.0 - max_rho)])
            new_lambd = lambd.value[0] + alpha * (target - lambd.value[0])
            error = target - lambd.value[0]
            satisfied = (np.sqrt(error**2) <= tol)
            lambd.value[0] = new_lambd
            # print(f"Current lambda: {lambd.value[0]:.4f}, spectral radius: {max_rho:.4f}, error: {error:.4e}, target: {target:.4f}")

        self.lam = lambd.value[0]

    def _svd_decomposition(self):
        """Compute SVD-based transformation matrices for reduced artificial variables.

        Alvarado PhD Thesis

        https://idus.us.es/server/api/core/bitstreams/07282423-137f-4294-b314-78092d044ff6/content
        """

        e1 = np.hstack((self.sys.A - np.eye(self.n), self.sys.B))
        e2 = np.hstack((self.sys.C, np.zeros((self.p, self.m))))
        e = np.vstack((e1, e2))

        f = np.vstack((np.zeros((self.n, self.p)), np.eye(self.p)))

        u_mat, sing_val, v_mat = np.linalg.svd(e)
        v_mat = (
            v_mat.T
        )  # I don't know why but matlab and numpy give different shape of V
        sing_val = np.diag(sing_val)  # transform the array in a matrix
        sing_inv = np.linalg.pinv(sing_val)
        if np.size(sing_inv, 1) != np.size(u_mat, 0):
            sing_inv = np.hstack((sing_inv, np.zeros((np.size(sing_val, 1), 1))))

        v_p = v_mat[:, np.linalg.matrix_rank(sing_val):]
        u_p = u_mat[:, np.linalg.matrix_rank(sing_val):]

        if np.size(u_mat, 1) == (self.p + self.n):
            g_mat = np.eye(self.p)
        elif np.size(u_mat, 1) < (self.p + self.n):
            g_mat = f.T @ u_p
            g_mat = g_mat[:][np.linalg.matrix_rank(sing_val):]

        if np.size(u_mat, 1) < (self.m + self.n):
            self.mn = np.hstack((v_mat @ sing_inv @ u_mat.T @ f @ g_mat, v_p))
            self.Nn = np.hstack(
                (g_mat, np.zeros((self.p, self.m + self.n - np.size(u_mat, 1))))
            )
        elif np.size(u_mat, 1) == (self.m, self.n):
            self.mn = v_mat @ sing_inv @ u_mat.T @ f @ g_mat
            self.Nn = g_mat

        self.nu = ca.MX.sym(
            "nu", np.matlib.repmat(np.size(self.mn, 1), 1, 2), np.ones((1, 2))
        )
        self.svd_flag = True

    def _lam_contract_set(self, a, x0_poly, lam):
        """Compute the lambda-contractive( or invariant) set.

        Parameters
        ----------
        A : np.ndarray
            System matrix, used to evolve the set.
        X0 : Polytope
            Initial polytope.
        lam : float
            Lambda contraction factor (1 for invariant set).
        """

        # Check if it there is more than 1 vertex

        if len(a.shape) > 2:  # TODO - do this in a better way, separating MPC and RMPC
            v = np.size(a, 2)
            na = np.size(a, 1)
            self.poly_x_aug = x0_poly
            max_iter = 1000
            a_tmp = np.eye(np.size(a, 0))

            for k in range(max_iter):

                a_tmp = np.einsum("ik...,kj...->ij...", a_tmp, a / lam)
                xa_1 = Polytope(
                    A=np.vstack(
                        (self.poly_x_aug.A, np.einsum("ij,jk...->ik...", x0_poly.A, a_tmp).reshape(-1, na))
                    ).squeeze(),
                    b=np.vstack(
                        (self.poly_x_aug.b[:,np.newaxis], np.matlib.repmat(x0_poly.b, v, 1).reshape(-1,1))
                    ).reshape(-1, 1),
                )

                for i in range(len(xa_1.A)):
                    for j in range(len(xa_1.A[i])):
                        if xa_1.A[i, j] > INF:
                            xa_1.A[i, j] = INF
                        if xa_1.A[i, j] < -INF:
                            xa_1.A[i, j] = -INF
                xa_1.minimize_H_rep()
                # from matplotlib import pyplot as plt
                # ax,_,_= xa_1.plot()
                # plt.show()
                if self.poly_x_aug == xa_1:
                    break
                self.poly_x_aug = xa_1
            print(f"Convergence reached after {k} iterations")
        else:
            v = 1
            self.poly_x_aug = x0_poly
            max_iter = 1000
            a_tmp = np.eye(np.size(a, 0))

            for k in range(max_iter):

                a_tmp = np.einsum("ik...,kj...->ij...", a_tmp, a / lam)
                xa_1 = Polytope(
                    A=np.vstack(
                        (self.poly_x_aug.A, np.einsum("ij,jk...->ik...", x0_poly.A, a_tmp))
                    ).squeeze(),
                    b=np.concatenate(
                        (self.poly_x_aug.b, np.matlib.repmat(x0_poly.b, v, 1).squeeze())
                    ),
                )
                xa_1.minimize_H_rep()

                if self.poly_x_aug == xa_1:
                    break
                self.poly_x_aug = xa_1
            print(f"Convergence reached after {k} iterations")


    def _set_controller(self, options=None):
        """Set up and store the solver/controller callable.

        Args:
            options (dict, optional): Contains casadi options - see casadi's documentation.\
                Defaults to None.
        """
        if options is None:
            options = {}

        decision_vars = ca.vertcat(
            self.sym.x.reshape((-1, 1)), self.sym.u.reshape((-1, 1))
        )
        if self.track:
            decision_vars = ca.vertcat(
                decision_vars,
                ca.reshape(self.sym.xa, (1, -1)).T,
                ca.reshape(self.sym.ua, (1, -1)).T,
            )

        if self.svd_flag:
            decision_vars = ca.vertcat(decision_vars, ca.reshape(self.nu, (1, -1)).T)
        new_g = ca.MX()
        for val in self.g:
            print(val)
            if isinstance(val, list):
                for vval in val:
                    print(vval)
                    new_g = ca.vertcat(new_g, vval)
            else:
                new_g = ca.vertcat(new_g, val)
        self.g = new_g
        qp = {
            "x": decision_vars,
            "f": self.cost_fun,
            "g": self.g,
            "p": ca.vertcat(self.sym.x_init, ca.reshape(self.sym.r, (1, -1)).T),
        }
        self.qpsol = ca.qpsol("qpsol", self.options.solver, qp, options)

    def _warm_start(self):

        #decode decision variables and shift them to create an initial guess for the next solve
        last_idx = 0
        dec_x = self.sol["x"][self.n: (self.N + 1) * self.n].toarray()
        dec_x = np.block([[dec_x, np.zeros((self.n, 1))]])

        last_idx = (self.N + 1) * self.n
        
        dec_u = self.sol["x"][last_idx + self.m: last_idx + self.m * self.N].toarray()
        dec_u = np.block([[dec_u, np.zeros((self.m, 1))]])

        last_idx = last_idx + self.m * self.N
        
        if self.track:
            dec_xa = self.sol["x"][last_idx + self.n: last_idx + self.n * self.sym.xa.shape[1]].toarray()
            dec_xa = np.block([[dec_xa, np.zeros((self.n, 1))]])

            last_idx = last_idx + self.n * self.sym.xa.shape[1]
            
            dec_ua = self.sol["x"][last_idx + self.m: last_idx + self.m * self.sym.ua.shape[1]].toarray()
            dec_ua = np.block([[dec_ua, np.zeros((self.m, 1))]])
            
            last_idx = last_idx + self.m * self.sym.ua.shape[1]


