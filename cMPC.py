import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse import block_diag
import casadi as ca
from pycvxset import Polytope

import os
import csv
from datetime import datetime
import cvxpy as cp

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

INF = 1e4

def cartesian_product(*polytopes):
    from itertools import product
    from pycvxset import Polytope

    """Compute the Cartesian product of multiple polytopes.

    The Cartesian product of polytopes P1, P2, ..., Pk is the set of all
    concatenated vertex combinations from each input polytope.

    Parameters
    ----------
    *polytopes : Polytope
        Input polytopes to combine.

    Returns
    -------
    Polytope
        A polytope representing the Cartesian product.
    """

    # Get vertex representations from each polytope
    V_rep = [p.V for p in polytopes]

    # Build all possible concatenations of vertices
    combinations = []
    for vertices in product(*V_rep):
        combinations.append(np.concatenate(vertices))

    # Return as a new Polytope
    return Polytope(V=np.array(combinations))


def pagemtimes(A, B):
    import numpy as np

    """
    A and B are 3D arrays with q layers
    each layer of A and B is compatible
    for matrix multiplication
    """

    for i in range(3-len(np.shape(A))):
        A = np.expand_dims(A, i+len(np.shape(A)))
    
    for i in range(3-len(np.shape(B))):
        B = np.expand_dims(B, i+len(np.shape(B)))
    

    C = np.zeros((np.shape(A)[0], np.shape(B)[1], np.shape(B)[2]))
    for i in range(np.shape(A)[2]):
        C[:, :, i] = np.matmul(A[:, :, i], B[:, :, i])
    return C


class cMPC:
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
            TODO document better the datatype
        Q : np.ndarray
            State cost matrix (quadratic weight on state deviations).
            TODO reshape or check if the weight matricies 2D array-like
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

        if options is None:
            options = {}

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

        if len(sys['A'].shape) == 2:
            sys['A'] = sys['A'][:, :, np.newaxis]

        if len(sys['B'].shape) == 2:
            sys['B'] = sys['B'][:, :, np.newaxis]

        if len(sys['C'].shape) == 2:
            sys['C'] = sys['C'][:, :, np.newaxis]

        if options.get("Ts") is None or options["Ts"] == 0:
            self.A = sys['A']
            self.B = sys['B']
            self.C = sys['C']
            self.Ts = 0
        else:
            # Build an identity tensor of shape (n,n,?); only the first slice is used.
            I = np.zeros((sys['A'].shape[0], sys['A'].shape[1], sys['A'].shape[2]))
            I[:, :, 0] = np.eye(sys['A'].shape[2])

            self.Ts = options["Ts"]
            self.A = self.Ts * sys['A'] + I
            self.B = self.Ts * sys['B']
            self.C = sys['C']

        # ---------------------------------------------------------------------
        # Prediction horizon + state/input/output dimensions
        # ---------------------------------------------------------------------
        self.N = N
        self.n = self.A.shape[0]
        self.m = self.B.shape[1]
        self.p = self.C.shape[0]

        # ---------------------------------------------------------------------
        # Decision variables (CasADi symbolic definitions)
        # ---------------------------------------------------------------------
        # x and u are horizon-length sequences of states/inputs.
        self.x = ca.MX.sym("x", self.n, self.N + 1)
        self.u = ca.MX.sym("u", self.m, self.N)
        self._x = ca.MX.sym("_x", self.n)
        self._u = ca.MX.sym("_u", self.m)
        self._z = ca.vertcat(self._x, self._u)
        
        self.x_init = ca.MX.sym("x_init", self.n)
        # z is a generic stacked (x,u) decision variable used in some formulations
        #self.z = ca.vertcat(self.x.reshape((-1, 1)), self.u.reshape((-1, 1)))
        self.z = ca.vertcat(self.x[:,:self.N],self.u)

        # Cost accumulators (may be overwritten during optimization setup)
        self.J = 0

        # ---------------------------------------------------------------------
        # Cost weighting matrices
        # ---------------------------------------------------------------------

        if Q is None:
            self.Q = np.zeros((self.n, self.n))
        else:
            self.Q = Q

        if R is None:
            self.R = np.zeros((self.m, self.m))
        else:
            self.R = R

        # Soft-term weights (small regularization terms)
        self.S = Q * 1e-4
        self.T = R * 1e-4

        # ---------------------------------------------------------------------
        # Options parsing
        # ---------------------------------------------------------------------
        self.solver = options.get("solver") or "qpoases" # Expect a string

        self.verbose = options.get("verbose") if options.get("verbose") != "" else 0 # expect a boolean 
        self.relax = options.get("relax") if options.get("relax") != "" else False # expect a boolean
        self.customJ = options.get("customJ") if options.get("customJ") != "" else False # expect a boolean

        # ---------------------------------------------------------------------
        # Tracking versus regulation setup
        # ---------------------------------------------------------------------
        # If a reference is supplied, enable tracking mode.

        if options.get("ref") is not None and options["ref"].all():
            self.track = True

            # Trajectory tracking uses a full sequence of references.
            if options["ref"].lower() in ("traj", "trajectory"):
                self.xa = ca.MX.sym("xa", self.n, self.N + 1)
                self.ua = ca.MX.sym("ua", self.m, self.N)
                self.r = ca.MX.sym("r", self.p, self.N + 1)
            else:
                # Setpoint reference tracking
                self.xa = ca.MX.sym("xa", self.n, 1)
                self.ua = ca.MX.sym("ua", self.m, 1)
                self.r = ca.MX.sym("r", self.p, 1)
        else:
            # Regulation mode (zero reference)
            self.track = False
            self.xa = np.zeros((self.n, 1))
            self.ua = np.zeros((self.m, 1))
            self.r = ca.MX.sym("r", self.p, 1)

        # ---------------------------------------------------------------------
        # Controller identity
        # ---------------------------------------------------------------------
        self.name = (
            options.get("name") or f"{type(self).__name__}_{np.random.randint(1000)}"
        )

        # ---------------------------------------------------------------------
        # Optional SVD-based reduction for tracking (reduces artificial variables)
        # ---------------------------------------------------------------------
        self.svd_flag = bool(options.get("svd") and self.track)

        # ---------------------------------------------------------------------
        # Control horizon (Nc) defaults to prediction horizon if not provided
        # ---------------------------------------------------------------------
        self.Nc = options.get("Nc") if options.get("Nc") not in (None, 0) else self.N

        # ---------------------------------------------------------------------
        # Reachability set shaping factor (sigma)
        # ---------------------------------------------------------------------
        self.sigma = (
            options.get("sigma") if options.get("sigma") not in (None, 0) else 0.95
        )

        # ---------------------------------------------------------------------
        # Terminal stabilizing gain (K) for terminal constraint
        # ---------------------------------------------------------------------
        self.K = (
            options.get("K")
            if (options.get("K") is not None and not np.all(options.get("K") == 0))
            else np.zeros((self.m, self.n))
        )

        # ---------------------------------------------------------------------
        # State / input / output constraints
        # ---------------------------------------------------------------------
        # Bounds are expected as tuples of arrays: (lower_bound, upper_bound)
        if options.get("xBound") is not None and options["xBound"] != 0:
            lxb = np.copy(options["xBound"][0])
            uxb = np.copy(options["xBound"][1])
        else:
            lxb = np.full((self.n, 1), -INF)
            uxb = np.full((self.n, 1), INF)

        if options.get("uBound") is not None and options["uBound"] != 0:
            lub = np.copy(options["uBound"][0])
            uub = np.copy(options["uBound"][1])
        else:
            lub = np.full((self.m, 1), -INF)
            uub = np.full((self.m, 1), INF)

        if options.get("yBound") is not None and options["yBound"] != 0:
            lyb = np.copy(options["yBound"][0])
            uyb = np.copy(options["yBound"][1])
        else:
            lyb = np.full((self.p, 1), -INF)
            uyb = np.full((self.p, 1), INF)

        # Initialize the MX variables storing the symbolic constraints as empty symbolic MX

        self.g = []
        self.lbg = []
        self.ubg = []

        # Polyhedral sets for constraints (X: state, U: input, Y: output)
        self.X = Polytope(
            A=np.vstack((np.eye(self.n), -np.eye(self.n))), b=np.vstack((uxb, -lxb))
        )
        self.U = Polytope(
            A=np.vstack((np.eye(self.m), -np.eye(self.m))), b=np.vstack((uub, -lub))
        )
        self.Y = Polytope(
            A=np.vstack((np.eye(self.p), -np.eye(self.p))), b=np.vstack((uyb, -lyb))
        )

        # Product set of (x,u) pairs. Used for reachability and constraint computations.
        self.Z = cartesian_product(self.X, self.U)
        D = np.zeros((self.p, self.m))
        self.Zc = Polytope(
            A=self.Y.A @ np.hstack((self.C[:, :, 0], D)),
            b=self.Y.b,
        )

        if self.track:
            # Shrink (x,u) constraint set to account for reachability margin
            self.Zs = self.sigma * self.Z
            # Intersect the reachable set with the equilibrium condition
            self.Zs.intersection_with_affine_set(
                Ae=np.hstack((self.A[:, :, 0] - np.eye(self.n), self.B[:, :, 0])),
                be=np.zeros(self.n, 1),
            ) # TODO - check if this is redundant
        else:
            self.Zs = Polytope(V=np.zeros((1,self.n+self.m)))

        self.time = 0

        self.F = []
        self.G = []

    def add_hard_constraints(self, *hConstraints):
        """Add hard constraints to the MPC problem."""

        _fg = ca.MX.sym("_fg", self.n + self.m, 1)
        # lin_constr = ca.Function("lin_constr", [self._z, _fg], [_fg.T @ self._z]) TODO deprecate

        lin_constr = ca.Function("lin_constr", [self._z, _fg], [np.matmul(_fg.T, self._z)])

        if not hConstraints:

            F_G = np.vstack(
                (
                    self.Z.A[self.Z.b < INF, :] / self.Z.b[self.Z.b < INF, np.newaxis],
                    self.Zc.A[self.Zc.b < INF, :] / self.Zc.b[self.Zc.b < INF, np.newaxis],
                )
            )
            self.F = F_G[:, : self.n ]
            self.G = F_G[:, self.n :]
            for i in range(F_G.shape[0]):
                self.g.append(
                    ca.reshape(lin_constr.map(self.N)(self.z, F_G[i, :].T), (-1, 1))
                )
                self.lbg.append([-ca.inf] * self.N)
                self.ubg.append([1] * self.N)

        else:
            for hConstraint in hConstraints:
                if isinstance(hConstraint, Polytope):
                    P = hConstraint
                    F = (P.A / P.b[:, np.newaxis])[:, : self.n + 1]
                    G = (P.A / P.b[:, np.newaxis])[:, self.n + 1 :]
                    self.F = np.vstack((self.F, F))
                    self.G = np.vstack((self.G, G))
                    F = ca.DM(F)
                    G = ca.DM(G)

                elif isinstance(hConstraint, np.ndarray):
                    if hConstraint.shape[1] > (self.m + self.n):
                        raise ValueError("Wrong size of constraints matrix!")
                    F = hConstraint[:, : self.n + 1]
                    G = hConstraint[:, self.n + 1 : self.n + 1 + self.m]
                    self.F = np.vstack((self.F, F))
                    self.G = np.vstack((self.G, G))
                    F = ca.DM(F)
                    G = ca.DM(G)

                elif isinstance(hConstraint, tuple):
                    F, G = hConstraint
                    F = ca.DM(F)
                    G = ca.DM(G)
                    self.F = np.vstack((self.F, np.array(F)))
                    self.G = np.vstack((self.G, np.array(G)))

                else:
                    raise ValueError(
                        "Check the constraints datatype, accepted types are: \n"
                        "    1) pycvxset.Politope()\n"
                        "    2) numpy.ndArray()\n"
                        "    3) tuple(F,G) as (numpy.ndarray/casadi.MX, numpy.ndarray/casadi.MX)"
                    )
                for i in range(F.shape(1)):
                    self.g.append(
                        ca.reshape(
                            lin_constr.map(self.N)(
                                self.z, np.hstack((F[i, :], G[i, :]))
                            ),
                            (-1, 1),
                        )
                    )
                    self.lbg.append([-ca.inf] * self.N)
                    self.ubg.append([1] * self.N)

                

    def add_soft_constraints(self, *sConstraints):
        """Add soft constraints (penalized violations)."""

        # TODO - Similar to the previous constraint

        pass

    def initialize(self, mode=None):
        """Prepare the controller (compute terminal sets, assemble QP)."""

        # Build hard constraints (user-defined)
        self.add_hard_constraints()

        # If no stabilizing gain is provided, compute one via LMI solver.
        if not self.K.any():
            self._stab_gain(self.A, self.B, None, mode)

        # Closed-loop matrix for terminal set computations.
        self.Ak = self.A + pagemtimes(self.B, self.K[:, :, np.newaxis])

        # Optionally reduce the number of artificial variables via SVD.
        if not self.svd_flag:
            self.Mn = np.eye((self.n + self.m))
        else:
            self._svd_decomposition()
            self.Mx = (
                np.hstack((np.eye(self.n), np.zeros((self.n, self.m)))) @ self.Mn
            )

        # Build augmented system for terminal set computation.
        self.L = np.hstack((-self.K, np.eye(self.m))) @ self.Mn
        Aa1 = np.hstack((self.Ak, pagemtimes(self.B, self.L[:,:,np.newaxis]))).squeeze()
        Aa2 = np.hstack(
            (np.zeros((np.size(self.Mn, 1), self.n)), np.eye(np.size(self.Mn, 1)))
        )
        Aa = np.vstack((Aa1, Aa2))

        # Initial polyhedron for maximal invariant set computation.
        poly_A1 = self.Z.A @ np.vstack(
            (
                np.hstack(
                    (np.eye(self.n), np.zeros((self.n, np.size(self.Mn, 1))))
                ),
                np.hstack((self.K, self.L)),
            )
        )
        self.Z.b[self.Z.b == INF] = np.inf 
        if self.track:
            X0 = Polytope(A=self.Z.A @ np.block([[np.eye(self.n), np.zeros((self.n, np.size(self.Mn, 1)))],[self.K, self.L]]), b=self.Z.b)
        # poly2 -> cartesian_product(Polytope(V=np.zeros((1,self.n))),self.Zs)
        
            X0 = X0.intersection(cartesian_product(Polytope(V=np.zeros((1,self.n))),self.Zs))
        else:
            X0 = Polytope(A=self.Z.A @ np.block([[np.eye(self.n)],[self.K]]), b=self.Z.b)
            Aa=self.A.squeeze()
        poly_A2 = np.hstack(
            (np.zeros((np.size(self.Zs.A, 0), self.n)), self.Zs.A @ self.Mn)
        )
        Poly_A = np.vstack((poly_A1, poly_A2))
        Poly_b = np.vstack((self.Z.b[:,np.newaxis], self.Zs.b[:,np.newaxis]))

        Poly = Polytope(A=Poly_A, b=Poly_b)

        # Compute invariant terminal set (lambda = 1)
        # TODO Aa should be constructed under self.track using np.blocks and np.einsum similarly as now, when not tracking Aa should be self.A
        self._lam_contract_set(Aa, X0, 1)

        # DMPC-related flag (kept for compatibility)
        self.partSol = False

        # Build cost and constraints for the underlying QP
        self._cost_fun_build()
        self._system_build()

        opt = {}  # TODO: Fill the dictionary with options for the QP solver
        self._set_controller(opt)

    def solve(self, x0, r=None):
        """Solve the MPC problem for the current state and reference."""
        if r is None:
            r = np.zeros((self.p))

        self.sol = self.qpsol(p=ca.vertcat(x0, r), lbg=self.lbg, ubg=self.ubg)

        self.u_star = self.sol['x'][(self.N+1)*self.n:(self.N+1)*self.n+self.m]

    def _cost_fun_build(self):
        """Build the MPC quadratic cost function."""

        # If Q , R, P ,S or T are None wont be used for cost construction
        _Q = ca.MX.sym("_Q", self.n, self.n)
        _R = ca.MX.sym("_R", self.m, self.m)

        quad_cost = ca.Function(
            "quad_cost",
            [self._x, self._u, _Q, _R],
            [self._x.T @ _Q @ self._x + self._u.T @ _R @ self._u],
        )

        if not self.track:
            stage_costs = ca.sum2(
                quad_cost.map(self.N)(self.x[:, : self.N], self.u, self.Q, self.R)
            )
            terminal_cost = quad_cost(
                self.x[:,-1],
                np.zeros((self.m, 1)),
                self.P,
                np.zeros((self.m, 1)),
            )
            offset_cost = 0

        else:

            stage_costs = ca.sum2(
                quad_cost.map(self.N)(
                    self.x[:, : self.N] - self.xa[:, : self.N],
                    self.u - self.ua,
                    self.Q,
                    self.R,
                )
            )
            terminal_cost = quad_cost(
                self.x[:, self.N + 1],
                np.zeros((self.m, 1)),
                self.P,
                np.zeros((self.m, 1)),
            )
            offset_cost = quad_cost.map(self.N + 1)
            (
                self.C.T @ (self.C @ self.xa - self.r),
                np.zeros((self.m, 1)),
                self.Q,
                np.zeros((self.m, self.m)),
            )

        if self.customJ is None:
            self.customJ = 0

        self.J = stage_costs + terminal_cost + offset_cost + self.customJ

    def _system_build(self):
        """Internal function for building system related constraints such as:
        - system dynamics
        - artificial setpoints dynamics/equilibrium
        - initial conditions
        - terminal conditions
        """

        step = ca.Function(
            "step", [self._x, self._u], [self.A.squeeze() @ self._x + self.B.squeeze() @ self._u]
        )

        linearK = ca.Function(
            "linearK", [self._x] , [self.K@self._x]
        )

        g_dyn = self.x[:, 1:] - step.map(self.N)(self.x[:, : self.N], self.u)  # system dynamic
        g_init = self.x[:, 0] - self.x_init  # Initial condition

        self.g.append([g_init, ca.reshape(g_dyn, (-1, 1))])
        self.lbg.append(np.zeros((self.N + 1)))
        self.ubg.append(np.zeros((self.N + 1)))
        
        # Control Horizon Constraints
        
        self.g.append(ca.reshape((self.u[:,self.Nc:self.N] - linearK.map(self.N-self.Nc)(self.x[:,self.Nc:self.N])),(-1, 1)))
        self.lbg.append(np.zeros((self.N - self.Nc)))
        self.ubg.append(np.zeros((self.N - self.Nc)))


        if self.track:
            if self.xa.shape(2) > 1:
                # Artificial trajectory
                self.g.append(
                    self.xa[:, 1:] - step.map(self.N)(self.xa[:, : self.N], self.ua)
                )
                self.lbg.append(np.zeros((self.N*self.n)))
                self.ubg.append(np.zeros((self.N*self.n)))
            # Artificial Setpoints
            if self.svd_flag:
                self.g.append(ca.vertcat(self.xa[:, -1], self.ua[:, -1], self.r[:, -1])
                              - block_diag(self.Mn, self.Nn).toarray()@self.nu)
                self.lbg.append(np.zeros((self.n+self.m+self.p)))
                self.ubg.append(np.zeros((self.n+self.m+self.p)))
                self.g.append(self.Xa.A@ca.vertcat(self.x[:, -1], self.nu))
                self.lbg.append([-ca.inf]*(len(self.Xa.b)))
                self.ubg.append(self.Xa.b)
            else:
                self.g.append(self.Xa.A@ca.vertcat(self.x[:, -1], self.xa[:, -1], self.ua[:, -1]))
                self.lbg.append([-ca.inf]*(len(self.Xa.b)))
                self.ubg.append(self.Xa.b)

            # Artificial Setpoints
            self.g.append((self.A - np.eye(self.n)@self.xa[:, -1] + self.B@self.ua[:, -1]))
            self.lbg.append(np.zeros((self.n)))
            self.ubg.append(np.zeros((self.n)))

    def _stab_gain(self, A, B, w_v=None, cost=None):
        """Compute a stabilizing state-feedback gain K via LMIs.

        Uses CVXPY to solve the LMI formulation that yields a stabilizing gain.

        The literature reference is : Linear Robust adaptive model predictive control: Computational complexity and conservatism - exended version - Appendix Kohler et al.

        """
        # Arbitrarly Small Tolerance Value for strict inequality approximation
        tol = 1e-8

        if cost is None:
            cost = "LQR"

        X = cp.Variable((self.n, self.n), symmetric=True)
        Y = cp.Variable((self.m, self.n))

        g = cp.Variable(1, "gamma")
        l = cp.Parameter(1, "lambda")
        t = cp.Parameter(1, "tau")

        v = A.shape[2]  # Number of vertices

        # Define the LMIs
        constraints = []

        c1 = []

        c2_1 = cp.hstack(
            (
                cp.vstack((X @ fractional_matrix_power(self.Q, 0.5), np.zeros((self.n,self.n)))),
                cp.vstack(
                    (
                        Y.T @ fractional_matrix_power(self.R, 0.5),
                        np.zeros((self.n, self.m)),
                    )
                ),
            )
        )

        for j in range(v):
            c1_1 = cp.hstack((X, (A[:, :, j] @ X + B[:, :, j] @ Y).T))
            c1_1 = cp.vstack((c1_1, cp.hstack((A[:, :, j] @ X + B[:, :, j] @ Y, X))))

            c1.append(
                cp.vstack(
                    (
                        cp.hstack((c1_1, c2_1)),
                        cp.hstack((c2_1.T, g * np.eye(self.n + self.m))),
                    )
                )
                >> tol
            )

        # Contractiveness

        c2 = []
        for j in range(v):
            c1_2 = cp.hstack((l * X, A[:, :, j] @ X + B[:, :, j] @ Y))
            c1_2 = cp.vstack(
                (
                    c1_2,
                    cp.hstack(((A[:, :, j] @ X + B[:, :, j] @ Y).T, l * X)),
                )
            )
            c2.append(c1_2 >> tol)
        # Constraint Satisfaction

        c3 = []

        for i in range(np.size(self.F, 0)):
            c1_3 = cp.hstack((1, self.F[i, :] @ X + self.G[i, :] @ Y))
            c2_3 = cp.vstack(
                (c1_3, cp.hstack(((self.F[i, :] @ X + self.G[i, :] @ Y)[:, np.newaxis], X)))
            )
            c3.append(c2_3 >> tol)
        
        # Noise Attenuation

        c4 = []

        w_v = np.array(w_v)

        if w_v.any():
            for j in range(v):
                for k in range(np.size(w_v, 0)):
                    c1_4 = np.hstack(
                        (
                            t * X,
                            np.zeros((self.n, 1)),
                            (A[:, :, j] @ X + B[:, :, j] @ Y)[:, np.newaxis],
                        )
                    )
                    c2_4 = cp.hstack((np.zeros((1, self.n)), 1 - t, w_v[k, :]))
                    c3_4 = cp.vstack(
                        (
                            c1_4,
                            c2_4,
                            cp.hstack(
                                (
                                    A[:, :, j] @ X + B[:, :, j] @ Y,
                                    w_v[k, :].T,
                                    X,
                                )
                            ),
                        )
                    )
                    c4.append(c3_4 >> tol)

        constraints.append(X >> tol * np.eye(self.n))
        constraints = constraints+c1+c2+c3+c4

        J = 0
        if cost == "volume":
            J = cp.Minimize(-cp.log_det(X))
        elif cost == "performance":
            J = cp.Minimize(g[0])
        elif cost == "LQR":
            J = cp.Minimize(cp.trace(X))
        
        # LMI Problem 

        LMIP = cp.Problem(J, constraints)

        t.value = [0.95]
        l.value = [0.95]

        satisfied = False

        rho = np.zeros(v)  # spectral radius for each vertex

        while not satisfied:
            LMIP.solve()
            self.P = np.linalg.inv(X.value)
            self.K = Y.value @ self.P
            for j in range(v):

                rho[j] = np.linalg.norm(
                    np.linalg.eig(A[:, :, j] + B[:, :, j] @ self.K).eigenvalues, np.inf
                )

            if (rho >= 1).any() or np.isnan(rho).any():
                l.value = l.value * 0.9
                t.value = t.value * 0.9
            elif l.value <= 1.01 * (3 + np.max(rho)) / 4:
                satisfied = True
                l.value = [np.max(rho)]
            else:
                l.value = [(3 + np.max(rho)) / 4]
                t.value = t.value * 0.99

        self.lam = l.value

    def _svd_decomposition(self):
        """Compute SVD-based transformation matrices for reduced artificial variables.

        Alvarado PhD Thesis

        https://idus.us.es/server/api/core/bitstreams/07282423-137f-4294-b314-78092d044ff6/content
        """

        E1 = np.hstack((self.A - np.eye(self.n), self.B))
        E2 = np.hstack((self.C, np.zeros((self.p, self.m))))
        E = np.vstack((E1, E2))

        F = np.vstack((np.zeros((self.n, self.p)), np.eye(self.p)))

        U, S, V = np.linalg.svd(E)
        V = (
            V.T
        )  # I don't know why but matlab and numpy give different shape of V
        S = np.diag(S)  # transform the array in a matrix
        S_inv = np.linalg.pinv(S)
        if np.size(S_inv, 1) != np.size(U, 0):
            S_inv = np.hstack((S_inv, np.zeros((np.size(S, 1), 1))))

        Vp = V[:, np.linalg.matrix_rank(S):]
        Up = U[:, np.linalg.matrix_rank(S):]

        if np.size(U, 1) == (self.p + self.n):
            G = np.eye(self.p)
        elif np.size(U, 1) < (self.p + self.n):
            G = F.T @ Up
            G = G[:][np.linalg.matrix_rank(S):]

        if np.size(U, 1) < (self.m + self.n):
            self.Mn = np.hstack((V @ S_inv @ U.T @ F @ G, Vp))
            self.Nn = np.hstack(
                (G, np.zeros((self.p, self.m + self.n - np.size(U, 1))))
            )
        elif np.size(U, 1) == (self.m, self.n):
            self.Mn = V @ S_inv @ U.T @ F @ G
            self.Nn = G

        self.nu = ca.MX.sym(
            "nu", np.matlib.repmat(np.size(self.Mn, 1), 1, 2), np.ones((1, 2))
        )
        self.svd_flag = True

    def _lam_contract_set(self, A, X0, lam):
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

        if len(A.shape) > 2:
            v = np.size(A, 2)
        else:
            v = 1

        n = np.size(A, 1)
        self.Xa = X0
        max_iter = 100 #TODO set again a value right value
        A_tmp = np.eye(np.size(A, 0))

        for k in range(max_iter):

            A_tmp = np.einsum('ik...,kj...->ij...', A_tmp, A/lam)
            Xa_1 = Polytope(A=np.vstack((self.Xa.A,
                                        np.einsum('ij,jk...->ik...',X0.A,A_tmp))).squeeze(), 
                            b= np.concatenate((self.Xa.b, np.matlib.repmat(X0.b,v,1).squeeze())))
            Xa_1.minimize_H_rep()

            if self.Xa == Xa_1:
                break
                # TODO check with a test if the algorithm can converge
            else:
                self.Xa = Xa_1
                print("Iteration:", k)

    def _set_controller(self, options=None):
        """Set up and store the solver/controller callable."""
        if options is None:
            options = {}
        
        decision_vars = ca.vertcat(self.x.reshape((-1, 1)),self.u.reshape((-1, 1)))
        if self.track:
            decision_vars = ca.vertcat(
                decision_vars, ca.reshape(self.xa, -1, 1), ca.reshape(self.ua, -1, 1)
            )

        if self.svd_flag:
            decision_vars = ca.vertcat(decision_vars, ca.reshape(self.nu, -1, 1))

        qp = {
            "x": decision_vars,
            "f": self.J,
            "g": self.g,
            "p": ca.vertcat(self.x_init, self.r),
        }
        self.qpsol = ca.qpsol("qpsol", self.solver, qp, options)

    def _warm_start(self, x_star):
        pass
