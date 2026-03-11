import numpy as np
from scipy.linalg import fractional_matrix_power
import casadi as ca
from pycvxset import Polytope

import os
import csv
from datetime import datetime
import cvxpy as cp


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def pagemtimes(A,B):
    import numpy as np
    """
    A and B are 3D arrays with q layers
    each layer of A and B is compatible 
    for matrix multiplication
    """
    
    
    szA=np.shape(A)
    szB=np.shape(B)
    C=np.zeros((szA[0],szB[1],szB[2]))
    for i in range(szA[2]):
        C[:,:,i]=np.matmul(A[:,:,i],B[:,:,i])
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

        if options is None:
            options = {}

        # ---------------------------------------------------------------------
        # System matrices and sampling time handling
        # ---------------------------------------------------------------------
        # If a sampling time is provided, assume sys.A/sys.B are continuoustime
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

        if len(sys.A.shape) == 2:
            sys.A = sys.A[ :, :, np.newaxis]

        if len(sys.B.shape) == 2:
            sys.B = sys.B[ :, :, np.newaxis]

        if len(sys.C.shape) == 2:
            sys.C = sys.C[:, :, np.newaxis]

        if options.get("Ts") is None or options["Ts"] == 0:
            self.A = sys.A
            self.B = sys.B
            self.C = sys.C
            self.Ts = 0
        else:
            # Build an identity tensor of shape (n,n,?); only the first slice is used.
            I = np.zeros((sys.A.shape[0], sys.A.shape[1], sys.A.shape[2]))
            I[:,:,0] = np.eye(sys.A.shape[2])

            self.Ts = options["Ts"]
            self.A = self.Ts * sys.A + I
            self.B = self.Ts * sys.B
            self.C = sys.C

        # ---------------------------------------------------------------------
        # Prediction horizon + state/input/output dimensions
        # ---------------------------------------------------------------------
        self.N = N
        self.n = len(self.A.shape[1])
        self.m = len(self.B.shape[2])
        self.p = len(self.C.shape[1])

        # ---------------------------------------------------------------------
        # Decision variables (CasADi symbolic definitions)
        # ---------------------------------------------------------------------
        # x and u are horizon-length sequences of states/inputs.
        self.x = ca.MX.sym("x", np.matlib.repmat(self.n, 1, self.N), np.ones((1, self.N)))
        self.u = ca.MX.sym("u", np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))

        # z is a generic stacked (x,u) decision variable used in some formulations
        self.z = ca.MX.sym("z", self.n + self.m, 1)

        # Cost accumulators (may be overwritten during optimization setup)
        self.J = 0

        # ---------------------------------------------------------------------
        # Cost weighting matrices
        # ---------------------------------------------------------------------
        self.Q = Q
        self.R = R
        # Soft-term weights (small regularization terms)
        self.S = Q * 1e-4
        self.T = R * 1e-4

        # ---------------------------------------------------------------------
        # Options parsing
        # ---------------------------------------------------------------------
        self.solver = options.get("solver") or "qpoases"

        self.verbose = options.get("verbose") if options.get("verbose") != "" else 0
        self.relax = options.get("relax") if options.get("relax") != "" else False

        # ---------------------------------------------------------------------
        # Tracking versus regulation setup
        # ---------------------------------------------------------------------
        # If a reference is supplied, enable tracking mode.
        if options.get("ref") is not None and options["ref"].all():
            self.track = True

            # Trajectory tracking uses a full sequence of references.
            if options["ref"].lower() in ("traj", "trajectory"):
                self.xa = ca.MX.sym("xa", self.n, 1, self.N + 1)
                self.ua = ca.MX.sym("ua", self.m, 1, self.N)
                self.xr = ca.MX.sym("xr", self.n, 1, self.N + 1)
                self.ur = ca.MX.sym("ur", self.m, 1, self.N)
                self.r = ca.MX.sym("r", self.p, 1)
            else:
                # Setpoint reference tracking
                self.xa = ca.MX.sym("xa", self.n, 1)
                self.ua = ca.MX.sym("ua", self.m, 1)
                self.xr = ca.MX.sym("xr", self.n, 1)
                self.ur = ca.MX.sym("ur", self.m, 1)
                self.r = ca.MX.sym("r", self.p, 1)
        else:
            # Regulation mode (zero reference)
            self.track = False
            self.xa = np.zeros((self.n, 1))
            self.ua = np.zeros((self.m, 1))
            self.xr = np.zeros((self.n, 1))
            self.ur = np.zeros((self.m, 1))
            self.r = ca.MX.sym("r", self.p, 1)

        # ---------------------------------------------------------------------
        # Controller identity
        # ---------------------------------------------------------------------
        self.name = options.get("name") or f"{type(self).__name__}_{np.random.randint(1000)}"

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
        self.sigma = options.get("sigma") if options.get("sigma") not in (None, 0) else 0.95

        # ---------------------------------------------------------------------
        # Terminal stabilizing gain (K) for terminal constraint
        # ---------------------------------------------------------------------
        self.K = options.get("K") if (options.get("K") is not None and not np.all(options.get("K") == 0)) else np.zeros((self.m, self.n))

        # ---------------------------------------------------------------------
        # State / input / output constraints
        # ---------------------------------------------------------------------
        # Bounds are expected as tuples: (lower_bound, upper_bound)
        if options.get("xBound") is not None and options["xBound"] != 0:
            lxb = np.copy(options["xBound"][0])
            uxb = np.copy(options["xBound"][1])
        else:
            lxb = np.full((self.n, 1), -np.inf)
            uxb = np.full((self.n, 1), np.inf)

        if options.get("uBound") is not None and options["uBound"] != 0:
            lub = np.copy(options["uBound"][0])
            uub = np.copy(options["uBound"][1])
        else:
            lub = np.full((self.m, 1), -np.inf)
            uub = np.full((self.m, 1), np.inf)

        if options.get("yBound") is not None and options["yBound"] != 0:
            lyb = np.copy(options["yBound"][0])
            uyb = np.copy(options["yBound"][1])
        else:
            lyb = np.full((self.p, 1), -np.inf)
            uyb = np.full((self.p, 1), np.inf)

        # Polyhedral sets for constraints (X: state, U: input, Y: output)
        self.X = Polytope(A=np.vstack((np.eye(self.n), -np.eye(self.n))),
                          b=np.vstack((uxb, -lxb)))
        self.U = Polytope(A=np.vstack((np.eye(self.m), -np.eye(self.m))),
                          b=np.vstack((uub, -lub)))
        self.Y = Polytope(A=np.vstack((np.eye(self.p), -np.eye(self.p))),
                          b=np.vstack((uyb, -lyb)))

        # Product set of (x,u) pairs. Used for reachability and constraint computations.
        self.Z = cartesian_product(self.X, self.U)

        self.Zc = Polytope(A=self.Y.A @ np.hstack((self.C[:,:,0],
                                                   np.zeros((self.p, self.m)))), b=self.Y.b)

        if self.track:
            # Shrink (x,u) constraint set to account for reachability margin
            self.Zs = self.sigma * self.Z
            # Intersect the reachable set with the equilibrium condition
            self.Zs.intersection_with_affine_set(
                Ae=np.hstack(self.A[:,:,0] - np.eye(self.n), self.B[:,:,0]),
                be=np.zeros(self.n, 1),
            )

        self.time = 0

    def add_hard_constraints(self, *hConstraints):
        """Add hard constraints to the MPC problem."""

        pass

    def add_soft_constraints(self, *sConstraints):
        """Add soft constraints (penalized violations)."""

        pass

    def initialize(self, mode):
        """Prepare the controller (compute terminal sets, assemble QP)."""

        # Build hard constraints (user-defined)
        self.add_hard_constraints()

        # If no stabilizing gain is provided, compute one via LMI solver.
        if not self.K.any():
            self._stab_gain(self.A, self.B, [], mode)

        # Closed-loop matrix for terminal set computations.
        self.Ak = self.A + self.B @ self.K

        # Optionally reduce the number of artificial variables via SVD.
        if not self.svd_flag:
            self.Mn = np.eye((self.n + self.m))
        else:
            self._svd_decomposition()
            self.Mx = np.column_stack((np.eye(self.n),
                                       np.zeros((self.n, self.m)))) @ self.Mn

        # Build augmented system for terminal set computation.
        self.L = np.column_stack((-self.K, np.eye(self.m))) @ self.Mn
        Aa1 = np.column_stack((self.Ak, self.B @ self.L))
        Aa2 = np.column_stack((np.zeros((np.size(self.Mn, 1), self.n)),
                               np.eye(np.size(self.Mn, 1))))
        Aa = np.vstack((Aa1, Aa2))

        # Initial polyhedron for maximal invariant set computation.
        poly_A1 = self.Z.A @ np.vstack(
            (
                np.column_stack((np.eye(self.n), np.zeros((self.n,
                                                           np.size(self.Mn, 1))))),
                np.column_stack((self.K, self.L)),
            )
        )
        poly_A2 = np.column_stack((np.zeros((np.size(self.Z.A, 0), self.n)),
                                   self.Zs.A @ self.Mn))
        Poly_A = np.vstack((poly_A1, poly_A2))
        Poly_b = np.vstack((self.Z.b, self.Zs.b))

        Poly = Polytope(A=Poly_A, b=Poly_b)

        # Compute invariant terminal set (lambda = 1)
        self._lam_contract_set(Aa, Poly, 1)

        # DMPC-related flag (kept for compatibility)
        self.partSol = False

        # Build cost and constraints for the underlying QP
        self._cost_fun_build()
        self._constraint_build()

        opt = {}  # TODO: Fill the dictionary with options for the QP solver
        self._set_controller(x0=None, r=None, opt=opt)

    def solve(self, x0, r=None):
        """Solve the MPC problem for the current state and reference."""

        self.xo = x0
        if r is None:
            r = 0

        # NOTE: ``self.controller`` is expected to be created in _set_controller
        Sol, _, _, _, _, Data = self.controller(self.x0, r)  # TODO: confirm API
        self._warm_start(Sol)

        # Store diagnostics
        self.time = Data.solvertime  # TODO: confirm attribute name
        self.u_star = Sol[0]  # first input in solution
        self.J_star = Sol[1]  # optimal cost
        self._verbose_print()

    def _cost_fun_build(self):
        """Build the MPC quadratic cost function."""

        pass

    def _constraint_build(self):
        """Build constraints (dynamics, bounds, terminal set) for the MPC QP."""

        pass

    def _stab_gain(self, A, B, w_v, cost):
        """Compute a stabilizing state-feedback gain K via LMIs.

        Uses CVXPY to solve the LMI formulation that yields a stabilizing gain.
        
        The literature reference is : Linear Robust adaptive model predictive control: Computational complexity and conservatism - exended version - Appendix Kohler et al.
        
        """
        # Arbitrarly Small Tolerance Value for strict inequality approximation
        tol = 1e-8

        X = cp.Variable((self.n, self.n), symmetric=True)
        Y = cp.Variable((self.m, self.n))

        g = cp.Variable(1, "gamma")
        l = cp.Parameter(1, "lambda")
        t = cp.Parameter(1, "tau")

        v = A.shape[0] # Number of vertices

        # Define the LMIs

        c1 = []

        c2_1 = np.hstack((np.vstack((X@fractional_matrix_power(self.Q, .5),
                                     np.zeros(self.n))),
                          np.vstack((Y.transpose()@fractional_matrix_power(self.R, .5),
                                     np.zeros((self.n, self.m))))))
        for j in range(v):
            c1_1 = np.hstack((X, (A[:,:,j] @ X + B[:,:,j] @ Y).transpose()))
            c1_1 = np.vstack((c1_1, np.hstack((A[:,:,j] @ X + B[:,:,j] @ Y, X))))

            c1.append(np.vstack(
                (np.hstack((c1_1, c2_1)),
                 np.hstack((c2_1.transpose(),
                            g*np.eye(self.n+self.m))))) >> tol)

        # Contractiveness

        c2 = []
        for j in range(v):
            c1_2 = np.hstack((l*X, A[:,:,j] @ X + B[:,:,j] @ Y))
            c1_2 = np.vstack(
                (c1_2, np.hstack(((A[:,:,j] @ X + B[:,:,j] @ Y).transpose(), l*X))))
            c2.append(c1_2 >> tol)

        # Constraint Satisfaction

        c3 = []
        Fbar = np.vstack((self.F, self.Fc))
        Gbar = np.vstack((self.G,
                          (np.zeros((np.size(self.Fc, 0),
                                     np.size(self.G, 1))))))

        for i in range(np.size(Fbar, 0)):
            c1_3 = np.hstack((1, Fbar[i,:]@X+Gbar[i,:]@Y))
            c2_3 = np.vstack((c1_3,
                              np.hstack(((Fbar[i,:]@X+Gbar[i,:]@Y).transpose(), X))))
            c3.append(c2_3 >> tol)

        # Noise Attenuation

        c4 = []

        if w_v.any():
            for j in range(v):
                for k in range(np.size(w_v, 0)):
                    c1_4 = np.hstack((t*X, np.zeros((self.n, 1)),
                                      (A[:,:,j] @ X + B[:,:,j] @ Y).transpose()))
                    c2_4 = np.hstack((np.zeros((1, self.n)), 1-t, w_v[k,:]))
                    c3_4 = np.vstack((c1_4, c2_4,
                                      np.hstack((A[:,:,j] @ X + B[:,:,j] @ Y,
                                                 w_v[k,:].transpose(), X))))

                    c4.append(c3_4 >> tol)

        constraints = [X >> tol*np.eye(self.n), c1, c2, c3, c4]

        if cost == 'volume':
            J = cp.Minimize(-cp.log_det(X))
        elif cost == 'performance':
            J = cp.Minimize(g[0])
        elif cost == 'LQR':
            J = cp.Minimize(cp.trace(X))

        LMIP = cp.Problem(J, constraints, solver='mosek')

        t.value = [0.95]
        l.value = [0.95]

        satisfied = False

        rho = np.zeros(v)  # spectral radius for each vertex

        while not satisfied:
            LMIP.solve()
            self.P=np.linalg.inv(X.value)
            self.K=Y.value @ self.P
            for j in range(v):

                rho[j] = np.linalg.norm(
                    np.linalg.eig(A[:,:,j]+B[:,:,j]@self.K), np.inf)

            if (rho >= 1).any() or np.isnan(rho).any():
                l.value = l.value * .9
                t.value = t.value * .9
            elif l.value <= 1.01 * (3+np.max(rho))/4:
                satisfied = True
                l.value = [np.max(rho)]
            else:                
                l.value = [(3+np.max(rho))/4]
                t.value = t.value * .99

        self.lam = l.value

    def _svd_decomposition(self):
        """Compute SVD-based transformation matrices for reduced artificial variables.
        
        Alvarado PhD Thesis
        
        https://idus.us.es/server/api/core/bitstreams/07282423-137f-4294-b314-78092d044ff6/content
        """

        pass

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

        # Check if it there are more than 1 evaluation of the system matrices at uncertainty vertices

        if len(A.shape)>2:
            v=np.size(A,2)
        else:
            v=1

        n = np.size(A,1)
        self.Xa=X0
        max_iter = 2500
        A_tmp = np.eye(np.size(A,0))

        for k in range(max_iter):

            A_tmp = pagemtimes(A_tmp, A/lam)
            Xa_1 = Polytope(A=np.vstack((self.Xa.A,np.reshape(pagemtimes(X0.A,A_tmp.transpose(0,2,1)),(-1,n)))),b=np.vstack((self.Xa.b,np.matlib.repmat(X0.b[:,np.newaxis],v,1)))).minimize_H_rep()
            if Xa_1 == Xa_1:
                break
            else:
                self.Xa=Xa_1
                print("Iteration: %d",k)
            



    def _set_controller(self, x0, r=None):
        """Set up and store the solver/controller callable."""

        pass

    def _verbose_print(self):
        """Print solver results if verbosity is enabled."""

        pass

    def _save_results(self, results, filename=None):
        """Persist solution/results to disk (CSV or similar)."""

        pass

    def _warm_start(self, previous_solution):
        """Warm-start solver with previous solution to accelerate convergence."""

        pass
