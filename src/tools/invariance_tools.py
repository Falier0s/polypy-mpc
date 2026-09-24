import itertools

import numpy as np
from pycvxset import Polytope
from pycvxset.common import constants
import cvxpy as cp

import matplotlib.pyplot as plt

from scipy.linalg import fractional_matrix_power as mat_pow

from numpy.linalg import matrix_power as mpow

from .polytope_reduction import robust_minimal_hrep, robust_contains

class GainSynthesis:

    def __init__(self, A : np.ndarray, B : np.ndarray, K = None,
                 Q = None, R = None,
                 Z_bnd : Polytope = Polytope(), W : Polytope = Polytope(),
                 mode = 'LQR', contraction_factor : float = 1.0):

        self._F_G = (Z_bnd.A/Z_bnd.b[:, np.newaxis]).copy()

        self._gain = K
    
        A = A.copy() 
        if A.ndim < 3: 
            A = A[:, :, np.newaxis].copy()
        
        B = B.copy()
        if B is not None and B.ndim < 3:
            B = B[:, :, np.newaxis].copy()

        self.transition_matrices = np.concatenate((A, B), axis=1) if B is not None else A.copy()

        self.noise_vertices = W.V.copy()

        self.n = A.shape[0]
        self.m = B.shape[1] if B is not None else 0


        self._state_weight = Q.copy() if Q is not None else np.eye(self.n)
        # TODO - Correctly compyte the P for each vertex, currently works only on with one vertex.

        try:
            self._terminal_weight = np.linalg.inv(
                mat_pow(Q, 0.5) @ (np.eye(self.n) + B[:, :, 0] @ K) @ mat_pow(Q, -0.5)
                ) if K is not None else None
        except:
            self._terminal_weight = Q.copy() * 100

        self._input_weight = np.eye(self.m)

        # if Z_bnd.dim > 0:
        #     input_bnd = Z_bnd.projection(list(range(self.n))).normalize()
        #     self._input_weight = R.copy() if R is not None else np.eye(self.m)/(np.max(input_bnd.b)**2)


        self._mode = mode
        self._lam = contraction_factor

    def synthesize_controller(self):
        """
        Return a stabilizing gain K for the system x+ = Ax + Bu, solving LMIs with cvxpy.

        The literature reference is : Linear Robust adaptive model predictive control:\
            Computational complexity and conservatism - exended version - Appendix Kohler et al.

        """

        tol = 1e-8

        num_vertices = self.transition_matrices.shape[2]

        x_mat = cp.Variable((self.n, self.n), symmetric=True)
        y_mat = cp.Variable((self.m, self.n))

        lambd = cp.Parameter(1, "lambda")
        tau = cp.Parameter((1, 1), "tau")

        contractivity = []
        diagonal = cp.bmat([[x_mat, np.zeros((self.n, self.n))],
                            [np.zeros((self.n, self.n)), x_mat]])

        for vert in range(num_vertices):
            off_diagonal = cp.bmat([[np.zeros((self.n, self.n)),
                                     (self.transition_matrices[:,:, vert] @ cp.vstack((x_mat, y_mat))).T],
                                     [np.zeros((self.n, 2*self.n))]])
            contractivity.append(lambd*diagonal + off_diagonal + off_diagonal.T >> tol)

        stability_at_vertices = []

        diagonal = cp.bmat([[diagonal, np.zeros((2*self.n, self.n + self.m))],
                            [np.zeros((self.n + self.m, 2*self.n)), np.eye(self.n + self.m)]])
        
        for vert in range(num_vertices):
            off_diagonal = cp.bmat([[np.zeros((self.n, self.n)),
                                     (self.transition_matrices[:,:, vert] @ cp.vstack((x_mat, y_mat))).T,
                                     x_mat @ mat_pow(self._state_weight,0.5),
                                     y_mat.T @ mat_pow(self._input_weight,0.5) ,],
                                     [np.zeros((2*self.n + self.m, 3*self.n + self.m))]])
            stability_at_vertices.append(diagonal + off_diagonal + off_diagonal.T >> tol)

        constraint_satisfaction = []

        for i in range(self._F_G.shape[0]):
            constraint_satisfaction.append(cp.bmat(
                [[np.array([[1]]), (self._F_G[i, :] @ cp.vstack((x_mat, y_mat)))[np.newaxis, :]],
                 [(self._F_G[i, :] @ cp.vstack((x_mat, y_mat)))[:, np.newaxis] , x_mat]]) >> tol
                 )

        noise_attenuation = []

        for k in range(self.noise_vertices.shape[0]):
            for j in range(num_vertices):
                noise_attenuation.append(cp.bmat(
                    [[tau*x_mat, np.zeros((self.n, 1)), (self.transition_matrices[:,:, j] @ cp.vstack((x_mat, y_mat))).T],
                     [np.zeros((1, self.n)), (1-tau), self.noise_vertices[k, :][np.newaxis, :]],
                     [(self.transition_matrices[:,:, j] @ cp.vstack((x_mat, y_mat))), self.noise_vertices[k, :][:, np.newaxis], x_mat]]
                ) >> tol)

        positive_definitivness = [x_mat >> tol * np.eye(self.n)]

        lmis = contractivity + stability_at_vertices + positive_definitivness + constraint_satisfaction + noise_attenuation 
 
        j_cost = 0
        if self._mode == "volume":
            j_cost = cp.Minimize(-cp.log_det(x_mat))
        elif self._mode == "LQR":
            j_cost = cp.Minimize(cp.trace(x_mat))

        # LMI Problem

        lmip = cp.Problem(j_cost, lmis)

        tau.value = np.array([[0.95]])  # Arbitrarly positive value
        lambd.value = [
            self._lam
        ]

        satisfied = False

        error = 0.0
        new_lambd = 0.0
        alpha = 0.5

        rho = np.zeros((num_vertices))  # spectral radius for each vertex

        while not satisfied:

            lmip.solve(solver=constants.DEFAULT_SDP_SOLVER_STR, verbose=False)
            if x_mat.value is None:
                raise RuntimeError(
                    f"GainSynthesis LMI (mode={self._mode!r}, "
                    f"solver={constants.DEFAULT_SDP_SOLVER_STR!r}) did not return a "
                    f"solution: cvxpy status = {lmip.status!r}. The LMI is likely "
                    "infeasible for the current (Z_bnd, W, contraction_factor) -- try "
                    "loosening the bounds/W, lowering lam, or a different SDP solver."
                )
            self._terminal_weight = np.linalg.inv(x_mat.value)
            self._gain = y_mat.value @ self._terminal_weight
            
            closed_loop_matrices = np.einsum(
                'ijk,jl->ilk', 
                self.transition_matrices,
                np.concatenate((np.eye(self.n), self._gain), axis=0)
            )

            for vert in range(num_vertices):
                rho[vert] = max(abs(np.linalg.eigvals(closed_loop_matrices[:,:, vert])))

            max_rho = np.max(rho)
            target = np.max([self._lam, max_rho + 0.7 * (1.0 - max_rho)])
            new_lambd = lambd.value[0] + alpha * (target - lambd.value[0])
            error = target - lambd.value[0]
            satisfied = np.sqrt(error**2) <= tol
            lambd.value[0] = new_lambd
            # print(f"Current lambda: {lambd.value[0]:.4f}, spectral radius: {max_rho:.4f}, error: {error:.4e}, target: {target:.4f}")

        self._lam = lambd.value[0]

    def get_terminal_weight(self):
        """Simple Getter that return the terminal weight P."""
        return self._terminal_weight
    
    def get_gain(self):
        """Simple Getter that return the gain matrix K."""
        return self._gain


class InvariantSet:

    def __init__(self, Z_bnd : Polytope, A : np.ndarray,
                 B = None, K = None, contractive_factor : float = 1.0):

        if A.ndim < 3:
            A = A[:, :, np.newaxis].copy()

        n_state = A.shape[0]

        n_vert = A.shape[2]

        if B is None:
            B = np.zeros((n_state, 0, n_vert))
            K = np.zeros((0, n_state))

        n_input = B.shape[1]

        if B.ndim < 3:  
            B = B[:, :, np.newaxis].copy()

        if K is None:
            K = np.zeros((0, n_state))

        self._transition_matrices = np.concatenate((A, B), axis=1).copy()

        _formatter = np.concatenate((np.eye(n_state + n_input - K.shape[0]),
                                     np.concatenate((K, np.zeros((K.shape[0],n_input-K.shape[0]))),
                                                    axis=1)), axis=0)

        self._transition_matrices = np.einsum("ijk,jl->ilk", self._transition_matrices, _formatter)

        self._propagator = np.concatenate((
            self._transition_matrices,
            np.concatenate((
                np.zeros((n_input-K.shape[0], n_state, n_vert)),
                np.concatenate([np.eye(n_input-K.shape[0])[:,:,np.newaxis] for _ in range(n_vert)], axis = 2)),
                axis = 1
                )), axis = 0)

        self._boundary_polytope = Polytope(
            A = (Z_bnd.A @ _formatter)/Z_bnd.b[:, np.newaxis],
            b = np.ones_like(Z_bnd.b))
        
        self._contractive_factor = contractive_factor

        self._state_axes = list(range(n_state))

    def project_on_axis(self, poly: Polytope, axis : list = []):
        """
        Provide a method for Polytope objects, complementary to Polytope.projection(),
         which performs a projection away from the provided axes.

        Arguments:
        ----------

        poly: Polytope
            The polytope to be projected.

        axis: list of int
            The axes to be projected away from. For example, if axis=[0], the projection will be performed on all axes except the first one.
            
        Return:
        -------
        Polytope
            The projected polytope.

        """

        poly_axes = list(range(poly.dim))

        # reorder the axes list backward

        axis = sorted(axis, reverse=True)

        for ax in axis:
            _ = poly_axes.pop(ax)
        
        return poly.projection(poly_axes) if len(poly_axes) > 0 else poly.copy()

    def compute_invariant_set(self, max_iter = 1000, check_mode : bool = False,
                               cmplx_red : bool = False, shape : np.ndarray = None,
                               cmplx_tol : float = 1e-6):
        """
        Compute the invariant set by iteratively propagating the boundary of the set and taking the convex hull.
        Arguments:
        ----------
        max_iter: int (optional)
            The maximum number of iterations to perform. Default is 1000.
        cmplx_red: bool (optional)
            If True, do not run the facet-growing iteration below at all --
            delegate to compute_invariant_set_reduced(), which keeps a
            FIXED number of facets throughout (no combinatorial blow-up),
            solved in one shot (no iteration, max_iter is unused in this
            path) via a Perron-Frobenius eigenvalue condition instead of
            growing one facet at a time. See that method's docstring.
            Default is False (unchanged, original behavior).
        shape: np.ndarray (optional)
            Only used when cmplx_red=True: the fixed candidate facet
            normals to pass through to compute_invariant_set_reduced()
            (its `shape` argument -- see there for the default).
        cmplx_tol: float (optional)
            Only used when cmplx_red=True: numerical tolerance passed
            through to compute_invariant_set_reduced() (its `tol` argument).

        Return:
        -------
        np.ndarray
            The A matrix of the invariant set.

        """
        if cmplx_red:
            return self.compute_invariant_set_reduced(shape=shape, tol=cmplx_tol)

        poly = self._boundary_polytope.copy()

        prev_poly = poly.copy()

        for k in range(1,max_iter+1):
            
            maps = np.concatenate([(
                poly.A/poly.b[:,np.newaxis]
                ) @ mpow(self._propagator [:, :, vert],
                     k) for vert in range(self._propagator.shape[2])],
                     0)

            next_poly = Polytope(A=np.vstack((prev_poly.A, maps)),
                                 b= np.concatenate((prev_poly.b, np.ones(maps.shape[0])*self._contractive_factor**k),0))

            if check_mode and (k % 54 == 0):
                self._plot_invariant_set_projections(next_poly, prev_poly, k)
            # cdd's double-description method is fast but exact/combinatorial:
            # the accumulating, near-degenerate facets from a marginally-
            # stable (|eig|=1) propagator -- e.g. the artificial-reference
            # block in MPC-for-tracking -- reliably trip it ("Numerical
            # inconsistency ... use GMP exact arithmetic") long before this
            # iteration converges. robust_minimal_hrep/robust_contains keep
            # cdd's speed for the well-conditioned case and fall back to an
            # LP-based check (one LP per facet, no vertex enumeration) only
            # when cdd actually fails.
            next_poly = robust_minimal_hrep(next_poly)

            if robust_contains(next_poly, prev_poly):
                prev_poly = robust_minimal_hrep(prev_poly)
                print(f"Convergence reached after {k} iterations")
                break

            prev_poly = next_poly.copy()
        return self.project_on_axis(prev_poly, self._state_axes)

    def compute_invariant_set_reduced(self, shape : np.ndarray = None,
                                       tol : float = 1e-9):
        """Fixed-complexity alternative to compute_invariant_set().

        compute_invariant_set() grows the facet count every iteration (it
        appends propagator^k @ poly.A as new rows before pruning), which is
        exactly what makes na blow up for higher (n_state + n_input)
        problems -- see the tube_inclusion() discussion this method grew
        out of. This method never grows the row count at all: it fixes the
        facet *normals* F up front (`shape`, defaulting to the boundary
        polytope's own normals -- i.e. len(shape) == Z_bnd's own facet
        count, a natural "no worse than the original constraint set"
        complexity budget) and solves directly, in one shot, for the
        scalars b of Omega = {x : F @ x <= b} that make it
        contractive-invariant, so the result is *invariant by construction*
        rather than an already-invariant polytope that gets
        post-hoc-simplified (containment alone does not preserve
        invariance -- an inner approximation would have to be reverified).

        Two earlier versions of this method were tried and rejected before
        this one, both confirmed wrong/unnecessarily conservative on actual
        test cases (not just in theory) before being replaced:
        1. "Iterate b <- min(b, one-step propagated bound using the
           CURRENT, shrinking Omega)": for a fixed shape, propagating
           through a box-like Omega scales *linearly* (homogeneously) in
           b, so that recursion is a contracting linear map in b whose
           only fixed point is b=0 -- it collapsed to a degenerate
           polytope on a small synthetic 2-state example.
        2. "Combine each vertex's Farkas certificate H_v into one H =
           elementwise max over v, then use H's Perron eigenvector":
           mathematically valid (H_v <= H elementwise and b >= 0 implies
           H_v @ b <= H @ b, so H @ b <= lam*b is a sufficient condition),
           but the elementwise max over DIFFERENT vertices' certificates
           can require components that no single vertex actually needs
           simultaneously -- on a 4-state, 16-vertex synthetic test where
           every vertex individually satisfied the invariance condition
           comfortably (per-row L1 norm ~0.6-0.85 for a target
           contraction of 0.99), the combined H had spectral radius 1.30,
           rejecting a case that is in fact trivially feasible.

        The construction actually used here fixes both: for fixed F,
        1. For every closed-loop vertex v and every row i, find (via LP,
           Farkas/LP-duality certificate -- the same style of dual
           multiplier RMPC._dual_inclusion_mat computes for tube sections,
           here used to build the base invariant set instead) the minimal
               h_i^v >= 0 with h_i^v @ F = F[i, :] @ propagator[:, :, v],
           keeping every vertex's H_v = [h_1^v; ...; h_{n_facets}^v]
           SEPARATE (no combining across vertices). This always exists as
           long as F's rows positively span the whole space (true for the
           default boundary-polytope normals, e.g. a symmetric box's
           +/-e_i rows do); if it doesn't for a custom `shape`, that
           direction genuinely cannot be certified and this raises.
        2. Solve ONE linear program for b directly against every vertex's
           own H_v simultaneously (not through a combined worst-case
           matrix, and not iteratively):
               maximize sum(b)
               s.t.  H_v @ b <= lam * b   for every vertex v
                     b <= b_cap   (support function of the ORIGINAL
                                   boundary polytope in each of F's own
                                   directions, so Omega never exceeds what
                                   Z_bnd already allowed -- same "max_x
                                   c@x over a fixed polytope" LP pattern
                                   as RMPC._maximal_disturbance)
                     b >= eps
           This is a genuine LP (H_v @ b - lam*b <= 0 is linear in b for
           fixed H_v), so cvxpy's own infeasibility detection is the
           feasibility check -- no eigenvalue theory, no ad hoc tolerance
           on a spectral radius.

        No iteration at all is needed (na stays exactly len(shape)
        throughout), and infeasibility -- this shape genuinely cannot
        support the requested contraction, e.g. because it doesn't have
        enough directions to represent some vertex's dominant growth
        direction -- is detected explicitly (LP status) instead of either
        degenerating (version 1) or over-rejecting (version 2).

        Arguments:
        ----------
        shape: np.ndarray (optional)
            The fixed candidate facet normals (rows = candidate facets,
            columns = propagator's own dimension, i.e. matching
            self._boundary_polytope.A's shape). Defaults to
            self._boundary_polytope.A itself -- the original Z_bnd
            constraint normals, i.e. "no more facets than the constraint
            set already has".
        tol: float (optional)
            Small positive floor imposed on b (b >= tol) so the returned
            polytope is never literally degenerate even when a facet's
            true optimum is at/near 0. Default is 1e-9.

        Return:
        -------
        np.ndarray
            The A matrix of the (fixed-complexity) invariant set.
        """
        F = self._boundary_polytope.A.copy() if shape is None else np.asarray(shape)
        n_facets, n_dim = F.shape
        n_vert = self._propagator.shape[2]

        # Step 1: per-vertex Farkas certificates H_v (h_i^v @ F = F[i,:] @
        # Prop_v, h_i^v >= 0), kept separate -- never combined across v.
        h_cp = cp.Variable(n_facets)
        target_param = cp.Parameter(n_dim)
        farkas_lp = cp.Problem(
            cp.Minimize(cp.sum(h_cp)), [h_cp @ F == target_param, h_cp >= 0]
        )

        H_list = []
        for v in range(n_vert):
            H_v = np.zeros((n_facets, n_facets))
            for i in range(n_facets):
                target_param.value = F[i, :] @ self._propagator[:, :, v]
                farkas_lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR, warm_start=True)
                if h_cp.value is None:
                    raise RuntimeError(
                        f"compute_invariant_set_reduced: no nonnegative combination "
                        f"of `shape`'s {n_facets} facet normals reproduces facet {i} "
                        f"propagated through vertex {v} -- `shape` does not positively "
                        f"span enough directions for this system (the default, "
                        f"self._boundary_polytope.A, always does for a symmetric box). "
                        f"Pass a richer `shape`."
                    )
                H_v[i, :] = h_cp.value
            H_list.append(H_v)

        # Step 2 (part 1): b_cap -- support function of the ORIGINAL
        # boundary polytope in each of F's own directions.
        x_cp = cp.Variable((n_dim, 1))
        c_param = cp.Parameter((1, n_dim))
        support_lp = cp.Problem(
            cp.Maximize(c_param @ x_cp),
            [self._boundary_polytope.A @ x_cp <= self._boundary_polytope.b[:, np.newaxis]],
        )
        b_cap = np.zeros(n_facets)
        for i in range(n_facets):
            c_param.value = F[i, :][np.newaxis, :]
            support_lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR, warm_start=True)
            b_cap[i] = support_lp.value

        # Step 2 (part 2): the joint LP -- every vertex's own H_v enters as
        # its own constraint block, none of them combined beforehand.
        b_cp = cp.Variable(n_facets)
        lam = self._contractive_factor
        joint_constraints = [b_cp <= b_cap, b_cp >= tol]
        joint_constraints += [H_v @ b_cp <= lam * b_cp for H_v in H_list]
        joint_lp = cp.Problem(cp.Maximize(cp.sum(b_cp)), joint_constraints)
        joint_lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR)

        if b_cp.value is None:
            raise RuntimeError(
                f"compute_invariant_set_reduced: infeasible (LP status "
                f"{joint_lp.status!r}) -- this `shape` ({n_facets} fixed facets) "
                f"cannot support contraction lam={lam:.4f} for this system "
                f"jointly across all {n_vert} vertices; pass a richer `shape`, "
                f"relax contractive_factor, or fall back to "
                f"compute_invariant_set(cmplx_red=False)."
            )
        b = np.asarray(b_cp.value).flatten()

        print(f"Reduced-complexity invariant set: {n_facets} fixed facets "
              f"(lam={lam:.4f}, {n_vert} vertices)")

        return self.project_on_axis(Polytope(A=F, b=b), self._state_axes)

    def _plot_invariant_set_projections(self, next_poly: Polytope, prev_poly: Polytope, iteration: int):
        """
        Debug visualization used by compute_invariant_set(check_mode=True).

        Polytope.plot() only supports 2D (or 3D) sets, so for an n-dimensional
        invariant set this plots every pairwise combination of axes as a 2D
        projection, laid out in a subplot matrix -- instead of assuming a
        fixed axis pair (the previous implementation called
        ``poly.projection(2)`` unconditionally, which raised a ValueError for
        any n < 3).
        """
        n = next_poly.dim
        axis_pairs = list(itertools.combinations(range(n), 2))
        if not axis_pairs:
            return

        n_cols = min(len(axis_pairs), 3)
        n_rows = -(-len(axis_pairs) // n_cols)  # ceil division
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

        for idx, (i, j) in enumerate(axis_pairs):
            ax = axs[idx // n_cols][idx % n_cols]
            away_dims = [d for d in range(n) if d not in (i, j)]
            try:
                next_poly.projection(away_dims).plot(ax, {"facecolor": "red", "alpha": 0.8})
                prev_poly.projection(away_dims).plot(ax, {"facecolor": "green", "alpha": 0.8})
            except Exception as exc:  # noqa: BLE001 - a degenerate projection should not stop the plot
                ax.set_title(f"axes ({i},{j}): {type(exc).__name__}")
                continue
            ax.set_xlabel(f"x{i}")
            ax.set_ylabel(f"x{j}")
            ax.set_title(f"axes ({i},{j})")

        for idx in range(len(axis_pairs), n_rows * n_cols):
            axs[idx // n_cols][idx % n_cols].axis("off")

        fig.suptitle(f"Invariant set iteration k={iteration}")
        plt.show()

class myPolytope(Polytope):
    """
    A simple subclass of Polytope to implement the invariant set computation method
    and the projection on axis method
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def invariant_set(self, Z_bnd : Polytope, A : np.ndarray,
                 B = None, K = None, contractive_factor : float = 1.0,
                 max_iter = 1000):

        if A.ndim < 3:
            A = A[:, :, np.newaxis].copy()

        n_state = A.shape[0]

        n_vert = A.shape[2]

        if B is None:
            B = np.zeros((n_state, 0, n_vert))
            K = np.zeros((0, n_state))

        n_input = B.shape[1]

        if B.ndim < 3:  
            B = B[:, :, np.newaxis].copy()

        if K is None:
            K = np.zeros((0, n_state))

        _transition_matrices = np.concatenate((A, B), axis=1).copy()

        _formatter = np.concatenate((np.eye(n_state + n_input - K.shape[0]),
                                     np.concatenate((K, np.zeros((K.shape[0],n_input-K.shape[0]))),
                                                    axis=1)), axis=0)

        _transition_matrices = np.einsum("ijk,jl->ilk", _transition_matrices, _formatter)

        _propagator = np.concatenate((
            _transition_matrices,
            np.concatenate((
                np.zeros((n_input-K.shape[0], n_state, n_vert)),
                np.concatenate([np.eye(n_input-K.shape[0])[:,:,np.newaxis] for _ in range(n_vert)], axis = 2)),
                axis = 1
                )), axis = 0)

        _boundary_polytope = Polytope(
            A = (Z_bnd.A @ _formatter)/Z_bnd.b[:, np.newaxis],
            b = np.ones_like(Z_bnd.b))
        
        _contractive_factor = contractive_factor

        _state_axes = list(range(n_state))

        return self._compute_invariant_set(boundary_polytope=_boundary_polytope,
                                           propagator=_propagator,
                                           state_axes=_state_axes,
                                           contractive_factor=_contractive_factor,
                                           max_iter=max_iter)

    def _compute_invariant_set(self,boundary_polytope,
                               propagator,
                               state_axes,
                               contractive_factor : float = 1.0,
                               max_iter = 1000):
        """
        Compute the invariant set by iteratively propagating the boundary of the set and taking the convex hull.
        Arguments:
        ----------
        max_iter: int (optional)
            The maximum number of iterations to perform. Default is 1000.
        
        Return:
        -------
        np.ndarray
            The A matrix of the invariant set.

        """
        
        poly = boundary_polytope.copy()

        prev_poly = poly.copy()

        for k in range(1,max_iter+1):
            
            maps = np.concatenate([(
                poly.A/poly.b[:,np.newaxis]
                ) @ mpow(propagator [:, :, vert],
                     k) for vert in range(propagator.shape[2])],
                     0)

            next_poly = Polytope(A=np.vstack((prev_poly.A, maps)),
                                 b= np.concatenate((prev_poly.b, np.ones(maps.shape[0])*contractive_factor**k),0))
            
            if next_poly.contains(prev_poly):
                print(f"Convergence reached after {k} iterations")
                break

            prev_poly = next_poly.copy()
        return self.project_on_axis(next_poly, state_axes)
    
    def project_on_axis(self, poly: Polytope, axis : list = []):
        """
        Provide a method for Polytope objects, complementary to Polytope.projection(),
         which performs a projection away from the provided axes.

        Arguments:
        ----------

        poly: Polytope
            The polytope to be projected.

        axis: list of int
            The axes to be projected away from. For example, if axis=[0], the projection will be performed on all axes except the first one.
            
        Return:
        -------
        Polytope
            The projected polytope.

        """

        poly_axes = list(range(poly.dim))

        # reorder the axes list backward

        axis = sorted(axis, reverse=True)

        for ax in axis:
            _ = poly_axes.pop(ax)
        
        return poly.projection(poly_axes) if len(poly_axes) > 0 else poly.copy()