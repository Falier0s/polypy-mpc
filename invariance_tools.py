import numpy as np
from pycvxset import Polytope
import cvxpy as cp

from scipy.linalg import fractional_matrix_power as mat_pow

from numpy.linalg import matrix_power as mpow

class GainSynthesis:

    def __init__(self, A : np.ndarray, B : np.ndarray, K = None,
                 Q = None, R = None,
                 Z_bnd : Polytope = Polytope(), W : Polytope = Polytope(),
                 mode = 'LQR', contraction_factor : float = 1.0):

        self._F_G = (Z_bnd.A/Z_bnd.b[:, np.newaxis]).copy()

        self._gain = K

        self._state_weight = Q.copy() if Q is not None else np.eye(A.shape[0])
        # TODO - Correctly compyte the P for each vertex, currently works only on with one vertex.

        try:
            self._terminal_weight = np.linalg.inv(
                mat_pow(Q, 0.5) @ (np.eye(A.shape[0]) + B @ K) @ mat_pow(Q, -0.5)
                ) if K is not None else None
        except:
            self._terminal_weight = Q.copy() * 100

        input_bnd = Z_bnd.projection(list(range(A.shape[0]))).normalize()
        self._input_weight = R.copy() if R is not None else np.eye(B.shape[1])/(np.max(input_bnd.b)**2)
        
        A = A.copy() 
        if A.ndim < 2: 
            A = A[:, np.newaxis].copy()
        
        B = B.copy()
        if B is not None and B.ndim < 2:
            B = B[:, np.newaxis].copy()

        self.transition_matrices = np.concatenate((A, B), axis=1) if B is not None else A.copy()

        self.noise_vertices = W.V.copy()

        self.n = A.shape[0]
        self.m = B.shape[1] if B is not None else 0

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

        lmis = contractivity + stability_at_vertices + positive_definitivness + constraint_satisfaction  + noise_attenuation 
 
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

            lmip.solve(solver="CLARABEL")
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

    def compute_invariant_set(self, max_iter = 1000):
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
            
            next_poly.minimize_H_rep()
            
            if next_poly.contains(prev_poly):
                print(f"Convergence reached after {k} iterations")
                break

            prev_poly = next_poly.copy()
        return self.project_on_axis(next_poly, self._state_axes)

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