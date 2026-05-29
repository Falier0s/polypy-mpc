"""
**adaptive_tools.py**
This module contains the implementation of the Filter and SetUpdater classes,
which are used for parameter estimation and set-membership ID.
"""
import numpy as np

from pycvxset import Polytope

import casadi as ca

def ca_einsum(a: np.ndarray, b: ca.MX):
    """
    Perform the equivalent of np.einsum('ijk,kl->ijl', a, b) between a 3D np.ndarray and a casadi.MX object.
    Arguments:
    ----------
    a: 3D np.ndarray
        The first input array.
    b: casadi.MX
        The second input array.
    Returns:
    --------
    np.ndarray
        The result of the einsum operation.
    """
    res = np.reshape(a,(a.shape[0]*a.shape[1],b.shape[0])) @ b
    return res if len(b.shape)>1 and b.shape[-1]>1 else ca.reshape(res,a.shape[1],a.shape[0]).T    

class SetUpdater:
    """A simple wrapper for set-membership estimation."""

    def __init__(self, A_B, C, Theta: Polytope, Theta_c: Polytope, W: Polytope, E: Polytope, N=1):
        """
        Initialize the SetUpdater class.

        Arguments:
        ----------
        A_B: ndarray
            The concatenated system matrices A and B, with shape (q, n, n+m) where q is the number of parameters for the system matrices.
        C: ndarray
            The output matrix C, with shape (q_c, p, n) where q_c is the number of parameters for the output matrix.
        Theta: Polytope
            The uncertainty set for the parameters of the system matrices A and B.
        Theta_c: Polytope
            The uncertainty set for the parameters of the output matrix C.
        W: Polytope
            The Process Noise Set.
        E: Polytope
            The measurement Noise Set.
        N: int, optional
            The number of steps for propagating the tube expansion
        """

        self.N = N
        
        self.H = np.block(
            [[Theta.A/Theta.b, np.zeros((Theta.A.shape[0], Theta_c.A.shape[1]))],
             [np.zeros((Theta_c.A.shape[0], Theta.A.shape[1])), Theta_c.A/Theta_c.b]]
             )

        Hw = W.A/W.b
        He = E.A/E.b
        
        self._h = ca.MX.sym(
            "h",
            self.H.shape[0],
            self.N
            )

        self._h_k = ca.MX.sym(
            "h_k",
            self.H.shape[0],
            1
            )
        
        self._d_k = ca.MX.sym(
            "d_k",
            self.H.shape[0],
            self.N
            )

        self._lambda = ca.MX.sym(
            "lambda",
            self.H.shape[0]*(self.H.shape[0]+self.H.shape[1]),
            self.N
            )
        
        self._z = ca.MX.sym("z", A_B.shape[2], 1)
        self._x = ca.MX.sym("x", C.shape[2], 1)
        self._y = ca.MX.sym("y", C.shape[1], 1)

        self.D_eval = ca.Function(
            'D_eval',
            [],
            [ca.blockcat(
                [[ca_einsum(
                    np.einsum('ij,...jk->i...k', Hw, A_B),
                    _z) + ca.horzcat(
                        ca.DM.ones(Hw.shape[0], 1) - Hw @ _x,
                        ca.DM.zeros(Hw.shape[0], A_B.shape[0])),
                        ca.DM.zeros(Hw.shape[0], C.shape[0])],
                        [ca_einsum(
                            np.einsum('ij,...jk->i...k', He, C),
                            _x) + ca.horzcat(
                                ca.DM.ones(He.shape[0], 1) - He @ _y,
                                ca.DM.zeros(He.shape[0], C.shape[0])),
                                ca.DM.zeros(He.shape[0], A_B.shape[0])]])
                                ]
                                )
        
        self.problem = self._setmembership()

        self._generate_vertex_map(Theta, Theta_c)

    def update(self, Theta: Polytope, Theta_c: Polytope, z_prev, x_actual, y_actual, max_delta_th):
        """
        Update the uncertainty set using the set-membership estimation method.
        Arguments:
        ----------
        Theta : Polytope
            The uncertainty set for the parameters of the system matrices A and B.
        Theta_c : Polytope
            The uncertainty set for the parameters of the output matrix C.
        z_prev : ndarray
            The previous input measurement.
        x_actual : ndarray
            The actual state measurement/estimate.
        y_actual : ndarray
            The actual output measurement.
        max_delta_th : float
            The maximum allowed change in the uncertainty set.
        Returns:
        -------

        theta_updated: Polytope
            The updated uncertainty set for the parameters of the system matrices A and B.
        theta_c_updated: Polytope
            The updated uncertainty set for the parameters of the output matrix C.
        th_vertices: ndarray
            The vertices of the updated uncertainty set for the parameters of the system matrices A and B propagated along N.
        th_c_vertices: ndarray
            The vertices of the updated uncertainty set for the parameters of the output matrix C,
            propagated along N.

        """

        sol = self.problem(p=ca.vertcat(Theta.b, Theta_c.b, z_prev, x_actual, y_actual, max_delta_th), lbg=self.lbg, ubg=self.ubg)

        h_opt = sol['x'].reshape((-1, self.N))[:self.H.shape[0], :]

        h_opt_1 = h_opt[:, 0]
        th_vertices, th_c_vertices = self._compute_vertices(h_opt)

        theta_updated = Theta
        theta_updated.b = h_opt_1[:Theta.A.shape[0]]

        theta_c_updated = Theta_c
        theta_c_updated.b = h_opt_1[Theta.A.shape[0]:]

        return theta_updated, theta_c_updated, th_vertices, th_c_vertices

    def _setmembership(self, tol=0.2):
        """
        Construct the symbolic LP for the simple or parameter varying set-membership identification of the uncertainty set.

        Arguments:
        tol: float
            The tolerance for constraints relaxation on the initial set. This is introduced to avoid faulty approximation in the initial set.

        Returns:
        --------
        casadi.Function
            The casadi function representing the set-membership linear program.
        """

        
        fixed_complexity = ca.kron(
            ca.vertcat(self.H, self.H,-self.D_eval()[:,1:]).T,
            ca.DM.eye(self.H.shape[0])
            ) @ self._lambda - np.matlib.repmat(self.H.reshape(-1,1),1,self.N)

        new_bound = ca.kron(
            ca.vertcat(ca.DM.ones((self.H.shape[0],1))+tol,
                       self._h_k + ca.repmat(self.d_k,1,self.N) * np.matlib.repmat(
                           np.array(range(1,self.N+1)),self.H.shape[0], 1),
                       self.D_eval()[:,0]).T,
            ca.DM.eye(self.H.shape[0])
            ) @ self._lambda - self._h

        qp = {
            'x': ca.reshape(ca.vertcat(self._h,self._lambda),-1,1),
            'f': (ca.DM.ones(1,self._h.shape[0]*self.N) @ ca.reshape(self._h,-1,1)),
            'g': ca.reshape(ca.vertcat(fixed_complexity, new_bound, -self._lambda),-1,1),
            'p': ca.vertcat(self._h_k, self._z, self._x, self._y, self._d_k)
        }

        self.ubg = np.zeros(qp['g'].shape[0])
        self.lbg = np.vstack(
            (np.zeros((fixed_complexity.shape[0], self.N)),
            -np.inf * np.ones((new_bound.shape[0] + self._lambda.shape[0], self.N)))
            ).reshape(-1,1)

        return ca.qpsol('qpsol', 'osqp', qp, {})
    
    def _compute_vertices(self):
        pass

    def _generate_vertex_map(self, *sets: Polytope):
        """
        Computes a map of the vertex of the sets offline such that every set update does not require recomputing the vertices of the sets.
        
        Arguments:
        ----------
        sets: iterable of Polytope
            The list of sets for which the vertex map needs to be computed.
        """
        pass
            

class Filter:
    """A simple wrapper for parameter estimation filters."""

    def __init__(self, A_B, C, type='lms'):

        self.A_B = A_B
        self.C = C
        self.type = type
        
    def update(self, Theta: Polytope, z_prev, y_prev, th_prev_estimate, th_c_prev_estimate, mu=0.05):
        """
        Wrapper function for the filter update step. It calls the appropriate filter update function based on the selected filter type in options.par_filter.

        Arguments:
        ----------
        Theta : Polytope
            The uncertainty set for the parameters of the system matrices A and B.
        z_prev : ndarray
            The previous (x,u) estimates from measurements.
        y_prev : ndarray
            The previous output measurements.
        th_prev_estimate : ndarray
            The previous estimate for the parameters of the system matrices A and B.
        th_c_prev_estimate : ndarray
            The previous estimate for the parameters of the system matrix C.
        mu : float, optional
            The learning rate for the LMS filter. Default is 0.05.
        Returns:
            tuple: Updated estimates for th_hat and th_c_hat.
        """

        param = (Theta, z_prev, y_prev, th_prev_estimate, th_c_prev_estimate)
        if self.type == 'lms':
            return self._lms(param, mu=mu)
        elif self.type == 'chebyshev':
            return self._chebyshev()
        elif self.type == 'rls':
            return self._rls()
        elif self.type == 'kalman':
            return self._kalman()
        else:
            raise ValueError(f"Unsupported filter type: {self.type}")

    def _lms(self, params, mu=0.05):
        """
        Update the parameter estimates using the Least Mean Squares (LMS) filter.

        $$\hat{\theta}_k = \mathcal{\Pi}(\tilde{\theta}_k)$$
        where $\mathcal{\Pi}(\tilde{\theta}_k) = arg\min_{\theta \in \Theta} \lvert \theta - \tilde{\theta}_k \rvert$ and $\tilde{\theta}_k = \hat{\theta}_{k-1} - \mu D^T (z_{k-1} - \hat{x}_{k-1})$ and $\hat{x}_{k-1} = \hat{\theta}_{k-1}^T (A_B^T z_{k-1})$.

        Returns:
        -------
        th_hat: ndarray
            Updated estimate for the parameters of the system matrices A and B.
        th_c_hat: ndarray
            Updated estimate for the parameters of the system matrix C.
        """

        
        Theta, z_prev, y_prev, th_prev_estimate, th_c_prev_estimate = params

        x_estimate = th_prev_estimate.T @ (self.A_B.transpose(1,0,2) @ z_prev)

        th_tilde = th_prev_estimate - mu * self.D.T @ (z_prev[:self.A_B.shape[0]] - x_estimate)

        th_C_tilde = th_c_prev_estimate - mu * self.Dc.T @ (y_prev - self.C.transpose(1,0,2) @ x_estimate)
        
        th_hat, _ = Theta.project(th_tilde)
        th_c_hat, _ = Theta_c.project(th_C_tilde)

        return th_hat, th_c_hat
        

    def _chebyshev(self):
        """
        Update the parameter estimates using the Chebyshev centering method.

        

        Returns:
        -------
        th_hat: ndarray
            Updated estimate for the parameters of the system matrices A and B.
        th_c_hat: ndarray
            Updated estimate for the parameters of the system matrix C.
        """
        
        th_hat, _ = Theta.chebyshev_centering()
        th_c_hat, _ = Theta_c.chebyshev_centering()

        return th_hat, th_c_hat

    def _rls(self):
        pass

    def _kalman(self):
        pass


    