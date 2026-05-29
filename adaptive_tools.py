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
    res = np.reshape(a, (a.shape[0] * a.shape[1], b.shape[0])) @ b
    return (
        res
        if len(b.shape) > 1 and b.shape[-1] > 1
        else ca.reshape(res, a.shape[1], a.shape[0]).T
    )


class SetUpdater:
    """A simple wrapper for set-membership estimation."""

    def __init__(
        self, A_B, C, Theta: Polytope, Theta_c: Polytope, W: Polytope, E: Polytope, N=1
    ):
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
            [
                [
                    Theta.A / Theta.b[:, np.newaxis],
                    np.zeros((Theta.A.shape[0], Theta_c.A.shape[1])),
                ],
                [
                    np.zeros((Theta_c.A.shape[0], Theta.A.shape[1])),
                    Theta_c.A / Theta_c.b[:, np.newaxis],
                ],
            ]
        )

        Hw = W.A / W.b
        He = E.A / E.b

        self._h = ca.MX.sym("h", self.H.shape[0], self.N)

        self._h_k = ca.MX.sym("h_k", self.H.shape[0], 1)

        self._d_k = ca.MX.sym("d_k", self.H.shape[0], self.N)

        self._lambda = ca.MX.sym(
            "lambda", self.H.shape[0] * (self.H.shape[0] + self.H.shape[1]), self.N
        )

        self.NewTheta = Polytope(
            A=Theta.A / Theta.b[:, np.newaxis], b=np.ones((Theta.A.shape[0], 1))
        )
        self.NewTheta_c = Polytope(
            A=Theta_c.A / Theta_c.b[:, np.newaxis], b=np.ones((Theta_c.A.shape[0], 1))
        )
        self._z = ca.MX.sym("z", A_B.shape[2], 1)
        self._x = ca.MX.sym("x", C.shape[2], 1)
        self._y = ca.MX.sym("y", C.shape[1], 1)

        self.D_eval = ca.Function(
            "D_eval",
            [],
            [
                ca.blockcat(
                    [
                        [
                            ca_einsum(np.einsum("ij,...jk->i...k", Hw, A_B), _z)
                            + ca.horzcat(
                                ca.DM.ones(Hw.shape[0], 1) - Hw @ _x,
                                ca.DM.zeros(Hw.shape[0], A_B.shape[0]),
                            ),
                            ca.DM.zeros(Hw.shape[0], C.shape[0]),
                        ],
                        [
                            ca_einsum(np.einsum("ij,...jk->i...k", He, C), _x)
                            + ca.horzcat(
                                ca.DM.ones(He.shape[0], 1) - He @ _y,
                                ca.DM.zeros(He.shape[0], C.shape[0]),
                            ),
                            ca.DM.zeros(He.shape[0], A_B.shape[0]),
                        ],
                    ]
                )
            ],
        )

        self.problem = self._setmembership()

        self.theta_active, self.basis_inverses = self._extract_bases(
            Theta.V, Theta.A / Theta.b[:, np.newaxis], np.ones((Theta.A.shape[0], 1))
        )

        self.theta_c_active, self.basis_c_inverses = self._extract_bases(
            Theta_c.V,
            Theta_c.A / Theta_c.b[:, np.newaxis],
            np.ones((Theta_c.A.shape[0], 1)),
        )

    def update(
        self,
        Theta: Polytope,
        Theta_c: Polytope,
        z_prev,
        x_actual,
        y_actual,
        max_delta_th,
    ):
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

        sol = self.problem(
            p=ca.vertcat(Theta.b, Theta_c.b, z_prev, x_actual, y_actual, max_delta_th),
            lbg=self.lbg,
            ubg=self.ubg,
        )

        h_opt = sol["x"].reshape((-1, self.N))[: self.H.shape[0], :]

        h_opt_1 = h_opt[:, 0]

        th_vertices = self._get_vertices(
            h_opt[: Theta.A.shape[0], :], self.basis_inverses, self.theta_active
        )

        th_c_vertices = self._get_vertices(
            h_opt[Theta.A.shape[0] :, :], self.basis_c_inverses, self.theta_c_active
        )

        self.NewTheta.b = h_opt_1[: Theta.A.shape[0]]

        self.NewTheta_c.b = h_opt_1[Theta.A.shape[0] :]

        return self.NewTheta, self.NewTheta_c, th_vertices, th_c_vertices

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
            ca.vertcat(self.H, self.H, -self.D_eval()[:, 1:]).T,
            ca.DM.eye(self.H.shape[0]),
        ) @ self._lambda - np.matlib.repmat(self.H.reshape(-1, 1), 1, self.N)

        new_bound = (
            ca.kron(
                ca.vertcat(
                    ca.DM.ones((self.H.shape[0], 1)) + tol,
                    self._h_k
                    + ca.repmat(self.d_k, 1, self.N)
                    * np.matlib.repmat(
                        np.array(range(1, self.N + 1)), self.H.shape[0], 1
                    ),
                    self.D_eval()[:, 0],
                ).T,
                ca.DM.eye(self.H.shape[0]),
            )
            @ self._lambda
            - self._h
        )

        qp = {
            "x": ca.reshape(ca.vertcat(self._h, self._lambda), -1, 1),
            "f": (
                ca.DM.ones(1, self._h.shape[0] * self.N) @ ca.reshape(self._h, -1, 1)
            ),
            "g": ca.reshape(
                ca.vertcat(fixed_complexity, new_bound, -self._lambda), -1, 1
            ),
            "p": ca.vertcat(self._h_k, self._z, self._x, self._y, self._d_k),
        }

        self.ubg = np.zeros(qp["g"].shape[0])
        self.lbg = np.vstack(
            (
                np.zeros((fixed_complexity.shape[0], self.N)),
                -np.inf * np.ones((new_bound.shape[0] + self._lambda.shape[0], self.N)),
            )
        ).reshape(-1, 1)

        return ca.qpsol("qpsol", "osqp", qp, {})

    def _get_vertices(
        self, b: np.ndarray, basis_inverses: np.ndarray, active_bases: np.ndarray
    ) -> np.ndarray:
        """
        Computes new vertices {x : A*x <= b_new} from the precomputed bases.

        For each basis I:  v_new = A[I,:]^{-1} @ b_new[I]
        Check feasibility on all constraints and deduplicate.

        Arguments:
        ----------

        b: ndarray, shape (m,) or (m,1) or (m,N)
            The new vector b defining the polytope {x : A*x <= b}.
        basis_inverses: ndarray, shape (V, n, n)
            The precomputed inverses of the active constraint matrices for each vertex.
        active_bases: ndarray, shape (V, n)
            The indices of the active constraints for each vertex.

        Returns:
        --------

        vertices: np.ndarray shape (V, n) or (V, n, N) if b is (m,N)
            The vertices of the new polytope defined by A*x <= b.
        """
        vertices = np.einsum("ijk,ik...->ij...", basis_inverses, b[active_bases])

        return vertices.squeeze()

    def _extract_bases(self, vertices, A=None, b=None) -> tuple[np.ndarray, np.ndarray]:
        """
        For each known vertex it found the active basis (n linearly indipendent active constraints) and pre-compute the inverse matrix.

        Arguments:
        ----------

        vertices: np.ndarray shape (V, n)
        A: np.ndarray shape (m, n), optional
            The matrix A of the polytope. If not provided, it will be computed from the vertices.
        b: np.ndarray shape (m,), optional
            The vector b of the polytope. If not provided, it will be set to ones.

        Returns:
        --------

        active_bases: np.ndarray shape (V, n)
            The indices of the active constraints
        basis_inverses: np.ndarray shape (V, n, n)
            The inverse of the active constraint matrices
        """

        active_bases = []
        basis_inverses = []

        if not A:
            A = Polytope(V=vertices).A / Polytope(V=vertices).b[:, np.newaxis]
        if not b:
            b = np.ones(A.shape[0], 1)

        for v in vertices:
            # Indici dei vincoli attivi per questo vertice: A[i,:]*v ≈ b[i]
            residuals = A @ v - b
            active_idx = np.where(np.abs(residuals) < self.tol * 100)[0]

            basis_found = False
            for indices in combinations(active_idx, self.n):
                A_sub = A[list(indices), :]
                if abs(np.linalg.det(A_sub)) < self.tol:
                    continue
                active_bases.append(indices)
                basis_inverses.append(np.linalg.inv(A_sub))
                basis_found = True
                break

            if not basis_found:
                raise RuntimeWarning(f"No basis found for vertex {v}. ")

        return np.array(active_bases), np.array(basis_inverses)


class Filter:
    """A simple wrapper for parameter estimation filters."""

    def __init__(self, A_B, C, Theta: Polytope, Theta_c: Polytope, type="lms", mu=0.05):

        self.H = np.block(
            [
                [
                    Theta.A / Theta.b[:, np.newaxis],
                    np.zeros((Theta.A.shape[0], Theta_c.A.shape[1])),
                ],
                [
                    np.zeros((Theta_c.A.shape[0], Theta.A.shape[1])),
                    Theta_c.A / Theta_c.b[:, np.newaxis],
                ],
            ]
        )
        self.type = type

        self._h_k = ca.MX.sym("h_k", self.H.shape[0], 1)

        self.th_prev_estimate, self.th_c_prev_estimate = None, None

        self._z = ca.MX.sym("z", A_B.shape[2], 1)
        self._x = ca.MX.sym("x", C.shape[2], 1)
        self._y = ca.MX.sym("y", C.shape[1], 1)

        self._th_hat = ca.MX.sym("th_hat", A_B.shape[0] + C.shape[0], 1)
        self._th_prev = ca.MX.sym("th_prev", A_B.shape[0] + C.shape[0], 1)

        self._x_hat = ca.Function(
            "x_hat",
            [],
            [
                ca_einsum(self.A_B.transpose(1, 0, 2), self._z)
                @ self._th_prev[: self.A_B.shape[0]]
            ],
        )
        self._y_hat = ca.Function(
            "y_hat",
            [],
            [
                ca_einsum(self.C.transpose(1, 0, 2), self._x[: C.shape[2]])
                @ self._th_prev[self.A_B.shape[0] :]
            ],
        )

        self._th_tilde = ca.Function(
            "_th_tilde",
            [],
            [
                (
                    ca.blockcat(
                        [
                            [
                                ca_einsum(A_B[1:, :, :].transpose(1, 0, 2), _z),
                                ca.DM.zeros(A_B.shape[1], C.shape[0] - 1),
                            ],
                            [
                                ca.DM.zeros((C.shape[1], A_B.shape[0] - 1)),
                                ca_einsum(C[1:, :, :].transpose(1, 0, 2), _x),
                            ],
                        ]
                    ).T
                    @ ca.vertcat(self._x - self._x_hat(), self._y - self._y_hat())
                )
                * mu
                + self._th_prev
            ],
        )

        if self.type == "lms":
            self.problem = self._lms()

    def update(self, Theta: Polytope, Theta_c: Polytope, z_prev, x_actual, y_actual):
        """
        Wrapper function for the filter update step. It calls the appropriate filter update function based on the selected filter type in options.par_filter.

        Arguments:
        ----------
        Theta : Polytope
            The uncertainty set for the parameters of the system matrices A and B.
        Theta_c : Polytope
            The uncertainty set for the parameters of the system matrix C.
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

        if not self.th_prev_estimate or self.th_c_prev_estimate:
            self.th_prev_estimate, self.th_c_prev_estimate = self._chebyshev(
                Theta, Theta_c
            )

        if self.type == "lms":
            sol = self.problem(
                p=ca.vertcat(
                    self.th_prev_estimate,
                    self.th_c_prev_estimate,
                    z_prev,
                    x_actual,
                    y_actual,
                    Theta.b,
                ),
                lbg=-ca.inf(self.problem["g"].shape),
                ubg=ca.DM.zeros(self.problem["g"].shape),
            )

            th_hat = sol["x"][: self._th_hat.shape[0]]
            th_c_hat = sol["x"][self._th_hat.shape[0] :]
        elif self.type == "chebyshev":
            th_hat, th_c_hat = self._chebyshev(Theta, Theta_c)
        elif self.type == "rls":
            th_hat, th_c_hat = self._rls()
        elif self.type == "kalman":
            th_hat, th_c_hat = self._kalman()

        self.th_prev_estimate = th_hat
        self.th_c_prev_estimate = th_c_hat

        return th_hat, th_c_hat

    def _lms(self):
        """
        Construct the symbolic QP for the LMS filter update step.

        Returns:
        -------
        ca.Function
            A casadi function that computes the updated parameter estimates $\hat{\theta}_k$ based on the LMS update rule.
        """

        qp = {
            "x": self._th_hat,
            "f": ca.norm_2(self._th_hat - self._th_tilde()),
            "g": self.H @ self._th_hat - self._h_k,
            "p": ca.vertcat(self._th_prev, self._z, self._x, self._y, self._h_k),
        }

        return ca.qpsol("qpsol", "osqp", qp, {})

    def _chebyshev(self, Theta: Polytope, Theta_c: Polytope):
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
