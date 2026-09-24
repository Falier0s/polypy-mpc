"""A Robust MPC using CasADI."""

import numpy as np
import casadi as ca
import cvxpy as cp

from pycvxset import Polytope

from .rmpc import RMPC

from .tools.adaptive_tools import Filter, SetUpdater, initial_point_estimate



class RAMPC(RMPC):
    """
    A Robust Adaptive MPC using CasADI.

    This class implements a Robust Adaptive MPC problem using CasADI for optimization. It inherits from the RMPC class and adds functionality for parameter estimation and constraint tightening to handle uncertainties in the system.
    Attributes
    ----------
    th_hat : np.ndarray
        The estimated parameters for the system matrices A and B.
    th_c_hat : np.ndarray
        The estimated parameters for the output matrix C.
    z_1 : np.ndarray
        The previous value of the parameter estimation error, used for the filter update.
    """

    def __init__(self, system, Q, R, N, options):
        """
        Constructs the Robust Adaptive MPC problem.

        Parameters
        ----------
        system : PolytopeSystem
            The system to be controlled.
        Q : np.ndarray
            The state cost matrix.
        R : np.ndarray
            The control cost matrix.
        N : int
            The prediction horizon.
        options : dict
            The options for the MPC solver.

        Summary
        -------
        The Robust Adaptive MPC Constructor adopts the superClass constructor from RMPC and then initializes the variables for the parameter estimation and the constraint tightening.
        """
        super().__init__(system, Q, R, N, options)

        length = 1

        if self.options.lpv_flag:
            length += N

        if self.options.E is not None and isinstance(self.options.E, Polytope):
            self.E = self.options.E
        else:
            self.E = Polytope(
                A=np.vstack((np.eye(self.p), -np.eye(self.p))),
                b=np.concatenate((np.ones(self.p) * 0.01, np.ones(self.p) * 0.01)),
            )

        self.max_delta_th = self.options.max_delta_th

        self.sym.create_parameter_variables(
            self.q, self.q_c, self.vertices_number, self.c_vertices_number, length
        )
        
        ab_ = np.concatenate(
            (self.sys.A, self.sys.B),
            axis=1).transpose(2, 0, 1)
        
        c = self.sys.C.transpose(2, 0, 1)

        self.filter = Filter(
            ab_, c, self.theta, self.theta_c, self.options.par_filter,
            forgetting=self.options.forgetting, vff=False,
            th_hat_0=self.options.th_hat_0, th_c_hat_0=self.options.th_c_hat_0,
        )

        self.param_set_learn = SetUpdater(
            ab_, c, self.theta, self.theta_c, self.W, self.E, length,
            set_representation=self.options.set_representation,
        )

        self.z_prev = None
        self.th_hat, self.th_c_hat = None, None
        # Bounds (h of {H th <= h}) of the current parameter sets, carried from one solve() to the
        # next in the LTI case (max_delta_th == 0): the set-membership update is recursive.
        self.theta_b_hat, self.theta_c_b_hat = None, None

    def solve(self, x0, y0, r=None):

        if self.z_prev is None:
            self.z_prev = np.block([x0.toarray().squeeze(), np.zeros(self.m)])

        # self.theta_vertices = self.theta_vertices, self.theta_vertices
        # self.theta_vertices

        # Prior of the set-membership LP (h_0):
        #  - LTI (max_delta_th == 0): the set only shrinks, so the prior is the set of the previous
        #    step (Delta_k = Delta_{k-1} cap non-falsifying set); restarting from Theta_0 at every
        #    step would keep only the latest innovation and lose the accumulated information.
        #  - LPV (max_delta_th > 0): the LP dilates the prior by max_delta_th internally, so
        #    feeding the previous set back would compound the dilation at every step: restart
        #    from the original Theta_0 (the cap of the LP), as before.
        if self.max_delta_th or self.theta_b_hat is None:
            theta_prior, theta_c_prior = self.theta.b, self.theta_c.b
        else:
            theta_prior, theta_c_prior = self.theta_b_hat, self.theta_c_b_hat

        theta_b, theta_c_b, self.theta_vertices, self.theta_c_vertices = (
            self.param_set_learn.update(
                theta_prior, theta_c_prior, self.z_prev,
                x0.toarray(), y0.toarray(),
                self.max_delta_th
            )
        )
        self.theta_b_hat, self.theta_c_b_hat = theta_b, theta_c_b

        self.theta_vertices = np.concatenate(
            (
                np.ones((self.theta_vertices.shape[0], 1, self.theta_vertices.shape[2])),
                self.theta_vertices,
            ),
            axis=1,
        ).transpose(1, 0, 2).reshape(-1, self.theta_vertices.shape[2])

        self.theta_c_vertices = np.concatenate(
            (
                np.ones((self.theta_c_vertices.shape[0], 1, self.theta_c_vertices.shape[2])),
                self.theta_c_vertices,
            ),
            axis=1,
        ).transpose(1, 0, 2).reshape(-1, self.theta_c_vertices.shape[2])

        if self.theta_vertices.size == 0:
            self.theta_vertices = np.ones((1, self.sym.th_vertices_N.shape[1]))

        if self.theta_c_vertices.size == 0:
            self.theta_c_vertices = np.ones((1, self.sym.th_c_vertices_N.shape[1]))


        if self.th_hat is None:
            self.th_hat = np.zeros((self.q + 1, 1))
            self.th_hat[0] = 1
            theta_set = Polytope(A=self.theta.A, b=theta_b) if theta_b.size else Polytope()
            th0 = initial_point_estimate(theta_set, self.q, self.options.th_hat_0, "th_hat_0")
            self.th_hat[1:] = th0[:, np.newaxis] if th0.size else np.empty((0, 1))

        if self.th_c_hat is None:
            self.th_c_hat = np.zeros((self.q_c + 1, 1))
            self.th_c_hat[0] = 1
            theta_c_set = Polytope(A=self.theta_c.A, b=theta_c_b) if theta_c_b.size else Polytope()
            th_c0 = initial_point_estimate(theta_c_set, self.q_c, self.options.th_c_hat_0, "th_c_hat_0")
            self.th_c_hat[1:] = th_c0[:, np.newaxis] if th_c0.size else np.empty((0, 1))

        self.th_hat[1:], self.th_c_hat[1:] = self.filter.update(
            theta_b,
            theta_c_b,
            self.z_prev[:, np.newaxis],
            x0.toarray(),
            y0.toarray())
            
        # print(self.th_hat)

        if r is None:
            r = np.zeros(self.sym.r.shape)
        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))
        if self.first_time:
            self.sol = self.qpsol(
                p=ca.vertcat(x0,
                            r,
                            self.th_hat,
                            self.th_c_hat,
                            self.theta_vertices.reshape(-1,1),
                            self.theta_c_vertices.reshape(-1,1)),
                lbg=new_lbg,
                ubg=new_ubg,
            )
            self.first_time = False
        else:
            x_warm, _ = self._warm_start()
            self.sol = self.qpsol(
                p=ca.vertcat(x0,
                            r,
                            self.th_hat,
                            self.th_c_hat,
                            self.theta_vertices.reshape(-1,1),
                            self.theta_c_vertices.reshape(-1,1)),
                lbg=new_lbg,
                ubg=new_ubg,
                x0=x_warm
        )

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n : (self.N + 1) * self.n + self.m
        ]

        self.z_prev = np.block([[x0.toarray()], [self.u_star.toarray()]])

        if self.track:
            xa_idx = (self.N + 1) * self.n + self.N * self.m
            self.xa_star = (
                self.sol["x"][xa_idx: xa_idx + self.n * self.sym.xa.shape[1]]
                .toarray()
                .reshape((self.n, -1), order="F")
            )
        else:
            self.xa_star = None

        return self.sol['x']