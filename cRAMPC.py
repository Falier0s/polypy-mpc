"""A Robust MPC using CasADI."""

import numpy as np
import casadi as ca
import cvxpy as cp

from pycvxset import Polytope

from cRMPC import CRMPC

from adaptive_tool import Filter, SetUpdater


class CRAMPC(CRMPC):
    """
    A Robust Adaptive MPC using CasADI.

    This class implements a Robust Adaptive MPC problem using CasADI for optimization. It inherits from the CRMPC class and adds functionality for parameter estimation and constraint tightening to handle uncertainties in the system.
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
        system : cMPC.System
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
        The Robust Adaptive MPC Constructor adopts the superClass constructor from CRMPC and then initializes the variables for the parameter estimation and the constraint tightening.
        """
        super().__init__(system, Q, R, N, options)

        length = N if self.options.lpv_flag else 1

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
        
        self.z_prev = None
        self.th_hat, self.th_c_hat = None, None
        
        ab_ = np.concatenate(
            (self.sys.A, self.sys.B),
            axis=1).transpose(2, 0, 1)
        
        c= self.sys.C.transpose(2, 0, 1)

        self.filter = Filter(ab_, c, self.theta, self.theta_c, self.options.par_filter)

        self.param_set_learn = SetUpdater(
            ab_, c, self.theta, self.theta_c, self.W, self.E, length
        )

    def solve(self, x0, y0, r=None):

        if not self.z_prev:
            self.z_prev = np.block([x0, np.zeros(self.m)])

        theta_b, theta_c_b, th_vertices_N, th_c_vertices_N = (
            self.param_set_learn.update(
                self.theta, self.theta_c, self.z_prev, x0, y0, self.max_delta_th
            )
        )

        th_vertices_N = np.concatenate(
            (
                th_vertices_N,
                np.ones((th_vertices_N.shape[0], 1, th_vertices_N.shape[1])),
            ),
            axis=1,
        ).transpose(1, 0, 2).reshape(-1, self.N)

        th_c_vertices_N = np.concatenate(
            (
                th_c_vertices_N,
                np.ones((th_c_vertices_N.shape[0], 1, th_c_vertices_N.shape[1])),
            ),
            axis=1,
        ).transpose(1, 0, 2).reshape(-1, self.N)

        if not self.th_hat:
            self.th_hat = np.zeros((self.q + 1, 1))
            self.th_hat[0] = 1
            self.th_hat[1:] = Polytope(A=self.theta.A, b = theta_b).chebyshev_centering()

        if not self.th_c_hat:
            self.th_c_hat = np.zeros((self.q_c + 1, 1))
            self.th_c_hat[0] = 1
            self.th_c_hat[1:] = Polytope(A=self.theta_c.A, b = theta_c_b).chebyshev_centering()

        self.filter.update(theta_b, theta_b_c, self.z_prev, y0, self.th_hat, self.th_c_hat)

        if r is None:
            r = np.zeros(self.sym.r.shape)
        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))

        self.sol = self.qpsol(
            p=ca.vertcat(x0, r, th_vertices_N, th_c_vertices_N),
            lbg=new_lbg,
            ubg=new_ubg,
            x0=self._warm_start()
        )

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n : (self.N + 1) * self.n + self.m
        ]

        self.z_prev = np.block([[x0], [self.u_star]])

        return self.sol['x']