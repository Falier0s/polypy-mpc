"""A Robust MPC using CasADI."""
import numpy as np
import casadi as ca

from pycvxset import Polytope

from cMPC import CMPC
from pagemtimes import pagemtimes


class CRMPC(CMPC):
    """A Robust MPC using CasADI."""

    def __init__(self, system, Q, R, N, options):
        """Initialize the CRMPC controller."""
        super().__init__(system, Q, R, N, options)

        c_mask = np.any(system["C"] != 0, axis=[0, 1]).squeeze()
        ab_mask = np.any([np.any(system["A"] != 0, axis=[0, 1]),
                          np.any(system["B"] != 0, axis=[0, 1])], axis=1).squeeze()

        self.sys.A = self.sys.A[:, :, ab_mask]
        self.sys.B = self.sys.B[:, :, ab_mask]
        self.sys.C = self.sys.C[:, :, c_mask]

        self.q = np.sum(ab_mask) - 1
        self.q_c = np.sum(c_mask) - 1

        if options.get("W") is not None and isinstance(options.get("W"), Polytope):
            self.W = options["W"]
        else:
            self.W = Polytope(A=np.vstack((np.eye(self.n), -np.eye(self.n))),
                              b=np.concatenate((-np.ones(self.n) * 0.1, np.ones(self.n)*0.1)))
        self.c = ca.MX.sym("c", self.m, self.N)

    def initialize_uncertainties(self, theta=None, theta_c=None):
        
        if theta is not None:
            self.theta = theta
            self.theta_vertices, self.veticies_number = self.set_param_set(theta)
        else:
            self.theta = Polytope(V=np.eye(self.q))
            if self.q > 0:
                self.theta_vertices, self.veticies_number = self.set_param_set(self.theta)
            else:
                self.theta_vertices, self.veticies_number = 1, 1
        self.th = ca.MX.sym("th", self.q, 1)
        self.Av = pagemtimes(self.sys.A.transpose(0, 2, 1),
                             self.theta_vertices.T).transpose(0, 2, 1)
        self.Bv = pagemtimes(self.sys.B.transpose(0, 2, 1),
                             self.theta_vertices.T).transpose(0, 2, 1)
        self.add_hard_constraints()
        if theta_c.any():
            self.theta_c = theta_c
            self.theta_c_vertices, self.c_veticies_number = self.set_param_set(theta_c)
        else:
            self.theta_c = Polytope(V=np.eye(self.q_c))
            if self.q_c > 0:
                self.theta_c_vertices, self.c_veticies_number = self.set_param_set(self.theta_c)
            else:
                self.theta_c_vertices, self.c_veticies_number = 1, 1
        self.th_c = ca.MX.sym("th_c", self.q_c, 1)
        self.Cv = self.param_eval(self.sys.C, self.theta_c_vertices)

    def initiatialize(self, mode="None"):
        if not self.K.any():
            self._stab_gain(self.sys.A, self.sys.B, self.W.V, mode)
        self.Ak = self.sys.A + pagemtimes(self.sys.B, self.K[:, :, np.newaxis])
        self.Ak_vertices = pagemtimes(self.Ak.transpose(0, 2, 1),
                                      self.theta_vertices.T.squeeze()).transpose(0, 2, 1)
        x0_poly = Polytope(A=self.polys.z.A @ np.hstack((np.eye(self.n), self.K)), b=self.polys.z.b)
        self._lam_contract_set(self.Ak_vertices, x0_poly, self.lam)
        self.V = self.poly_x_aug
        self.na = self.V.A.shape[0]
        self.alpha = ca.MX.sym("alpha", self.na, self.N)
        self._tube_inclusion()
        self._tight_constraints()


        self._cost_fun_build()
        self._system_build()
        options = {}
        self._set_controller(options)

    def solve(self, x0, r=None):
        if r is None:
            r = np.zeros(self.sym.r.shape)
        
