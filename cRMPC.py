"""A Robust MPC using CasADI."""

import numpy as np
import casadi as ca
import cvxpy as cp

from pycvxset import Polytope

from cMPC import CMPC
from pagemtimes import pagemtimes


class CRMPC(CMPC):
    """A Robust MPC using CasADI."""

    def __init__(self, system, Q, R, N, options):
        """Initialize the CRMPC controller."""
        super().__init__(system, Q, R, N, options)

        c_mask = np.any(system["C"] != 0, axis=[0, 1]).squeeze()
        ab_mask = np.any(
            [
                np.any(system["A"] != 0, axis=[0, 1]),
                np.any(system["B"] != 0, axis=[0, 1]),
            ],
            axis=1,
        ).squeeze()

        self.sys.A = self.sys.A[:, :, ab_mask]
        self.sys.B = self.sys.B[:, :, ab_mask]
        self.sys.C = self.sys.C[:, :, c_mask]

        self.q = np.sum(ab_mask) - 1
        self.q_c = np.sum(c_mask) - 1

        if options.get("W") is not None and isinstance(options.get("W"), Polytope):
            self.W = options["W"]
        else:
            self.W = Polytope(
                A=np.vstack((np.eye(self.n), -np.eye(self.n))),
                b=np.concatenate((-np.ones(self.n) * 0.1, np.ones(self.n) * 0.1)),
            )
        self.c = ca.MX.sym("c", self.m, self.N)

    def initialize_uncertainties(self, theta=None, theta_c=None):

        if theta is not None:
            self.theta = theta
            self.theta_vertices, self.veticies_number = self.set_param_set(theta)
        else:
            self.theta = Polytope(V=np.eye(self.q))
            if self.q > 0:
                self.theta_vertices, self.veticies_number = self.set_param_set(
                    self.theta
                )
            else:
                self.theta_vertices, self.veticies_number = 1, 1
        self.th = ca.MX.sym("th", self.q, 1)
        self.Av = pagemtimes(
            self.sys.A.transpose(0, 2, 1), self.theta_vertices.T
        ).transpose(0, 2, 1)
        self.Bv = pagemtimes(
            self.sys.B.transpose(0, 2, 1), self.theta_vertices.T
        ).transpose(0, 2, 1)
        self.add_hard_constraints()
        if theta_c.any():
            self.theta_c = theta_c
            self.theta_c_vertices, self.c_veticies_number = self.set_param_set(theta_c)
        else:
            self.theta_c = Polytope(V=np.eye(self.q_c))
            if self.q_c > 0:
                self.theta_c_vertices, self.c_veticies_number = self.set_param_set(
                    self.theta_c
                )
            else:
                self.theta_c_vertices, self.c_veticies_number = 1, 1
        self.th_c = ca.MX.sym("th_c", self.q_c, 1)
        self.Cv = self.param_eval(self.sys.C, self.theta_c_vertices)

    def initiatialize(self, mode="None"):
        if not self.K.any():
            self._stab_gain(self.sys.A, self.sys.B, self.W.V, mode)
        self.Ak = self.sys.A + pagemtimes(self.sys.B, self.K[:, :, np.newaxis])
        self.Ak_vertices = pagemtimes(
            self.Ak.transpose(0, 2, 1), self.theta_vertices.squeeze().T
        ).transpose(0, 2, 1)
        x0_poly = Polytope(
            A=self.polys.z.A @ np.hstack((np.eye(self.n), self.K)), b=self.polys.z.b
        )
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
        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))
        self.sol = self.qpsol(p=ca.vertcat(x0, r), lbg=new_lbg, ubg=new_ubg)

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n : (self.N + 1) * self.n + self.m
        ]

    def _system_build(self):
        Ak = self.param_evel(self.Ak, self.theta_vertices.T)
        A = self.param_evel(self.sys.A, self.theta_vertices.T)
        B = self.param_evel(self.sys.B, self.theta_vertices.T)

        H = pagemtimes(
            self.Hbar.transpose(0, 2, 1), self.theta_vertices.squeeze().T
        ).transpose(0, 2, 1)
        HC = pagemtimes(
            self.HCbar.transpose(0, 2, 1), self.theta_c_vertices.squeeze().T
        ).transpose(0, 2, 1)
        Gb = np.vstack((self.G, np.zeros((self.Fc, 1)), self.G.shape[1]))

        self.constraints = self.V.A @ self.sym.x[0] <= self.alpha[0]
        for k in range(self.N):
            self.constraints = ca.vertcat(
                self.constraints,
                self.sym.x[k+1] == Ak @ (self.sym.x[k]-self.sym.xa[k]) + B @ self.c[k] + self.sys.A @ self.sym.xa[k] + B @ self.sym.ua[k],
                self.sym.u[k] == self.K @ (self.sym.x[k]-self.sym.xa[k]) + self.c[k] + self.sym.ua[k]
            )
            for j in range(self.veticies_number):
                self.constraints = ca.vertcat(
                    self.constraints,
                    H[:, :, j] @ self.alpha[k] + self.V.A @ (np.eye(self.n)-self.Ak_vertices[:, :, j]) @ self.sym.xa[k] + self.V.A @ (self.sys.A - self.Ak_vertices[:, :, j]) @ self.sym.x[k] + self.V.A @ self.Bv[:,:,j]@self.c[k] + self.w_bar <= self.alpha[k+1]
                )
            for j in range(self.c_veticies_number):
                self.constraints = ca.vertcat(
                    self.constraints,
                    HC[:, :, j] @ self.alpha[k] - Gb @ self.sym.xa + Gb @ self.c[k] + Gb @ self.sym.ua <= np.ones((np.size(Gb, 0), 1))
                )
        if self.track:
            for j in range(self.veticies_number):
                self.constraints = ca.vertcat(
                    self.constraints,
                    H[:, :, j] @ self.alpha[self.N] + self.V.A @ (np.eye(self.n)-self.Ak_vertices[:, :, j]) @ self.sym.xa[self.N] + self.w_bar <= self.alpha[self.N]
                )
            for j in range(self.c_veticies_number):
                self.constraints = ca.vertcat(
                    self.constraints,
                    HC[:, :, j] @ self.alpha[self.N] - Gb @ self.K @ self.sym.xa + Gb @ self.sym.ua <= np.ones((np.size(Gb; 0), 1)),
                )
            self.constraints = ca.vertcat(
                self.constraints,
                self.Zs.A @ ca.horzcat(self.sym.xa[k], self.sym.ua[k]) <= self.Zs.b,
                ca.vertcat(A - np.eye(self.n), B) @ ca.horzcat(self.sym.xa[k], self.sym.ua[k]) == np.zeros((self.n, 1))
                )

    def _set_controller(self, options=None):
        if options is None:
            options = {}

        decision_vars = ca.vertcat(
            self.sym.x.reshape((-1, 1)), self.sym.u.reshape((-1, 1)),
            self.theta.reshape((-1, 1)), self.theta_c.reshape((-1, 1))
        )

        if self.track:
            decision_vars = ca.vertcat(decision_vars, ca.reshape(self.nu, -1, 1))
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
            "p": ca.vertcat(self.sym.x_init, ca.reshape(self.sym.r, (- 1, 1))),
        } # TODO - check if need to have more params
        self.qpsol = ca.qpsol("qpsol", self.options.solver, qp, options)

    def _tight_constraints(self):
        Fb = np.concatenate((1, np.repeat(self.F, (1, 1, self.vC)), self.F@self.Cv))
        Gb = np.vstack((self.G, np.zeros((np.size(self.Fc, 0)), np.size(self.G, 1))))
        DF = Fb[:,:,0] + Gb @ self.K
        if self.q_c > 0:
            DF = np.vstack((DF, Fb[:, :, 2: -1]-Fb[:, :, 1]))

        Aineq = []
        bineq = np.zeros((self.na*self.vC, 1))
        for k in range(self.vC):
            Aineq = np.vstack((
                Aineq, -np.kron(np.eye(self.na), self.theta_c_vertices[k, :])
            ))
        x_cp = cp.Variable((Aineq.shape[0], 1))
        constraints = [Aineq @ x_cp << bineq]
        Aeq = np.kron(self.V.A.T, np.eye(self.q_c+1))
        beq = np.reshape(DF.transpose(2, 1, 0), ((self.q_c+1)*self.n, 1, np.size(DF, 0)))
        constraints += [Aeq @ x_cp == beq]
        options = None # TODO - check mskoptimset equivalence and function
        for i in range(np.size(Fb, 0)):
            f = np.zeros((1, 1, self.vC))
            for j in range(self.vC):
                cost = cp.kron(np.ones((1, self.na)), self.theta_c_vertices[j, :])
                lmip = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
                lmip.solve(solver=cp.CLARABEL, options=options)
                # TODO - check with Fabio

    def tube_inclusion(self):
        # With the following we want to derive the code for 
        # S[i] = {x| V*x[i] <= alpha[i]} and S[i+1] = {x| V*x[i+1] <= alpha[i+1]}
        # (Multiplicative uncertainty) [4] [5]
        # or S[i] = {x| V*x[i] <= alpha[i]} and S[i+1](theta) = {x| V*x[i+1](theta) <= alpha[i+1]}
        # (Parametric uncertainty) [3] [6]
        # such that S[i] \subseteq S[i+1] for all i=1,...,N-1

        # Simple Multiplicative uncertainty has been parametrized setting parameter
        # as one in each vertex of the set i.e. Th.V=eye(q)
        # in this way it can be manages as a parametric uncertainty
        options = None  # TODO - check mskoptimset equivalence and function
        for i in range(self.na):
            # TODO - check with Fabio
            pass

        Aineq = []
        bineq = np.zeros((self.na*self.v, 1))
        for k in range(self.v):
            Aineq = np.vstack((
                Aineq, -np.kron(np.eye(self.na), self.theta_vertices[k, :])
            ))
        Aeq = np.kron(self.V.A.T, np.eye(self.q+1))
        beq = np.reshape(self.Ak_vertices.transpose(2, 1, 0), ((self.q+1)*self.n, 1, np.size(self.Ak_vertices, 0)))
        x_cp = cp.Variable((Aeq.shape[1], 1))
        constraints = [Aineq @ x_cp << bineq, Aeq @ x_cp == beq]
        for i in range(self.na):
            # TODO - check with Fabio
            pass

    def param_evel(self, a, th):
        if isinstance(th, float) or isinstance(th, int):
            H = pagemtimes(th, a.transpose(2, 0, 1)).transpose(1, 2, 0)
        elif isinstance(th, ca.MX):
            H = 0
            for i in range(th.shape[0]):
                H += th[i] * a[:, :, i]
        return H

    def set_param_set(self, theta):
        theta_vertices = np.vstack((np.ones(np.size(theta.V, 0), 1), theta.V))
        vertices_number = np.size(theta_vertices, 0)
        return theta_vertices, vertices_number
