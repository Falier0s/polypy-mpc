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

        self.theta = None
        self.theta_vertices = None
        self.vertices_number = None
        self.th = None
        self.Av = None
        self.Bv = None
        self.theta_c = None
        self.theta_c_vertices = None
        self.c_vertices_number = None
        self.th_c = None
        self.Hbar = None
        self.HCbar = None
        self.w_bar = None
        # TODO - Recheck this part, I think my mask is broken
        c_mask = np.any(system["C"] != 0, axis=(0, 1)).squeeze()
        ab_mask = np.any(
            [[
                np.any(system["A"] != 0, axis=(0, 1)),
                np.any(system["B"] != 0, axis=(0, 1)),
            ]],
            axis=1,
        ).squeeze()

        self.sys.A = self.sys.A[:, :, ab_mask]
        self.sys.B = self.sys.B[:, :, ab_mask]
        self.sys.C = self.sys.C[:, :, c_mask]

        self.q = np.sum(ab_mask) - 1
        self.q_c = np.sum(c_mask) - 1

        self.c = ca.MX.sym("c", self.m, self.N)

    def initialize_uncertainties(self, theta=None, theta_c=None):

        if theta is not None:
            self.theta = theta
            self.theta_vertices, self.vertices_number = self.set_param_set(theta)
        else:
            self.theta = Polytope(V=np.eye(self.q))
            if self.q > 0:
                self.theta_vertices, self.vertices_number = self.set_param_set(
                    self.theta
                )
            else:
                self.theta_vertices, self.vertices_number = np.array([[1]]), 1
                raise Warning("No uncertainty in A and B matrices, consider using cMPC instead of CRMPC")
        self.th = ca.MX.sym("th", self.q+1, 1)
        self.Av = np.einsum('ikj,lj->ikl', self.sys.A, self.theta_vertices)
        # pagemtimes(
        #     self.sys.A.transpose(0, 2, 1), self.theta_vertices.T
        # ).transpose(0, 2, 1)
        self.Bv = np.einsum('ikj,lj->ikl', self.sys.B, self.theta_vertices)
        self.add_hard_constraints()
        if theta_c is not None:
            self.theta_c = theta_c
            self.theta_c_vertices, self.c_vertices_number = self.set_param_set(theta_c)
        else:
            self.theta_c = Polytope(V=np.eye(self.q_c))
            if self.q_c > 0:
                self.theta_c_vertices, self.c_vertices_number = self.set_param_set(
                    self.theta_c
                )
            else:
                self.theta_c_vertices, self.c_vertices_number = np.array([[1]]), 1
        self.th_c = ca.MX.sym("th_c", self.q_c+1, 1)
        self.Cv = self.param_eval(self.sys.C, self.theta_c_vertices)

    def initialize(self, mode="None", constraints=None):
        # Build hard constraints (user-defined)
        # self.add_hard_constraints()
        if constraints is not None:
            self.add_hard_constraints(constraints)

        if not self.K.any():
            self._stab_gain(self.sys.A, self.sys.B, self.W.V, mode)
        self.Ak = self.sys.A + np.einsum('ijk,lm->imk',self.sys.B, self.K) # TODO - check if correct
        self.Ak_vertices = np.einsum('ikj,lj->ikl', self.Ak, self.theta_vertices)
        x0_poly = Polytope(
            A=self.polys.z.A @ np.vstack((np.eye(self.n), self.K)), b=self.polys.z.b
        )
        self._lam_contract_set(self.Ak_vertices, x0_poly, self.lam)
        # from matplotlib import pyplot as plt
        # ax,_,_= self.poly_x_aug.plot()
        # plt.show()

        self.V = self.poly_x_aug
        self.na = self.V.A.shape[0]
        self.Hbar = np.zeros((self.na, self.q+1, self.na))
        self.HCbar = np.zeros((self.q_c+1, self.na, self.f_const.shape[0])) 
        self.w_bar = np.zeros((self.na, 1))
        self.alpha = ca.MX.sym("alpha", self.na, self.N + 1)
        self.tube_inclusion()
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
        self.sol = self.qpsol(p=ca.vertcat(x0, r, np.vstack((1,np.zeros((self.q,1)))), np.vstack((1,np.zeros((self.q_c,1))))), lbg=new_lbg, ubg=new_ubg)

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n : (self.N + 1) * self.n + self.m
        ]

    def _system_build(self):
        """


        """
        Ak = self.param_eval(self.Ak, self.th.T)
        A = self.param_eval(self.sys.A, self.th.T)
        B = self.param_eval(self.sys.B, self.th.T)

        H = np.einsum('ijk,lk->ijl', self.Hbar, self.theta_vertices)

        HC = np.einsum('ijk,lk->ijl', self.HCbar, self.theta_c_vertices)

        Gb = np.vstack((self.g_const, np.zeros((np.size(self.fc_const, 0),
                                                np.size(self.g_const, 1)))))


        sym_A = ca.MX.sym("A", self.n, self.n)
        sym_B = ca.MX.sym("B", self.n, self.m)


        step = ca.Function(
            "step",
            [self.sym.get_x(), self.sym.get_u(), sym_A, sym_B],
            [
                sym_A @ self.sym.get_x() + sym_B @ self.sym.get_u() 
            ],
        )

        linear_k = ca.Function(
            "linearK", [self.sym.get_x()], [self.K @ self.sym.get_x()]
        )

        reacheability = ca.Function(
            "reachibility", [self.sym.get_x(), self.sym.get_u()],
            [self.polys.zs.A @ ca.vertcat(self.sym.get_x(), self.sym.get_u()) - self.polys.zs.b]
        )
        artificial_idx = (slice(self.N), slice(1,self.N+1)) if (self.track and self.sym.r.shape[1] > 1) else (-1, -1)

            
        initial = self.sym.x[:, 0] - self.sym.x_init  # Initial condition
        
        
        initialTube = self.V.A @ self.sym.x[:, 0] - self.alpha[:, 0]

        artifEquil =  step(self.sym.xa[:, -1], self.sym.ua[:, -1], A-np.eye(self.n), B)
   
        dynamics = self.sym.x[:, 1:] - (step.map(self.N)(self.sym.x[:, : self.N] - self.sym.xa[:, artificial_idx[0]], self.c, Ak, B) + step.map(self.sym.ua.shape[1])(self.sym.xa[:, artificial_idx[0]], self.sym.ua, A, B))

        input_policy = self.sym.u - linear_k.map(self.N)(self.sym.x[:, : self.N] - self.sym.xa[:, artificial_idx[0]]) - (self.c + self.sym.ua)

        # tubEvolve = [ca.reshape(
        #     H[:, :, v] @ self.alpha[:, : self.N] + self.V.A @ (
        #         self.sym.xa[:, artificial_idx[1]] - self.Ak_vertices[:, :, v] @ self.sym.xa[:, artificial_idx[0]]
        #         + self.Bv[:, :, v] @ self.c) + self.w_bar - self.alpha[:, 1 : ], (-1,1)
        #     ) for v in range(self.vertices_number)]
        
        tightConstraint = [ca.reshape(
            HC[:, :, v] @ self.alpha[:, : self.N] + Gb @ (
                linear_k.map(self.sym.ua.shape[1])(self.sym.xa[:, artificial_idx[0]]) + self.c + self.sym.ua[:, artificial_idx[0]]
                ) - np.ones((Gb.shape[0],1)), (-1,1)
            ) for v in range(self.c_vertices_number)]
        
        terminalTube = [ca.reshape(
            H[:, :, v] @ self.alpha[:, -1] + self.V.A @ (np.eye(self.n) - self.Ak_vertices[:, :, v]) @ self.sym.xa[:, -1] + self.w_bar - self.alpha[:, -1], (-1,1)
        ) for v in range(self.vertices_number)]
        
        terminalConstraint = [ca.reshape(
            HC[:, :, v] @ self.alpha[:, -1] + Gb @ (
                linear_k(self.sym.xa[:, -1]) + self.sym.ua[:, -1]
                ) - np.ones((Gb.shape[0],1)), (-1,1)
            ) for v in range(self.c_vertices_number)]
        
        if self.sym.r.shape[1] > 1:
            tubEvolve = [ca.reshape(
                H[:, :, v] @ self.alpha[:, : self.N] + self.V.A @ (
                self.sym.xa[:, artificial_idx[1]] - self.Ak_vertices[:, :, v] @ self.sym.xa[:, artificial_idx[0]]
                + self.Bv[:, :, v] @ self.c) + self.w_bar - self.alpha[:, 1 : ], (-1,1)
            ) for v in range(self.vertices_number)]

            artificialTrajectory = step.map(self.N)(self.sym.xa[:, : self.N], self.sym.ua,A,B)
            self.g.append(artificialTrajectory)
            self.lbg.append([0.0] * (self.n*self.N))
            self.ubg.append([0.0] * (self.n*self.N))

        else:
            tubEvolve = [ca.reshape(
                H[:, :, v] @ self.alpha[:, : self.N] + self.V.A @ (
                (np.eye(self.n) - self.Ak_vertices[:, :, v]) @ self.sym.xa[:, artificial_idx[0]]
                + self.Bv[:, :, v] @ self.c) + self.w_bar - self.alpha[:, 1 : ], (-1,1)
            ) for v in range(self.vertices_number)]

        self.g.append(
            ca.reshape(reacheability.map(self.sym.ua.shape[1])(
                self.sym.xa[:, artificial_idx[0]],self.sym.ua),(-1,1))
        )

        self.lbg.append([-ca.inf] * (self.polys.zs.A.shape[0]*self.sym.ua.shape[1]))
        self.ubg.append([0.0] * (self.polys.zs.A.shape[0]*self.sym.ua.shape[1]))        

        self.g.append(initial)
        self.lbg.append([0.0] * (self.n))
        self.ubg.append([0.0] * (self.n))

        self.g.append(artifEquil)
        self.lbg.append([0.0] * (self.n))
        self.ubg.append([0.0] * (self.n))

        self.g.append(ca.reshape(dynamics, (-1, 1)))
        self.lbg.append([0.0] * (self.n * self.N))
        self.ubg.append([0.0] * (self.n * self.N))

        self.g.append(ca.reshape(input_policy, (-1, 1)))
        self.lbg.append([0.0] * (self.m * self.N))
        self.ubg.append([0.0] * (self.m * self.N))

        self.g.append(initialTube)
        self.lbg.append([-ca.inf] * (len(self.V.b)))
        self.ubg.append([0.0] * (len(self.V.b)))

        for tubeVertex in tubEvolve:
            self.g.append(tubeVertex)
            self.lbg.append([-ca.inf] * (self.na * self.N))
            self.ubg.append([0.0] * (self.na * self.N))

        if self.track:
            for tubeVertex in terminalTube:
                self.g.append(tubeVertex)
                self.lbg.append([-ca.inf] * (self.na))
                self.ubg.append([0.0] * (self.na))

        for constraintVertex in tightConstraint:
            self.g.append(constraintVertex)
            self.lbg.append([-ca.inf] * (Gb.shape[0] * self.N))
            self.ubg.append([0.0] * (Gb.shape[0] * self.N))

        for constraintVertex in terminalConstraint:
            self.g.append(constraintVertex)
            self.lbg.append([-ca.inf] * (Gb.shape[0]))
            self.ubg.append([0.0] * (Gb.shape[0]))
        


    def _set_controller(self, options=None):
        if options is None:
            options = {}

        decision_vars = ca.vertcat(
            self.sym.x.reshape((-1, 1)), self.sym.u.reshape((-1, 1)),
            self.alpha.reshape((-1, 1)), self.c.reshape((-1, 1))
        )

        if self.track:
            decision_vars = ca.vertcat(
                decision_vars,
                ca.reshape(self.sym.xa, -1, 1),
                ca.reshape(self.sym.ua, -1, 1),
            )

        # if self.track:
        #     decision_vars = ca.vertcat(decision_vars, ca.reshape(self.nu, -1, 1))
        new_g = ca.MX()
        for val in self.g:
            print(val)
            if isinstance(val, list):
                for vval in val:
                    print(vval)
                    new_g = ca.vertcat(new_g, vval)
            else:
                new_g = ca.vertcat(new_g, val.reshape((-1, 1)))
        self.g = new_g
        qp = {
            "x": decision_vars,
            "f": self.cost_fun,
            "g": self.g,
            "p": ca.vertcat(self.sym.x_init, ca.reshape(self.sym.r, (- 1, 1)),
                            self.th, self.th_c),
        }  # TODO - check if need to have more params
        self.qpsol = ca.qpsol("qpsol", self.options.solver, qp, options)

    def _tight_constraints(self):
        """
        Compute constraint tightening for the robust MPC problem. 
        
        The tightening is computed using the vertices of the uncertain parameters in the system matrices and in the output matrix C.
        
        vC is the number of vertices of the set of the uncertain parameters in the output matrix C, i.e. theta_c
        """
        # TODO - rework this function to use hard_constraint function.

        # Fc = self.polys.zc.A @ np.reshape(np.einsum('ijk,k...->ij...',self.sys.C,self.theta_c_vertices),
        #                                   (self.c_vertices_number*self.p,self.n))

        # If self.Cv is not the correct shape, check param_eval, with the exemple it's a (2, 3, 4) and Fc a (1, 2, 4)
        # Problem in the repeat can't work (8) with (3)
        Fb = np.vstack((np.repeat(self.f_const[:,:,np.newaxis], 
                                          (self.c_vertices_number), axis=2),
                                          np.einsum('ij,jkl->ikl', self.fc_const,self.Cv)))
        Gb = np.vstack((self.g_const, np.zeros((np.size(self.fc_const, 0),
                                                np.size(self.g_const, 1)))))
        DF = Fb[:, :, 0] + Gb @ self.K
        if self.q_c > 0:
            DF = np.vstack((DF, Fb[:, :, 2: -1]-Fb[:, :, 1]))

        Aineq = np.empty((0 , self.na))
        bineq = np.zeros((self.na*self.c_vertices_number, 1))
        for k in range(self.c_vertices_number):
            Aineq = np.vstack((
                Aineq, -np.kron(np.eye(self.na), self.theta_c_vertices[k, :])
            ))
        x_cp = cp.Variable((Aineq.shape[0], 1))
        Aeq = np.kron(self.V.A.T, np.eye(self.q_c+1))
        beq = np.reshape(DF[:, :, np.newaxis].transpose(2, 1, 0), ((self.q_c+1)*self.n, 1, np.size(DF, 0)))
        # TODO move constraint in the loop evaluating constraint for each beq[:, :, i] element
        options = None  # TODO - check mskoptimset equivalence and function
        for i in range(np.size(Fb, 0)):
            constraints = [Aineq @ x_cp <= bineq]
            constraints += [Aeq @ x_cp == beq[:, :, i]]
            f = np.zeros(self.c_vertices_number)
            for j in range(self.c_vertices_number):
                cost = np.kron(np.ones((1, self.na)), self.theta_c_vertices[j, :])
                lp = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
                lp.solve()
                f[j] = lp.value
                hbar = x_cp.value
                if f[j] >= np.max(f):
                    self.HCbar[:, :, i] = np.reshape(hbar, (self.q_c+1, self.na))
        self.HCbar = np.transpose(self.HCbar, (2, 1, 0))

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

        w_cp = cp.Variable((self.n,1))
        w_constraints = [
            self.W.A @ w_cp <= self.W.b
        ]

        for i in range(self.na):
            w_lp = cp.Problem(cp.Minimize(-self.V.A[i,:] @ w_cp), w_constraints)   
            w_lp.solve()
            self.w_bar[i, :] = w_lp.value
           

        for i in range(self.na):
            f = np.zeros(self.vertices_number)
            for j in range(self.vertices_number):
                Aineq = -np.kron(np.eye(self.na), self.theta_vertices[j, :])
                bineq = np.zeros((self.na, 1))

                Aeq = np.kron(self.V.A.T, np.eye(self.q+1))
                beq = np.reshape(np.einsum('ij,ljk->ijk', self.V.A, self.Ak).transpose(2, 1, 0), ((self.q+1)*self.n,1, self.na))

                x_cp = cp.Variable((Aeq.shape[1], 1))

                constraints = [Aineq @ x_cp <= bineq, Aeq @ x_cp == beq[:, :, i]]
                cost = np.kron(np.ones((self.na, 1)), self.theta_vertices[j, :, np.newaxis]).T
                lp = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
                lp.solve()
                f[j] = lp.value
                hbar = x_cp.value
                if f[j] >= np.max(f):
                    self.Hbar[:, :, i] = np.reshape(hbar, (self.q+1, self.na)).T
        self.Hbar = np.transpose(self.Hbar, (2, 0, 1))

        # Aineq = np.empty((0 , self.na*(self.q+1)))
        # bineq = np.zeros((self.na*self.vertices_number, 1))
        # for k in range(self.vertices_number):
        #     Aineq = np.vstack((
        #         Aineq, -np.kron(np.eye(self.na), self.theta_vertices[k, :])
        #     ))
        # Aeq = np.kron(self.V.A.T, np.eye(self.q+1))
        # beq = np.reshape(np.einsum('ij,ljk->ijk', self.V.A, self.Ak).transpose(2, 1, 0), ((self.q+1)*self.n, 1, self.na))
        # x_cp = cp.Variable((Aeq.shape[1], 1))
        # for i in range(self.na):
        #     f = np.zeros((1, 1, self.vertices_number))
        #     constraints = [Aineq @ x_cp <= bineq, Aeq @ x_cp == beq]
        #     for j in range(self.vertices_number):
        #         cost = np.reshape(np.kron(np.ones((1, self.na)), self.theta_vertices[j, :,np.newaxis]).T, (1,-1))
        #         lp = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
        #         lp.solve(verbose=True)  # TODO - add solver flag
        #         hbar = x_cp.value
        #         f[j] = lp.value
        #         if f[j].any() > np.max(f):
        #             self.Hbar[:, :, i] = np.reshape(hbar, (self.q+1, self.na))
        #     # TODO - check with Fabio

    def param_eval(self, a, th):
        H = None
        if isinstance(th, float) or isinstance(th, int) or isinstance(th, np.ndarray):
            # H = pagemtimes(th, a.transpose(2, 0, 1)).transpose(1, 2, 0)
            if isinstance(th, float) or isinstance(th, int):
                H = np.einsum('ikj...,->ikj...', a, th)
            else:
                H = np.einsum('ijk,...k->ij...', a, th)
        elif isinstance(th, ca.MX):
            H = 0
            for i in range(th.shape[0]):
                H += th[i] * a[:, :, i]
        return H

    def set_param_set(self, theta):
        theta_vertices = np.hstack((np.ones((np.size(theta.V, 0), 1)), theta.V))
        vertices_number = np.size(theta_vertices, 0)
        return theta_vertices, vertices_number
