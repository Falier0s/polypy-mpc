"""A Robust MPC using CasADI."""

import numpy as np
import casadi as ca
import cvxpy as cp

from pycvxset import Polytope

from cMPC import CRMPC
from pagemtimes import pagemtimes


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

        # TODO - check if option filter exists and initialize as default LMS filter
        # TODO - initialize the estimate variables for th_hat, th_c_hat and the z{k-1}


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

    def _filter(self,x0,y0):
        """
        Estimate the parameters theta and theta_c for prediction model adaptation

        Parameters
        ----------

        x0 : np.ndarray
            The current state of the system.
        y0 : np.ndarray
            The current output of the system.
        self.options['filter'] : dict value
            The filter type to be used for parameter estimation. It can be 'LMS' for Least Mean Squares filter or 'RLS' for Recursive Least Squares filter.

        Return
        ------
        theta_hat : np.ndarray
            The estimated parameters for the system matrices A and B.
        theta_c_hat : np.ndarray
            The estimated parameters for the output matrix C.
        """
        
        pass
    
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