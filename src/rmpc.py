"""A Robust MPC using CasADI."""

import numpy as np
import casadi as ca
import cvxpy as cp

from pycvxset import Polytope
from pycvxset.common import constants

from .mpc import MPC

from .tools.invariance_tools import InvariantSet, GainSynthesis



class RMPC(MPC):
    """A Robust MPC using CasADI."""

    def __init__(self, system, Q, R, N, options):
        """Initialize the RMPC controller."""
        super().__init__(system, Q, R, N, options)

        self.theta_vertices = None
        self.vertices_number = None
        self.th = None
        self.Av = None
        self.Bv = None
        self.theta_c_vertices = None
        self.c_vertices_number = None
        self.th_c = None
        self.Hbar = None
        self.HCbar = None
        self.w_bar = None

        c_mask = np.any(system["C"] != 0, axis=(0, 1)).squeeze()
        ab_mask = np.any(
            [
                [
                    np.any(system["A"] != 0, axis=(0, 1)),
                    np.any(system["B"] != 0, axis=(0, 1)),
                ]
            ],
            axis=1,
        ).squeeze()

        self.sys.A = self.sys.A[:, :, ab_mask]
        self.sys.B = self.sys.B[:, :, ab_mask]
        self.sys.C = self.sys.C[:, :, c_mask]

        self.q = np.sum(ab_mask) - 1
        self.q_c = np.sum(c_mask) - 1

        self.c = ca.MX.sym("c", self.m, self.N)
        try:
            self.theta = (
                self.options.theta if self.options.theta else Polytope(V=np.eye(self.q))
            )
        except:
            raise Warning("options.theta should be defined as a pycvxset.Polytope")

        self.theta_vertices, self.vertices_number = (
            self.set_param_set(self.theta) if self.q > 0 else (np.array([[1]]), 1)
        )

        try:
            self.theta_c = (
                self.options.theta_c
                if self.options.theta_c
                else Polytope(V=np.eye(self.q_c))
            )
        except:
            raise Warning("options.theta_c should be defined as a pycvxset.Polytope")

        self.theta_c_vertices, self.c_vertices_number = (
            self.set_param_set(self.theta_c) if self.q_c > 0 else (np.array([[1]]), 1)
        )

        self.sym.create_parameter_variables(
            self.q, self.q_c, self.vertices_number, self.c_vertices_number, 1
        )

        self.add_hard_constraints()

        if self.options.th_0 is not None:
            self.th = np.hstack(([1.0], self.options.th_0))
        elif self.q > 0:
            self.th = np.hstack(([1.0], self.theta.chebyshev_centering()[0]))
        else:
            self.th = np.array([1.0])  # no uncertain parameters: homogeneous coordinate only

        if self.options.th_c_0 is not None:
            self.th_c = np.hstack(([1.0], self.options.th_c_0))
        elif self.q_c > 0:
            self.th_c = np.hstack(([1.0], self.theta_c.chebyshev_centering()[0]))
        else:
            self.th_c = np.array([1.0])

    def _init_uncertainty_symbolic(self):
        """
        Initialize the symbolic variables for the uncertain parameters in the system matrices and in the output matrix C.

        Arguments:
        ----------
        length : int
            The length of the symbolic variables for the uncertain parameters evolving over steps.
            Default is 1 for the case of no parameters variation.
        """

        # Casadi Functions to Handle the matrix Evaluation at a given parameter

        """
        Evaluate self.A at theta.

        Arguments:
        ----------
        
        self.sym.th: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1), 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*n,1)

        """

        self.A_th_eval = ca.Function(
            "A_th_eval", [self.sym.th], [self.param_eval(self.sys.A, self.sym.th)]
        )

        """
        Evaluate self.B at theta.

        Arguments:
        ----------
        
        self.sym.th: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1), 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*m,1)

        """
        self.B_th_eval = ca.Function(
            "B_th_eval", [self.sym.th], [self.param_eval(self.sys.B, self.sym.th)]
        )

        """
        Evaluate self.C at theta.

        Arguments:
        ----------
        
        self.sym.th: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1), 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (p*n,1)

        """

        self.C_th_eval = ca.Function(
            "C_th_eval", [self.sym.th_c], [self.param_eval(self.sys.C, self.sym.th_c)]
        )

        """
        Evaluate self.Ak at theta.

        Arguments:
        ----------
        
        self.sym.th: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1), 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*n,1)

        """
        self.Ak_th_eval = ca.Function(
            "Ak_th_eval", [self.sym.th], [self.param_eval(self.Ak, self.sym.th)]
        )

        # Casadi Functions to Handle the matrix Evaluation at the vertices of the Parameters Set

        """
        Evaluate self.A at the vertices of Theta.

        Arguments:
        ----------
        
        self.sym.th_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1)*vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*n,vertices_number)

        """
        self.A_th_v_eval = ca.Function(
            "A_th_v_eval",
            [self.sym.th_vertices],
            [
                self.param_eval(
                    self.sys.A,
                    ca.reshape(
                        self.sym.th_vertices, self.vertices_number, self.q + 1
                    ).T,
                )
            ],
        )

        """
        Evaluate self.B at the vertices of Theta.

        Arguments:
        ----------
        
        self.sym.th_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1)*vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*m,vertices_number)

        """
        self.B_th_v_eval = ca.Function(
            "B_th_v_eval",
            [self.sym.th_vertices],
            [
                self.param_eval(
                    self.sys.B,
                    ca.reshape(
                        self.sym.th_vertices, self.vertices_number, self.q + 1
                    ).T,
                )
            ],
        )

        """
        Evaluate self.C at the vertices of ThetaC.

        Arguments:
        ----------
        
        self.sym.th_c_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q_c+1)*c_vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (p*n,c_vertices_number)
        
        """
        self.C_th_c_v_eval = ca.Function(
            "C_th_c_v_eval",
            [self.sym.th_c_vertices],
            [
                self.param_eval(
                    self.sys.C,
                    ca.reshape(
                        self.sym.th_c_vertices, self.c_vertices_number, self.q_c + 1
                    ).T,
                )
            ],
        )
        """
        Evaluate self.Ak at the vertices of Theta.

        Arguments:
        ----------
        
        self.sym.th_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1)*vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*n,vertices_number)

        """
        self.Ak_th_v_eval = ca.Function(
            "Ak_th_v_eval",
            [self.sym.th_vertices],
            [
                self.param_eval(
                    self.Ak,
                    ca.reshape(
                        self.sym.th_vertices, self.vertices_number, self.q + 1
                    ).T,
                )
            ],
        )

    def initialize(self, mode="None", constraints=None, P=None):
        """
        Initializes the Offline components of the OCP.

        Arguments:
        ----------

        mode : str
            The mode of linear control policy K to be adopted, which are:
            - "volume" for the volume maximization of the robust control invariant set
            - "LQR" for the LQR control policy
            - "performance" for the control policy that optimize the performance of the closed loop system with the nominal model.
        constraints : list of constraints
            List of additional constraints to be added to the problem.
            the available constraints typesare:
            - pycvxset.Politope()
            - numpy.ndArray()
            - tuple(F,G) as (numpy.ndarray/casadi.MX, numpy.ndarray/casadi.MX)

        Returns:

        self.Ak: ndArray()
            The closed loop transition matrix
        self.poly_x_a: pycvxset.Polytope
            The robust control invariant/contractive set for the closed loop system defined through 'na' halfspace constraints
        self.Hbar: ndArray()
            The array of the matrices H used for the tube evolution constraints, with shape (vertices_number, na, na)
        self.HCbar: ndArray()
            The array of the matrices HC used for the tight constraint, with shape (c_vertices_number, na, number of constraints)


        """

        if constraints is not None:
            self.add_hard_constraints(constraints)

        self.gain_synth = GainSynthesis(
            self.sys.A, self.sys.B, K=self.K,
            Q=self.Q, R=self.R,
            Z_bnd=self.polys.z, W=self.W,
            mode=mode, contraction_factor=self.lam)

        if self.K is None:
            # self._stab_gain(self.sys.A, self.sys.B, self.W.V, mode)
            self.gain_synth.synthesize_controller()

        self.K = self.gain_synth.get_gain()
     
        self.P = P

        if P is None:
            self.P = self.gain_synth.get_terminal_weight()

        #
        self.Ak = self.sys.A + np.einsum("ijk,jl->ilk", self.sys.B, self.K)

        self._init_uncertainty_symbolic()
        # self.Ak_vertices = np.einsum('ikj,lj->ikl', self.Ak, self.theta_vertices)

        # Ak_v = (
        #     self.Ak_th_v_eval(self.theta_vertices.T.reshape(-1, 1))
        #     .toarray()
        #     .reshape(self.n, self.n, -1)
        #     .squeeze()
        # )

        invariant_set = InvariantSet(
            Z_bnd=self.polys.z,
            A=self.A_th_v_eval(self.theta_vertices.T.reshape(-1, 1)).toarray().reshape(self.n, self.n, -1),
            B=self.B_th_v_eval(self.theta_vertices.T.reshape(-1, 1)).toarray().reshape(self.n, self.m, -1),
            K=self.K,
            contractive_factor=self.lam         
        )

        # x0_poly = Polytope(
        #     A=self.polys.z.A @ np.vstack((np.eye(self.n), self.K)), b=self.polys.z.b
        # )
        # # self._lam_contract_set(self.Ak_vertices, x0_poly, self.lam)
        # self._lam_contract_set(Ak_v, x0_poly, self.lam)

        # self.V = Polytope(
        #     A=self.poly_x_aug.A/self.poly_x_aug.b[:, np.newaxis],
        #     b=np.ones(self.poly_x_aug.b.shape)
        #     )

        self.V = invariant_set.compute_invariant_set()

        self.na = self.V.A.shape[0]
        self.V = Polytope(A = self.V.A/self.V.b[:, np.newaxis], b = np.ones(self.na))
        
        self.Hbar = np.zeros((self.na, self.q + 1, self.na))
        self.HCbar = np.zeros((self.q_c + 1, self.na, self.f_const.shape[0]))
        self.w_bar = np.zeros((self.na, 1))
        self.alpha = ca.MX.sym("alpha", self.na, self.N + 1)
        self.alpha_0 = np.ones((self.na, 1))

        self.tube_inclusion()
        self._tight_constraints()

        self._cost_fun_build()
        self._system_build()
        options = {}
        self._set_controller(options)

    def solve(self, x0, r=None):
        if r is None:
            r = np.zeros(self.sym.r.shape)
        # if th_center is None:
        #     th_center = [np.zeros((self.q, 1)), np.zeros((self.q_c, 1))]
        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))
        # th = np.block([[1], [th_center[0]]])
        # th_c = np.block([[1], [th_center[1]]])

        if self.first_time:
            self.sol = self.qpsol(
                p=ca.vertcat(
                    x0,
                    r,
                    self.th,
                    self.th_c,
                    self.theta_vertices.T.flatten()[:, np.newaxis],
                    self.theta_c_vertices.T.flatten()[:, np.newaxis],
                ),
                lbg=new_lbg,
                ubg=new_ubg,
            )
            self.first_time=False
        else:
            x_warm, _ = self._warm_start()
            self.sol = self.qpsol(
                p=ca.vertcat(
                    x0,
                    r,
                    self.th,
                    self.th_c,
                    self.theta_vertices.T.flatten()[:, np.newaxis],
                    self.theta_c_vertices.T.flatten()[:, np.newaxis],
                ),
                lbg=new_lbg,
                ubg=new_ubg,
                x0=x_warm,
            )

        self.u_star = self.sol["x"][
            (self.N + 1) * self.n : (self.N + 1) * self.n + self.m
        ]

        _, alpha_idx = super()._warm_start()

        self.alpha_0 = self.sol["x"][alpha_idx: alpha_idx + self.na].toarray().copy()

        if self.track:
            xa_idx = (self.N + 1) * self.n + self.N * self.m
            self.xa_star = (
                self.sol["x"][xa_idx: xa_idx + self.n * self.sym.xa.shape[1]]
                .toarray()
                .reshape((self.n, -1), order="F")
            )
        else:
            self.xa_star = None

        # self._warm_start()

    def _system_build(self):
        """ """

        """
        Evaluate Hbar at the vertices of Theta.

        Arguments:
        ----------
        
        self.sym.th_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q+1)*vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (na*n,vertices_number)

        """
        H_eval = ca.Function(
            "H_eval",
            [self.sym.th_vertices],
            [
                self.param_eval(
                    self.Hbar,
                    ca.reshape(
                        self.sym.th_vertices, self.vertices_number, self.q + 1
                    ).T,
                )
            ],
        )

        """
        Evaluate HCbar at the vertices of Theta_c.

        Arguments:
        ----------
        
        self.sym.th_c_vertices: casadi.MX, ndArray()
            Symbolic or Numeric Array with shape ((q_c+1)*c_vertices_number, 1)
        Returns:
        --------

        casadi.DM or casadi.MX with shape (n*n,c_vertices_number)

        """
        HC_eval = ca.Function(
            "HC_eval",
            [self.sym.th_c_vertices],
            [
                self.param_eval(
                    self.HCbar,
                    ca.reshape(
                        self.sym.th_c_vertices, self.c_vertices_number, self.q_c + 1
                    ).T,
                )
            ],
        )

        matAk = self.Ak_th_eval(self.sym.th)

        matA = self.A_th_eval(self.sym.th)

        matB = self.B_th_eval(self.sym.th)

        matGb = np.vstack(
            (
                self.g_const,
                np.zeros((np.size(self.fc_const, 0), np.size(self.g_const, 1))),
            )
        )

        sym_A = ca.MX.sym("_A", self.n, self.n)
        sym_B = ca.MX.sym("_B", self.n, self.m)
        _xa_next = ca.MX.sym("_xa_next", self.n, 1)
        _xa = ca.MX.sym("_xa", self.n, 1)
        _ua = ca.MX.sym("_ua", self.m, 1)
        _alpha = ca.MX.sym("_alpha", self.na, 1)
        _alpha_next = ca.MX.sym("_alpha_next", self.na, 1)

        step = ca.Function(
            "step",
            [self.sym.get_x(), self.sym.get_u(), sym_A, sym_B],
            [sym_A @ self.sym.get_x() + sym_B @ self.sym.get_u()],
        )

        linear_k = ca.Function(
            "linearK", [self.sym.get_x()], [self.K @ self.sym.get_x()]
        )

        reacheability = ca.Function(
            "reachibility",
            [self.sym.get_x(), self.sym.get_u()],
            [
                self.polys.zs.A @ ca.vertcat(self.sym.get_x(), self.sym.get_u())
                - self.polys.zs.b
            ],
        )

        tubeStep = ca.Function(
            "tubeStep",
            [
                self.sym.th_vertices,
                self.sym.get_c(),
                _xa_next,
                _xa,
                _alpha,
                _alpha_next,
            ],
            [
                ca.reshape(H_eval(self.sym.th_vertices)[:, v], -1, self.na).T @ _alpha
                + self.V.A
                @ (
                    _xa_next
                    - ca.reshape(
                        self.Ak_th_v_eval(self.sym.th_vertices)[:, v], -1, self.n
                    ).T
                    @ _xa
                    + ca.reshape(
                        self.B_th_v_eval(self.sym.th_vertices)[:, v], -1, self.n
                    ).T
                    @ self.sym.get_c()
                )
                + self.w_bar
                - _alpha_next
                for v in range(self.vertices_number)
            ],
        )

        tightStep = ca.Function(
            "tightStep",
            [self.sym.th_c_vertices, self.sym.get_u(), _xa, _ua, _alpha],
            [
                ca.reshape(HC_eval(self.sym.th_c_vertices)[:, v], self.na, -1).T
                @ _alpha
                + matGb @ (linear_k(_xa) + self.sym.get_u() + _ua)
                - np.ones((matGb.shape[0], 1))
                for v in range(self.c_vertices_number)
            ],
        )

        artificial_idx = (
            (slice(self.N), slice(1, self.N + 1))
            if (self.track and self.sym.r.shape[1] > 1)
            else (-1, -1)
        )

        initial = self.sym.x[:, 0] - self.sym.x_init  # Initial condition

        initialTube = self.V.A @ self.sym.x[:, 0] - self.alpha[:, 0]

        artifEquil = step(
            self.sym.xa[:, -1],
            self.sym.ua[:, -1],
            ca.reshape(matA[:, -1], self.n, self.n).T - np.eye(self.n),
            ca.reshape(matB[:, -1], self.m, self.n).T,
        )

        dynamics = self.sym.x[:, 1:] - (
            step.map(self.N)(
                self.sym.x[:, : self.N] - self.sym.xa[:, artificial_idx[0]],
                self.c,
                ca.reshape(matAk, self.n, self.n).T,
                ca.reshape(matB, self.m, self.n).T,
            )
            + step.map(self.sym.ua.shape[1])(
                self.sym.xa[:, artificial_idx[0]],
                self.sym.ua,
                ca.reshape(matA[:, -1], self.n, self.n).T,
                ca.reshape(matB[:, -1], self.m, self.n).T,
            )
        )

        input_policy = (
            self.sym.u
            - linear_k.map(self.N)(
                self.sym.x[:, : self.N] - self.sym.xa[:, artificial_idx[0]]
            )
            - (self.c + self.sym.ua)
        )

        tightConstraint = tightStep.map(self.N)(
            self.sym.th_c_vertices_N[:, :-1],
            self.c,
            self.sym.xa[:, artificial_idx[0]],
            self.sym.ua,
            self.alpha[:, : self.N],
        )

        terminalConstraint = tightStep(
            self.sym.th_c_vertices_N[:, -1],
            np.zeros((self.m, 1)),
            self.sym.xa[:, -1],
            self.sym.ua[:, -1],
            self.alpha[:, -1],
        )

        if self.sym.r.shape[1] > 1:
            tubeDyn = tubeStep.map(self.N)(
                self.sym.th_vertices_N[:, :-1],
                self.c,
                self.sym.xa[:, artificial_idx[1]],
                self.sym.xa[:, artificial_idx[0]],
                self.alpha[:, : self.N],
                self.alpha[:, 1:],
            )

            artificialTrajectory = self.sym.xa[:, 1:] - step.map(self.N)(
                self.sym.xa[:, : self.N],
                self.sym.ua,
                ca.reshape(matA[:, -1], self.n, self.n).T,
                ca.reshape(matB[:, -1], self.m, self.n).T,
            )
            self.g.append(artificialTrajectory)
            self.lbg.append([0.0] * (self.n * self.N))
            self.ubg.append([0.0] * (self.n * self.N))

        else:
            tubeDyn = tubeStep.map(self.N)(
                self.sym.th_vertices_N[:, :-1],
                self.c,
                self.sym.xa,
                self.sym.xa,
                self.alpha[:, : self.N],
                self.alpha[:, 1:],
            )

        terminalTube = tubeStep(
            self.sym.th_vertices_N[:, -1],
            np.zeros((self.m, 1)),
            self.sym.xa[:, -1],
            self.sym.xa[:, -1],
            self.alpha[:, -1],
            self.alpha[:, -1],
        )

        self.g.append(
            ca.reshape(
                reacheability.map(self.sym.ua.shape[1])(
                    self.sym.xa[:, artificial_idx[0]], self.sym.ua
                ),
                (1, -1),
            ).T
        )

        self.lbg.append([-ca.inf] * (self.polys.zs.A.shape[0] * self.sym.ua.shape[1]))
        self.ubg.append([0.0] * (self.polys.zs.A.shape[0] * self.sym.ua.shape[1]))

        self.g.append(initial)
        self.lbg.append([0.0] * (self.n))
        self.ubg.append([0.0] * (self.n))

        self.g.append(artifEquil)
        self.lbg.append([0.0] * (self.n))
        self.ubg.append([0.0] * (self.n))

        self.g.append(ca.reshape(dynamics, (1, -1)).T)
        self.lbg.append([0.0] * (self.n * self.N))
        self.ubg.append([0.0] * (self.n * self.N))

        self.g.append(ca.reshape(input_policy, (1, -1)).T)
        self.lbg.append([0.0] * (self.m * self.N))
        self.ubg.append([0.0] * (self.m * self.N))

        self.g.append(initialTube)
        self.lbg.append([-ca.inf] * (len(self.V.b)))
        self.ubg.append([0.0] * (len(self.V.b)))

        for tubeVertex in tubeDyn:
            self.g.append(ca.reshape(tubeVertex, 1, -1).T)
            self.lbg.append([-ca.inf] * (self.na * self.N))
            self.ubg.append([0.0] * (self.na * self.N))

        if self.track:
            for tubeVertex in terminalTube:
                self.g.append(ca.reshape(tubeVertex, 1, -1).T)
                self.lbg.append([-ca.inf] * (self.na))
                self.ubg.append([0.0] * (self.na))

        if not isinstance(tightConstraint, tuple):
            tightConstraint = (tightConstraint,)

        for constraintVertex in tightConstraint:
            self.g.append(ca.reshape(constraintVertex, 1, -1).T)
            self.lbg.append([-ca.inf] * (matGb.shape[0] * self.N))
            self.ubg.append([0.0] * (matGb.shape[0] * self.N))

        if not isinstance(terminalConstraint, tuple):
            terminalConstraint = (terminalConstraint,)

        for constraintVertex in terminalConstraint:
            self.g.append(ca.reshape(constraintVertex, 1, -1).T)
            self.lbg.append([-ca.inf] * (matGb.shape[0]))
            self.ubg.append([0.0] * (matGb.shape[0]))

    def _set_controller(self, options=None):
        if options is None:
            options = {}

        options["error_on_fail"] = False  # TEMP: disabled to let QP failures raise
        options["osqp"] = {
            'verbose': False,
            'max_iter': 1000,
            'polish': False
            }
        options['jit'] = True
        options['print_time'] = False
        options['record_time'] = True
        options['compiler'] = 'shell'
        options['jit_options'] = {
            'flags': ['-O3', 'march=native'],}
        

        decision_vars = ca.vertcat(
            self.sym.x.reshape((-1, 1)), self.sym.u.reshape((-1, 1))
        )

        if self.track:
            decision_vars = ca.vertcat(
                decision_vars,
                ca.reshape(self.sym.xa, -1, 1),
                ca.reshape(self.sym.ua, -1, 1),
            )

        decision_vars = ca.vertcat(
            decision_vars, self.alpha.reshape((-1, 1)), self.c.reshape((-1, 1))
        )

        # if self.track:
        #     decision_vars = ca.vertcat(decision_vars, ca.reshape(self.nu, -1, 1))
        new_g = ca.MX()
        for val in self.g:
            if isinstance(val, list):
                for vval in val:
                    new_g = ca.vertcat(new_g, vval)
            else:
                new_g = ca.vertcat(new_g, val.reshape((-1, 1)))
        self.g = new_g

        new_lbg = []
        new_ubg = []
        for i, val in enumerate(self.lbg):
            new_lbg = np.concatenate((new_lbg, val))
            new_ubg = np.concatenate((new_ubg, self.ubg[i]))

        qp = {
            "x": decision_vars,
            "f": self.cost_fun,
            "g": self.g,
            "p": ca.vertcat(
                self.sym.x_init,
                ca.reshape(self.sym.r, (1, -1)).T,
                ca.reshape(self.sym.th, (1, -1)).T, # TODO - ca.reshape(self.sym.th_N, (1, -1)).T,
                ca.reshape(self.sym.th_c, (1, -1)).T,
                ca.reshape(self.sym.th_vertices_N, (1, -1)).T,
                ca.reshape(self.sym.th_c_vertices_N, (1, -1)).T,
            ),
        }
        self.qpsol = ca.qpsol("qpsol", self.options.solver, qp, options)

        _ = self.qpsol(
            p=ca.DM.zeros(qp['p'].shape[0], 1),
            lbg=new_lbg,
            ubg=new_ubg,)
        
        self.new_lbg_dm = ca.DM(new_lbg.copy())
        self.new_ubg_dm = ca.DM(new_ubg.copy())


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

        C_v = (
            self.C_th_c_v_eval(self.theta_c_vertices.T.reshape(-1, 1))
            .toarray()
            .reshape(self.p, self.n, self.c_vertices_number)
        )

        # C_vertices = np.einsum('ijk,kl->ijl', self.sys.C, self.theta_c_vertices)

        Fb = np.vstack(
            (
                np.repeat(
                    self.f_const[:, :, np.newaxis], (self.c_vertices_number), axis=2
                ),
                np.einsum("ij,jkl->ikl", self.fc_const, C_v),
            )
        )
        Gb = np.vstack(
            (
                self.g_const,
                np.zeros((np.size(self.fc_const, 0), np.size(self.g_const, 1))),
            )
        )
        DF = Fb[:, :, 0] + Gb @ self.K
        if self.q_c > 0:
            DF = np.vstack((DF, Fb[:, :, 2:-1] - Fb[:, :, 1]))

        Aineq = np.empty((0, self.na))
        bineq = np.zeros((self.na * self.c_vertices_number, 1))
        for k in range(self.c_vertices_number):
            Aineq = np.vstack(
                (Aineq, -np.kron(np.eye(self.na), self.theta_c_vertices[k, :]))
            )
        x_cp = cp.Variable((Aineq.shape[0], 1))
        Aeq = np.kron(self.V.A.T, np.eye(self.q_c + 1))
        beq = np.reshape(
            DF[:, :, np.newaxis].transpose(2, 1, 0),
            ((self.q_c + 1) * self.n, 1, np.size(DF, 0)),
        )
        # TODO move constraint in the loop evaluating constraint for each beq[:, :, i] element
        options = None  # TODO - check mskoptimset equivalence and function
        for i in range(np.size(Fb, 0)):
            constraints = [Aineq @ x_cp <= bineq]
            constraints += [Aeq @ x_cp == beq[:, :, i]]
            f = np.zeros(self.c_vertices_number)
            for j in range(self.c_vertices_number):
                cost = np.kron(np.ones((1, self.na)), self.theta_c_vertices[j, :])
                lp = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
                lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR)
                f[j] = lp.value
                hbar = x_cp.value
                if f[j] >= np.max(f):
                    self.HCbar[:, :, i] = np.reshape(hbar, (self.q_c + 1, self.na))
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

        w_cp = cp.Variable((self.n, 1))
        w_constraints = [self.W.A @ w_cp <= self.W.b]

        for i in range(self.na):
            w_lp = cp.Problem(cp.Minimize(-self.V.A[i, :] @ w_cp), w_constraints)
            w_lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR)
            self.w_bar[i, :] = w_lp.value

        for i in range(self.na):
            f = np.zeros(self.vertices_number)
            for j in range(self.vertices_number):
                Aineq = -np.kron(np.eye(self.na), self.theta_vertices[j, :])
                bineq = np.zeros((self.na, 1))

                Aeq = np.kron(self.V.A.T, np.eye(self.q + 1))
                beq = np.reshape(
                    np.einsum("ij,ljk->ijk", self.V.A, self.Ak).transpose(2, 1, 0),
                    ((self.q + 1) * self.n, 1, self.na),
                )

                x_cp = cp.Variable((Aeq.shape[1], 1))

                constraints = [Aineq @ x_cp <= bineq, Aeq @ x_cp == beq[:, :, i]]
                cost = np.kron(
                    np.ones((self.na, 1)), self.theta_vertices[j, :, np.newaxis]
                ).T
                lp = cp.Problem(cp.Minimize(cost @ x_cp), constraints)
                lp.solve(solver=constants.DEFAULT_LP_SOLVER_STR)
                f[j] = lp.value
                hbar = x_cp.value
                if f[j] >= np.max(f):
                    self.Hbar[:, :, i] = np.reshape(hbar, (self.q + 1, self.na)).T
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
                H = np.einsum("ikj...,->ikj...", a, th)
            else:
                H = np.einsum("ijk,...k->ij...", a, th)
        elif isinstance(th, ca.MX):
            H = np.reshape(a, (a.shape[0] * a.shape[1], th.shape[0])) @ th
            # H = 0
            # for i in range(th.shape[0]):
            #     H += th[i] * a[:, :, i]
        return H

    def set_param_set(self, theta):
        theta_vertices = np.hstack((np.ones((np.size(theta.V, 0), 1)), theta.V))
        vertices_number = np.size(theta_vertices, 0)
        return theta_vertices, vertices_number

    def _warm_start(self):

        warm_start, last_idx = super()._warm_start()

        dec_alpha = self.sol["x"][
            last_idx + self.na : last_idx + self.na * (self.N + 1)
        ].toarray()
        dec_alpha = np.block([[dec_alpha],[np.zeros((self.na, 1))]])

        last_idx = last_idx + self.na * (self.N + 1)

        dec_c = self.sol["x"][last_idx + self.m : last_idx + self.m * self.N].toarray()
        dec_c = np.block([[dec_c],[np.zeros((self.m, 1))]])

        last_idx = last_idx + self.m * self.N

        return np.vstack((warm_start, dec_alpha, dec_c)), last_idx

    def initial_tube(self):

        A_mat = self.V.A.copy()
        b_vec = self.alpha_0.copy()

        return Polytope(A=A_mat, b=b_vec)
