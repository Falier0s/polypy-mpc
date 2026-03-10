import numpy as np
import casadi as ca 
from pycvxset import Polytope
from itertools import product
import os
import csv
from datetime import datetime

# Utility function to compute the Cartesian product of polytopes


def cartesian_product(*polytopes):
    """Compute the Cartesian product of multiple polytopes"""
    V_rep = [p.V for p in polytopes]
    combinations = []
    for vertices in product(*V_rep):
        combinations.append(np.concatenate(vertices))
    return Polytope(V=np.array(combinations))


class cMPC:
    """
    Linear Quadratic MPC using CasADi

    This class implements a Model Predictive Controller for linear discrete-time systems
    using CasADi for optimization. It supports both regulation and tracking problems.

    Similar to the MATLAB yMPC class but using CasADi (Python) instead of YALMIP (MATLAB).
    """

    def __init__(self, sys, Q, R, N, options=None):
        """
        Initialize the MPC controller

        Parameters
        ----------
        sys : object
            System object with attributes A (state matrix), B (input matrix), C (output matrix)
        Q : np.ndarray
            State cost matrix
        R : np.ndarray
            Input cost matrix
        N : int
            Prediction horizon
        options : dict, optional
            Dictionary of options for configuration
        """

        if options is None:
            options = {}

        # System matrices initialization
        if options.get('Ts') is None or options['Ts'] == 0:
            self.A = sys.A
            self.B = sys.B
            self.C = sys.C
            self.Ts = 0
        else:
            I = np.zeros((len(sys.A[0]), len(sys.A[0]), len(sys.A[2])))
            I[:][:][0] = np.eye(len(sys.A[0]))

            self.Ts = options['Ts']
            self.A = self.Ts*sys.A+I
            self.B = self.Ts*sys.B
            self.C = sys.C

        # Prediction horizon initialization
        self.N = N

        # System dimensions
        self.n = len(self.A[0])
        self.m = len(self.B[1])
        self.p = len(self.C[0])

        # Decision Variables - TODO - check this with Fabio
        self.x = ca.MX.sym('x', np.matlib.repmat(self.n, 1, self.N), np.ones((1, self.N)))
        self.u = ca.MX.sym('u', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))

        self.z = ca.MX.sym('z', self.n+self.m, 1)

        self.J = 0  # TODO - I think Fabio said that it wasn't mandatory

        # Cost matrices initialization
        self.Q = Q
        self.R = R
        self.S = Q*1e-4
        self.T = R*1e-4

        # Check for options

        # If no solver defined use quadprog
        if options.get('solver') is None or options['solver'] == '':
            self.solver = 'quadprog'
        else:
            self.solver = options['solver']

        # If no verbose defined use 0
        if options.get('verbose') is None or options['verbose'] == '':
            self.verbose = 0
        else:
            self.verbose = options['verbose']
        if options.get('relax') is None or options['relax'] == '':
            self.relax = False
        else:
            self.relax = options['relax']

        # Define reference signal and the artificial equilibrium variable as the last in the state and input sequence

        # If no reference defined use 0
        if options.get('ref') is not None and options['ref'].all():
            self.track = True
            if options['ref'].lower() == 'traj' or options['ref'].lower() == 'trajectory':
                self.xa = ca.MX.sym('xa', np.matlib.repmat(self.n, 1, self.N+1), np.ones((1, self.N+1)))
                self.ua = ca.MX.sym('ua', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))
                self. xr = ca.MX.sym('xr', np.matlib.repmat(self.n, 1, self.N+1), np.ones((1, self.N+1)))
                self.ur = ca.MX.sym('ur', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))
                self.r = ca.MX.sym('r', self.p, 1)
            else:
                self.xa = ca.MX.sym('xa', self.n, 1)
                self.ua = ca.MX.sym('ua', self.m, 1)
                self.xr = ca.MX.sym('xr', self.n, 1)
                self.ur = ca.MX.sym('ur', self.m, 1)
                self.r = ca.MX.sym('r', self.p, 1)
        else:
            self.track = False
            self.xa = np.zeros((self.n, 1))
            self.ua = np.zeros((self.m, 1))
            self.xr = np.zeros((self.n, 1))
            self.ur = np.zeros((self.m, 1))
            self.r = ca.MX.sym('r', self.p, 1)

        # If the name of the controller is not defined, assign a random name to it

        if options.get('name') is not None and options['name'] != '':
            self.name = options['name']
        else:
            self.name = type(self).__name__+'_'+str(np.random.randint(1000))

        # If no SVD flag defined use False (i.e. no SVD decomposition of the tracking problem)

        if options.get('svd') is not None and options['svd'] != '' and self.track:
            self.svd_flag = options['svd']
        else:
            self.svd_flag = False

        # If no control horizon defined use the control horizon copies the prediction horizon

        if options.get('Nc') is not None and options['Nc'] != 0:
            self.Nc = options['Nc']
        else:
            self.Nc = self.N

        # If no sigma for reachibility set definition, use arbitrarly large 0.95 (should be 0<sigma<1)

        if options.get('sigma') is not None and options['sigma'] != 0:
            self.sigma = options['sigma']
        else:
            self.sigma = 0.95

        # If no stabilizing gain for the terminal constraint is defined, use 0 (i.e. no terminal constraint)

        if options.get('K') is None or options['K'] == 0:
            self.K = np.zeros((self.m, self.n))
        else:
            self.K = options['K']

        # define Upper and lower bounds for state variables

        if options.get('xBound') is not None and options['xBound'] != 0:
            lxb = np.copy(options['xBound'][0])
            uxb = np.copy(options['xBound'][1])
        else:
            lxb = np.full((self.n, 1), -np.inf)
            uxb = np.full((self.n, 1), np.inf)

        # define Upper and lower bounds for input variables

        if options.get('uBound') is not None and options['uBound'] != 0:   
            lub = np.copy(options['uBound'][0])
            uub = np.copy(options['uBound'][1])
        else:
            lub = np.full((self.m, 1), -np.inf)
            uub = np.full((self.m, 1), np.inf)

        # define Upper and lower bounds for output variables

        if options.get('yBound') is not None and options['yBound'] != 0:
            lyb = np.copy(options['yBound'][0])
            uyb = np.copy(options['yBound'][1]) 
        else:
            lyb = np.full((self.p, 1), -np.inf)
            uyb = np.full((self.p, 1), np.inf)

        # Initialize Polyhedral set for the state and input constraints
        self.X = Polytope(A=np.vstack((np.eye(self.n), -np.eye(self.n))), b=np.vstack((uxb, -lxb)))
        self.U = Polytope(A=np.vstack((np.eye(self.m), -np.eye(self.m))), b=np.vstack((uub, -lub)))
        self.Y = Polytope(A=np.vstack((np.eye(self.p), -np.eye(self.p))), b=np.vstack((uyb, -lyb)))
        # Compute the set of (x,u) pairs Z such a Carthesian product of X and U 

        self.Z = cartesian_product(self.X, self.U)

        if self.track:
            self.Zs = self.sigma*self.Z

        # TODO - Output Constraint set 

        self.time = 0

        # TODO - Define the class functions

        def add_hard_constraints(self,*hConstraints):
            pass
        def add_soft_constraints(self,*sConstraints):
            pass

        def initialize(self, mode):

            # Call the hard constraint builder function
            self.add_hard_constraints()

            
            # Check if the stabilizing gain is defined, otherwise compute it
            
            if not self.K.any():
                self._stab_gain(self.A, self.B, [], mode)

            # Compute the closed loop matrix Ak
            
            self.Ak = self.A + self.B @ self.K


            # Check if the SVD should be applied to reduce the tracking problem (number of artificial variables)

            if not self.svd_flag:

                # If decomposition is not required just use eye for the transformation matrix Mn

                self.Mn = np.eye((self.n+self.m))
            else:

                # Otherwise compute the transformation matrix, than rebuild

                self._svd_decomposition()
                self.Mx = np.column_stack((np.eye(self.n), np.zeros((self.n, self.m)))) @ self.Mn

            # For the Terminal set computation [Avlvarado PhD Thesis Chapter 3.3.2]
            # Compute the Augmented System (with artificial variables or reduced artificial variables with svd -->  [x; x_a; x_u])


            self.L = np.column_stack((-self.K, np.eye(self.m))) @ self.Mn
            Aa1 = np.column_stack((self.Ak, self.B @ self.L))
            Aa2 = np.column_stack((np.zeros((np.size(self.Mn, 1), self.n)), np.eye(np.size(self.Mn, 1))))
            Aa = np.vstack((Aa1, Aa2))

            # Define The initial Polyhedron for the computation of the maximal admissible invariant set as terminal set

            poly_A1 = self.Z.A @ np.vstack((np.column_stack((np.eye(self.n), np.zeros((self.n, np.size(self.Mn, 1))))), np.column_stack((self.K, self.L))))
            poly_A2 = np.column_stack((np.zeros((np.size(self.Z.A, 0), self.n)), self.Zs.A@self.Mn))
            Poly_A = np.vstack((poly_A1, poly_A2))
            Poly_b = np.vstack((self.Z.b, self.Zs.b))
            
            Poly = Polytope(A=Poly_A, b=Poly_b)

            # Call the lambda-contractive set computation but using lambda as 1 (invariant)

            self._lam_contract_set(Aa, Poly, 1)


            self.partSol = False  # Maybe it is related to the DMPC (not to be changed)


            self._cost_fun_build()
            self._constraint_build()

            opt = {} # TODO - Fill the dictionary with options for the qp solver

            self._set_controller(x0=None, r=None, opt=opt)

        def solve(self, x0, r=None):
            self.xo = x0

            if r is None:
                r = 0
            Sol, _, _, _, _, Data = self.controller(self.x0, r)  # TODO - check this with Fabio
            self._warm_start(Sol)
            self.time = Data.solvertime  # TODO - check this with Fabio
            self.u_star = Sol[0]  # TODO - check this with Fabio
            self.J_star = Sol[1]  # TODO - check this with Fabio
            self._verbose_print()

        def _cost_fun_build(self):
            pass

        def _constraint_build(self):
            pass

        def _stab_gain(self,A,B,w_v,cost):
            pass

        def _svd_decomposition(self):
            pass

        def _lam_contract_set(self,Aa,X0,lam):
            pass

        def _set_controller(self, x0, r=None):
            pass

        def _verbose_print(self):
            pass

        def _save_results(self, results, filename=None):
            pass

        def _warm_start(self, previous_solution):
            pass
