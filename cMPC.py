import numpy as np
from casadi import MX
from pycvxset import Polytope

class cMPC():
    def __init__(self, sys, Q, R, N, options):
        # Initialize the MPC object with the system matrices and the cost

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
        self.x = MX.sym('x', np.matlib.repmat(self.n, 1, self.N+1), np.ones((1, self.N+1)))
        self.u = MX.sym('u', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))

        self.z = MX.sym('z', self.n+self.m, 1)

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
                self.xa = MX.sym('xa', np.matlib.repmat(self.n, 1, self.N+1), np.ones((1, self.N+1)))
                self.ua = MX.sym('ua', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))
                self. xr = MX.sym('xr', np.matlib.repmat(self.n, 1, self.N+1), np.ones((1, self.N+1)))
                self.ur = MX.sym('ur', np.matlib.repmat(self.m, 1, self.N), np.ones((1, self.N)))
                self.r = MX.sym('r', self.p, 1)
            else:
                self.xa = MX.sym('xa', self.n, 1)
                self.ua = MX.sym('ua', self.m, 1)
                self.xr = MX.sym('xr', self.n, 1)
                self.ur = MX.sym('ur', self.m, 1)
                self.r = MX.sym('r', self.p, 1)
        else:
            self.track = False
            self.xa = np.zeros((self.n, 1))
            self.ua = np.zeros((self.m, 1))
            self.xr = np.zeros((self.n, 1))
            self.ur = np.zeros((self.m, 1))
            self.r = MX.sym('r', self.p, 1)

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

        self.X_poly = cdd.Polyhedron(cdd.Matrix(np.hstack((np.vstack((np.eye(self.n), -np.eye(self.n))), np.vstack((uxb, -lxb)))), dtype=np.float64))
        self.U_poly = cdd.Polyhedron(cdd.Matrix(np.hstack((np.vstack((np.eye(self.m), -np.eye(self.m))), np.vstack((uub, -lub)))), dtype=np.float64))
        self.Y_poly = cdd.Polyhedron(cdd.Matrix(np.hstack((np.vstack((np.eye(self.p), -np.eye(self.p))), np.vstack((uyb, -lyb)))), dtype=np.float64))

        # Compute the set of (x,u) pairs Z such a Carthesian product of X and U 


