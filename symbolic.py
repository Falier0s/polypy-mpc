import casadi as ca
import numpy as np


class Symbolic:
    """A simple wrapper for CasADi symbolic variables and related operations."""

    def __init__(self, n, m, N):
        self.x = ca.MX.sym("x", n, N + 1)
        self.u = ca.MX.sym("u", m, N)

        self._x = ca.MX.sym("_x", n)
        self._u = ca.MX.sym("_u", m)

        self._z = ca.vertcat(self._x, self._u)
        self.x_init = ca.MX.sym("x_init", n)
        self.z = ca.vertcat(self.x[:, :N], self.u)

        self.xa = None
        self.ua = None
        self.r = None

        self.th = None
        self.th_c = None

        self.th_N = None
        self.th_c_N = None

        self.th_vertices = None
        self.th_c_vertices = None

        self.th_vertices_N = None
        self.th_c_vertices_N = None

    def create_tracking_variables(self, n, m, p, N, symbolic=True, traj=True):
        """Create the tracking variables

        Args:
            n (int): number of states
            m (int): number of inputs
            p (int): number of outputs
            N (int): horizon
            symbolic (bool, optional): if you should create some symbolic variables.\
                Defaults to True.
            traj (bool, optional): if there is some tracking or not. Defaults to True.
        """

        lenght = (N + 1, N) if traj else (1, 1)
        
        if symbolic:    
            self.xa = ca.MX.sym("xa", n, lenght[0])
            self.ua = ca.MX.sym("ua", m, lenght[1])
            self.r = ca.MX.sym('r', p, lenght[0])
        else:
            self.xa = np.zeros((n, 1))
            self.ua = np.zeros((m, 1))
            self.r = ca.MX.sym("r", p, 1)
    
    def create_parameter_variables(self, q, q_c, q_vert_num, q_c_vert_num, length=1):
        """Create the parameter variables

        Arguments:
        -----
        q (int): 
            number of parameters for the system matrices A and B
        q_c (int): 
            number of parameters for the output matrix C
        q_vert_num (int): 
            number of vertices for the parameter set of A and B
        q_c_vert_num (int): 
            number of vertices for the parameter set of C
        length (int): 
            The length of the symbolic variables for the uncertain parameters evolving over steps.
            Default is 1 for the case of no parameters variation.

        """

        self.th = ca.MX.sym("th", q+1, 1)
        self.th_c = ca.MX.sym("th_c", q_c+1, 1)

        self.th_vertices = ca.MX.sym("th_vertices", (q+1)*q_vert_num, 1)
        self.th_c_vertices = ca.MX.sym("th_c_vertices", (q_c+1)*q_c_vert_num, 1)

        self.th_N = ca.MX.sym("th", q+1, length)
        self.th_c_N = ca.MX.sym("th_c", q_c+1, length)

        self.th_vertices_N = ca.MX.sym("th_vertices_N", (q+1)*q_vert_num, length)
        self.th_c_vertices_N = ca.MX.sym("th_c_vertices_N", (q_c+1)*q_c_vert_num, length)

    def get_x(self):
        """Get the symbolic variable for states."""
        return self._x

    def get_u(self):
        """Get the symbolic variable for inputs

        Returns:
            casadi.MX: the symbolic
        """
        return self._u

    def get_z(self):
        """Get the syòbolic variable for outputs

        Returns:
            casadi.MX: the symbolic
        """
        return self._z
    
    def get_sym(self,n):
        """Get the symbolic variables for states and inputs

        Returns:
            tuple: the symbolic variables for states and inputs
        """
        return self._x, self._u
