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
        if symbolic:
            if traj:
                self.xa = ca.MX.sym("xa", n, N + 1)
                self.ua = ca.MX.sym("ua", m, N)
                self.r = ca.MX.sym('r', p, N+1)
            else:
                self.xa = ca.MX.sym("xa", n, 1)
                self.ua = ca.MX.sym("ua", m, 1)
                self.r = ca.MX.sym('r', p, 1)
        else:
            self.xa = np.zeros((n, 1))
            self.ua = np.zeros((m, 1))
            self.r = ca.MX.sym("r", p, 1)

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
