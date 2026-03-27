import numpy as np

from pycvxset import Polytope

from cartesian_product import cartesian_product


class PolytopeSystem:
    """A simple wrapper for polytopic systems and related operations."""

    def __init__(self, state_size, input_size, output_size, bounds):
        uxb, lxb, uub, lub, uyb, lyb = bounds
        self.x = Polytope(A=np.vstack((np.eye(state_size), -np.eye(state_size))),
                          b=np.concatenate((uxb, -lxb)))
        self.y = Polytope(A=np.vstack((np.eye(output_size), -np.eye(output_size))),
                          b=np.vstack((uyb, -lyb)))
        self.u = Polytope(A=np.vstack((np.eye(input_size), -np.eye(input_size))),
                          b=np.concatenate((uub, -lub)))
        self.z = cartesian_product(self.x, self.u)
        self.zc = None
        self.zs = None
