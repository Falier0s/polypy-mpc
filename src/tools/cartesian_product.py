from itertools import product

import numpy as np

from pycvxset import Polytope


def cartesian_product(*polytopes):
    """Compute the Cartesian product of multiple polytopes.

    The Cartesian product of polytopes P1, P2, ..., Pk is the set of all
    concatenated vertex combinations from each input polytope.

    Parameters
    ----------
    *polytopes : Polytope
        Input polytopes to combine.

    Returns
    -------
    Polytope
        A polytope representing the Cartesian product.
    """

    # Get vertex representations from each polytope
    v_rep = [p.V for p in polytopes]

    # Build all possible concatenations of vertices
    combinations = []
    for vertices in product(*v_rep):
        combinations.append(np.concatenate(vertices))

    # Return as a new Polytope
    return Polytope(V=np.array(combinations))
