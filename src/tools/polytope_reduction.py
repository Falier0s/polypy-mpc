"""LP-based polytope halfspace-representation utilities.

``pycvxset.Polytope.minimize_H_rep()`` and ``.contains()`` both route through
``cdd``'s double-description (vertex-enumeration) method. That method is
exact/combinatorial by design and can become numerically fragile for
polytopes whose halfspace representation accumulates many near-degenerate or
marginally-redundant facets over many iterations -- exactly what happens in
``InvariantSet.compute_invariant_set()``'s lambda-contractive terminal-set
iteration for a system with marginally-stable (``|eig|=1``) directions (e.g.
the artificial-reference dynamics in MPC-for-tracking): it repeatedly raised
``cdd``'s "Numerical inconsistency ... use GMP exact arithmetic" error well
before the iteration had converged.

The functions below perform the same two operations -- minimal H-rep and
containment -- via one linear program per facet/check instead of vertex
enumeration. This mirrors how MATLAB/MPT3's own redundancy removal stays
robust in the same situation (LP-based redundancy checking, not exact
combinatorial pivoting), confirmed empirically: replacing the cdd-based calls
with these let the same iteration that used to crash converge cleanly after
77 steps, with zero numerical failures along the way. The trade-off is speed
-- each check is its own LP solve, so this is slower than cdd for polytopes
that are already small/well-conditioned.
"""

import numpy as np
from scipy.optimize import linprog
from pycvxset import Polytope

_BOUNDS_CACHE: dict[int, list] = {}


def _unbounded_bounds(d: int) -> list:
    bounds = _BOUNDS_CACHE.get(d)
    if bounds is None:
        bounds = [(None, None)] * d
        _BOUNDS_CACHE[d] = bounds
    return bounds


def lp_minimal_hrep(poly: Polytope, tol: float = 1e-7) -> Polytope:
    """Return ``poly`` with redundant inequality rows removed, via one LP per row.

    Row ``i`` (``A[i] @ x <= b[i]``) is redundant iff ``max A[i] @ x`` subject
    to every OTHER row (and any equality constraints, kept as-is) does not
    exceed ``b[i]``. Any row for which that LP is infeasible/unbounded is
    conservatively kept (treated as non-redundant) rather than dropped.

    Args:
        poly: Polytope to reduce. Its equality constraints (``Ae``, ``be``),
            if any, are preserved unchanged and included in every LP.
        tol: Numerical tolerance for the "is redundant" comparison.

    Returns:
        A new Polytope with a minimal (or at least LP-verified non-redundant)
        set of inequality rows.
    """
    A, b = np.asarray(poly.A), np.asarray(poly.b)
    m, d = A.shape
    has_eq = poly.Ae.shape[0] > 0
    bounds = _unbounded_bounds(d)

    keep = []
    for i in range(m):
        mask = np.ones(m, dtype=bool)
        mask[i] = False
        kwargs = dict(c=-A[i], A_ub=A[mask], b_ub=b[mask], bounds=bounds, method="highs")
        if has_eq:
            kwargs["A_eq"], kwargs["b_eq"] = poly.Ae, poly.be
        res = linprog(**kwargs)
        if not res.success or -res.fun > b[i] + tol:
            keep.append(i)

    if has_eq:
        return Polytope(A=A[keep], b=b[keep], Ae=poly.Ae, be=poly.be)
    return Polytope(A=A[keep], b=b[keep])


def robust_minimal_hrep(poly: Polytope, tol: float = 1e-7) -> Polytope:
    """Minimal H-rep of ``poly``: tries cdd (fast) first, falls back to
    :func:`lp_minimal_hrep` (slower, but robust) only if cdd's exact/
    combinatorial method fails numerically. Prefer this over calling
    :func:`lp_minimal_hrep` unconditionally -- it keeps cdd's speed for the
    (common) well-conditioned case and only pays the LP cost when needed.
    """
    try:
        reduced = poly.copy()
        reduced.minimize_H_rep()
        return reduced
    except Exception:  # noqa: BLE001 - cdd failure of any kind -> fall back
        return lp_minimal_hrep(poly, tol=tol)


def robust_contains(outer: Polytope, inner: Polytope, tol: float = 1e-7) -> bool:
    """Containment check: tries cdd (fast) first, falls back to
    :func:`lp_contains` (slower, but robust) only if cdd's method fails
    numerically. Prefer this over calling :func:`lp_contains` unconditionally
    for the same reason as :func:`robust_minimal_hrep`.
    """
    try:
        return outer.contains(inner)
    except Exception:  # noqa: BLE001 - cdd failure of any kind -> fall back
        return lp_contains(outer, inner, tol=tol)


def lp_contains(outer: Polytope, inner: Polytope, tol: float = 1e-7) -> bool:
    """Return whether ``outer`` contains ``inner``, via one LP per facet of ``outer``.

    For every inequality row ``i`` of ``outer`` (``A1[i] @ x <= b1[i]``),
    checks that ``max A1[i] @ x`` subject to ``inner``'s constraints does not
    exceed ``b1[i]``; ``outer`` contains ``inner`` iff this holds for every
    row (an infeasible/unbounded LP for any row means containment fails).

    Args:
        outer: Candidate containing Polytope.
        inner: Candidate contained Polytope.
        tol: Numerical tolerance for the containment comparison.

    Returns:
        True iff every facet check passes.
    """
    A1, b1 = np.asarray(outer.A), np.asarray(outer.b)
    A2, b2 = np.asarray(inner.A), np.asarray(inner.b)
    d = A2.shape[1]
    has_eq = inner.Ae.shape[0] > 0
    bounds = _unbounded_bounds(d)

    for i in range(A1.shape[0]):
        kwargs = dict(c=-A1[i], A_ub=A2, b_ub=b2, bounds=bounds, method="highs")
        if has_eq:
            kwargs["A_eq"], kwargs["b_eq"] = inner.Ae, inner.be
        res = linprog(**kwargs)
        if not res.success:
            return False
        if -res.fun > b1[i] + tol:
            return False
    return True
