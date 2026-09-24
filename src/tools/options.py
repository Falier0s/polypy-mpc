import numpy as np

INF = 1e4


class Options:
    """A simple wrapper for MPC options."""

    def __init__(self, options=None):
        if options is None:
            options = {}
        self.ts = options.get("Ts") if options.get("Ts") not in (None, 0) else 0
        self.solver = options.get("solver") or "qpoases"
        self.verbose = options.get("verbose") if options.get("verbose") != "" else 0
        self.relax = options.get("relax") if options.get("relax") != "" else False
        # Weight of the L1 penalty on the slack variables that relax the
        # state/input/output box constraints (see MPC.add_hard_constraints)
        # when relax=True. Large enough to make violating the box costly
        # relative to the tracking cost, but finite so the relaxed QP stays
        # bounded (and hence always solvable) even when the original hard
        # problem would be infeasible.
        self.relax_weight = (
            options.get("relax_weight") if options.get("relax_weight") is not None else 1e3
        )
        self.customJ = options.get("customJ") if options.get("customJ") != "" else False
        self.ref = options.get("ref") if options.get("ref") is not None else None
        self.name = options.get("name") or f"CMPC_{np.random.randint(1000)}"
        self.svd = options.get("svd") if options.get("svd") is not None else False
        self.Nc = options.get("Nc") if options.get("Nc") not in (None, 0) else None
        self.sigma = options.get("sigma") if options.get("sigma") not in (None, 0) else 0.95
        self.K = options.get("K") if options.get("K") is not None else None
        self.P = options.get("P") if options.get("P") is not None else None
        self.xBound = options.get("xBound") if options.get("xBound") is not None else None
        self.uBound = options.get("uBound") if options.get("uBound") is not None else None
        self.yBound = options.get("yBound") if options.get("yBound") is not None else None
        self.W = options.get("W") if options.get("W") is not None else None
        self.E = options.get("E") if options.get("E") is not None else None
        self.lam = options.get("lam") if options.get("lam") is not None else 1.0
        # Contraction factor for the tracking terminal set's OWN invariant-set
        # iteration (InvariantSet.compute_invariant_set), independent of self.lam
        # (used elsewhere for GainSynthesis). Defaults to 1.0 -- today's existing
        # behaviour, unchanged -- since whether lam<1 helps or hurts convergence/
        # conditioning there is genuinely system-dependent (confirmed empirically:
        # helps a larger, near-marginally-stable augmented system, breaks a small
        # one that already converges fine at lam_term=1).
        self.lam_term = options.get("lam_term") if options.get("lam_term") is not None else 1.0
        # RMPC.initialize(): try InvariantSet.compute_invariant_set(cmplx_red=True)
        # first (fixed facet count, one-shot LP, no combinatorial growth) and only
        # fall back to the full growing method (cmplx_red=False) if that fixed
        # shape genuinely cannot support the requested contraction (RuntimeError).
        # Default False: this changes the closed-loop result (not just speed) --
        # confirmed on examples/turtlebot4/rampc_example.py, where turning it on
        # measurably diverges from the original trajectory (grows over time, not
        # just numerical noise) even though that script never asked for it. Existing
        # callers must keep getting the original behaviour unless they explicitly
        # opt in with cmplx_red=True.
        self.cmplx_red = options.get("cmplx_red") if options.get("cmplx_red") is not None else False
        # Optional fixed facet normals passed through to compute_invariant_set's
        # `shape` argument when cmplx_red=True. None -> that method's own default
        # (the terminal set's own boundary-polytope normals).
        self.cmplx_shape = options.get("cmplx_shape") if options.get("cmplx_shape") is not None else None
        self.theta = options.get("theta") if options.get("theta") is not None else None
        self.theta_c = options.get("theta_c") if options.get("theta_c") is not None else None
        self.lpv_flag = options.get("lpv_flag") if options.get("lpv_flag") is not None else False # Default is false
        self.par_filter = options.get("par_filter") if options.get("par_filter") is not None else 'lms' # Default is lms, other options could be 'rls', 'kalman', 'chebyshev'
        self.max_delta_th = options.get("max_delta_th") if options.get("max_delta_th") is not None else 0.0
        # Internal representation of the parameter sets in the SetUpdater (RAMPC):
        # "reduced" -> vertices from fixed facet bases (cheap, exact only while the set keeps the
        #              combinatorial structure of the initial one); "full" -> exact vertices at every
        #              step, padded to the initial number of vertices (heavier, always consistent).
        # Default "full": "reduced" is only exact under an assumption existing
        # callers never agreed to (unchanged combinatorial structure) -- opt in
        # explicitly with set_representation="reduced" once that assumption is
        # verified for a given setup.
        self.set_representation = options.get("set_representation") if options.get("set_representation") is not None else "full"
        # Constant exponential forgetting factor lambda in (0, 1] for the (projected GF-)RLS estimator.
        self.forgetting = options.get("forgetting") if options.get("forgetting") is not None else 0.98
        # Initial point estimate of the parameter vectors, WITHOUT the leading homogeneous 1
        # (added internally). None -> Chebyshev centre of the parameter set (legacy behaviour).
        self.th_0 = options.get("th_0")
        self.th_c_0 = options.get("th_c_0")
        self.th_hat_0 = options.get("th_hat_0")
        self.th_c_hat_0 = options.get("th_c_hat_0")

    def get_bound(self, n, m, p):
        """Generate bound of state, input and output

        Args:
            n (int): number of states
            m (int): number of inputs
            p (int): number of outputs

        Returns:
            tuple: the boundaries
        """

        if self.xBound is not None and self.xBound != 0:
            lxb = np.copy(self.xBound[0])
            uxb = np.copy(self.xBound[1])
        else:
            lxb = np.full((n, 1), -INF)
            uxb = np.full((n, 1), INF)
        if self.uBound is not None and self.uBound != 0:
            lub = np.copy(self.uBound[0])
            uub = np.copy(self.uBound[1])
        else:
            lub = np.full((m, 1), -INF)
            uub = np.full((m, 1), INF)
        if self.yBound is not None and self.yBound != 0:
            lyb = np.copy(self.yBound[0])
            uyb = np.copy(self.yBound[1])
        else:
            lyb = np.full((p, 1), -INF)
            uyb = np.full((p, 1), INF)
        return lxb, uxb, lub, uub, lyb, uyb

    def set_track(self, sym, n, m, p, N):
        """Set track based on reference or not

        Args:
            sym (Symbolic): the symbolic class
            n (int): nuòber of states
            m (int): number of inputs
            p (int): number of outputs
            N (int): horizons

        Returns:
            bool: True if there is some tracking, False if not
        """
        if self.ref is not None:
            if str(self.ref).lower() in ("traj", "trajectory"):
                sym.create_tracking_variables(n, m, p, N)
            else:
                sym.create_tracking_variables(n, m, p, N, traj=False)
            return True
        sym.create_tracking_variables(n, m, p, N, symbolic=False, traj=False)
        return False
