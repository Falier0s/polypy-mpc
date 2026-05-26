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
        self.customJ = options.get("customJ") if options.get("customJ") != "" else False
        self.ref = options.get("ref") if options.get("ref") is not None else None
        self.name = options.get("name") or f"CMPC_{np.random.randint(1000)}"
        self.svd = options.get("svd") if options.get("svd") is not None else False
        self.Nc = options.get("Nc") if options.get("Nc") not in (None, 0) else None
        self.sigma = options.get("sigma") if options.get("sigma") not in (None, 0) else 0.95
        self.K = options.get("K") if options.get("K") is not None else None
        self.xBound = options.get("xBound") if options.get("xBound") is not None else None
        self.uBound = options.get("uBound") if options.get("uBound") is not None else None
        self.yBound = options.get("yBound") if options.get("yBound") is not None else None
        self.W = options.get("W") if options.get("W") is not None else None
        self.lam = options.get("lam") if options.get("lam") is not None else 1.0
        self.theta = options.get("theta") if options.get("theta") is not None else None
        self.theta_c = options.get("theta_c") if options.get("theta_c") is not None else None
        
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
            if self.ref.lower() in ("traj", "trajectory"):
                sym.create_tracking_variables(n, m, p, N)
            else:
                sym.create_tracking_variables(n, m, p, N, traj=False)
            return True
        sym.create_tracking_variables(n, m, p, N, symbolic=False, traj=False)
        return False
