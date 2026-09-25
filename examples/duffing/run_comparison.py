"""
Closed-loop comparison across controllers on the Duffing oscillator
====================================================================
Re-plots cached results from results/<ref>/{mpc,rmpc,rampc}_example.csv (each
produced by the matching *_example.py script) on the SAME axes, one color per
controller, instead of re-simulating. Figures:

1. output tracking (y = x1 vs. r) and input effort (u), stacked;
2. phase plane (x1, x2) with the state constraint box;
3. if "rampc" is among the selected controllers, its online parameter
   estimate delta_hat against its true value and the projection of its
   shrinking parameter set over time (needs the theta_hat_i/theta_up_i/theta_low_i
   columns in rampc_example.csv and theta_true in rampc_example_bounds.npz).

Usage
-----
    python run_comparison.py --ref square --ctrl mpc rmpc rampc
    python run_comparison.py --ref harmonic --ctrl rmpc rampc --save
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from os import path
from pathlib import Path

import duffing_common as dc

CURRENT_DIR = path.dirname(path.abspath(__file__))

# Same layout convention as the individual *_example.py scripts.
cep_width_inches = 3.54 * 3
aspect_ratio = 0.618
fig_height = cep_width_inches * aspect_ratio
FIGSIZE = (cep_width_inches, fig_height)

LINE_WIDTH = 2.8
MARKER_SIZE = 6
FONT_SIZE = {
    "labels": 24,
    "ticks": 20,
    "legend": 22,
}
BOUND_COLOR = "#FF0000B0"
REF_COLOR = "#383838B0"
REF_STYLE = dict(ls=':')            # dotted: distinct from bound (dashed) and every controller below
BOUND_STYLE = dict(ls='--')         # dashed, flat hline -- shape alone already sets it apart

# One consistent color AND line style/marker combo per controller, reused in
# every figure (tracking/input, phase plane, zoom) -- color alone is not
# enough to tell lines apart for color-blind readers, and since these are
# near-overlapping trajectories, distinguishing by marker alone (all solid)
# is not enough either, so each gets its own dash pattern; MPC keeps a plain
# solid line with no marker (solid is already visually distinct on its own).
CTRL_COLOR = {
    "mpc": "#006EFFEA",
    "rmpc": "#E28000D2",
    "rampc": "#00AD17D2",
}
CTRL_STYLE = {
    "mpc": dict(ls='-'),
    "rmpc": dict(ls='-.', marker='s', markevery=8, markersize=MARKER_SIZE),
    "rampc": dict(ls=(0, (5, 1)), marker='^', markevery=8, markersize=MARKER_SIZE),
}
CTRL_LABEL = {"mpc": "MPC", "rmpc": "RMPC", "rampc": "RAMPC"}

# Signals checked by --zoom: CSV column -> (bound key, index in that bound, y-label)
ZOOM_SIGNALS = {
    "x1": ("xBound", 0, r"$y = x_1$"),
    "x2": ("xBound", 1, r"$x_2$"),
    "u": ("uBound", 0, r"$u$"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MPC/RMPC/RAMPC closed-loop results on the Duffing oscillator example."
    )
    parser.add_argument("--ref", required=True, choices=["square", "harmonic"],
                         help="Reference signal of the runs to compare (selects results/<ref>/).")
    parser.add_argument("--ctrl", required=True, nargs="+", choices=list(CTRL_COLOR),
                         help="Controllers to compare, e.g. --ctrl mpc rmpc rampc.")
    parser.add_argument("--save", action="store_true",
                         help="Also save each figure as PDF in results/<ref>/.")
    parser.add_argument("--zoom", action="store_true",
                         help="Extra zoomed-in figure on MPC's longest contiguous constraint "
                              "violation (any of x1, x2, u), if one exists.")
    parser.add_argument("--table", action="store_true",
                         help="Export a LaTeX table of tracking RMSE (y = x1 vs r) per selected controller.")
    args = parser.parse_args()
    args.ctrl = list(dict.fromkeys(args.ctrl))  # de-duplicate, keep order
    return args


def shared_legend(fig, axes, loc="outside upper center", ncol=None, **kwargs):
    """One fig-level legend replacing a repeated per-subplot one, keeping
    only the FIRST handle for each distinct label across ``axes``.
    ``loc="outside upper center"`` needs a "constrained" layout figure.
    (Same helper as mass-spring-damper/common.py -- duffing_common.py does
    not carry it.)
    """
    seen = {}
    for ax in np.atleast_1d(axes):
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            seen.setdefault(l, h)
    if ncol is None:
        ncol = len(seen)
    fig.legend(list(seen.values()), list(seen.keys()), loc=loc, ncol=ncol, frameon=False, **kwargs)


def longest_violation_run(values: np.ndarray, lb: float, ub: float):
    """Return (start, end, length) of the longest contiguous run of samples
    with values[k] < lb or values[k] > ub, or None if there is no violation.
    ``end`` is inclusive.
    """
    mask = (values < lb) | (values > ub)
    best = None
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if best is None or (j - i) > best[2]:
                best = (i, j - 1, j - i)
            i = j
        else:
            i += 1
    return best


def signal_bounds(bounds, signal):
    """(lb, ub) of CSV column ``signal`` (one of ZOOM_SIGNALS)."""
    key, idx, _ = ZOOM_SIGNALS[signal]
    lb, ub = bounds[key]
    return float(lb[idx]), float(ub[idx])


def find_mpc_violation(runs):
    """Return (signal, start, end, length) for MPC's longest contiguous
    bound violation across x1, x2, u, or None if "mpc" was not among the
    compared controllers or it never violates any bound.
    """
    if "mpc" not in runs:
        print('--zoom requires "mpc" among --ctrl (it is the one most likely to violate '
              'constraints); skipping the zoom figure.')
        return None
    data, bounds = runs["mpc"]
    best = None
    for name in ZOOM_SIGNALS:
        run = longest_violation_run(data[name], *signal_bounds(bounds, name))
        if run is not None and (best is None or run[2] > best[3]):
            best = (name, run[0], run[1], run[2])
    if best is None:
        print("--zoom: MPC never violates any bound (x1, x2, u) in this run; skipping the zoom figure.")
    return best


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def write_rmse_table(path_tex: Path, runs, ctrl_list, ref: str):
    """Write a booktabs-style LaTeX table of tracking RMSE (y = x1 vs r)
    for each controller in ctrl_list."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Tracking RMSE, {ref} reference}}",
        rf"\label{{tab:rmse_duffing_{ref}_{'_'.join(ctrl_list)}}}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Controller & RMSE $y$ \\",
        r"\midrule",
    ]
    for ctrl in ctrl_list:
        data, _ = runs[ctrl]
        lines.append(rf"{CTRL_LABEL[ctrl]} & {rmse(data['x1'], data['r']):.4f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path_tex.parent.mkdir(parents=True, exist_ok=True)
    path_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved RMSE table to:", path_tex)


def load_run(results_dir: Path, ctrl: str):
    csv_path = results_dir / f"{ctrl}_example.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found -- run {ctrl}_example.py first (it writes this results/<ref>/ file)."
        )
    data = dc.load_results_csv(csv_path)
    bounds = dc.load_bounds_npz(csv_path)
    return data, bounds


def main():
    args = parse_args()
    RESULTS_DIR = Path(CURRENT_DIR) / "results" / args.ref
    runs = {ctrl: load_run(RESULTS_DIR, ctrl) for ctrl in args.ctrl}
    tag = '_'.join(args.ctrl)

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': FONT_SIZE['labels'],
        'axes.titlesize': FONT_SIZE['labels'],
        'axes.labelsize': FONT_SIZE['labels'],
        'xtick.labelsize': FONT_SIZE['ticks'],
        'ytick.labelsize': FONT_SIZE['ticks'],
        'legend.fontsize': FONT_SIZE['legend'],
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

    # Reference/bounds are shared across controllers (same physical system,
    # same reference generator) -- drawn once, from the first loaded run.
    ref_data, ref_bounds = next(iter(runs.values()))
    k = np.arange(len(ref_data["t"]))
    x_lb, x_ub = ref_bounds["xBound"]
    u_lb, u_ub = ref_bounds["uBound"]

    # --- Figure 1: output tracking (y = x1 vs. r) + input effort (u) -------
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    ax[0].plot(k, ref_data["r"], color=REF_COLOR, label=r'$r_k$', lw=LINE_WIDTH, **REF_STYLE)
    ax[0].hlines([x_lb[0], x_ub[0]], k[0], k[-1], color=BOUND_COLOR,
                 label='bound', lw=LINE_WIDTH, **BOUND_STYLE)
    ax[1].hlines([u_lb[0], u_ub[0]], k[0], k[-1], color=BOUND_COLOR,
                 label='bound', lw=LINE_WIDTH, **BOUND_STYLE)
    for ctrl in args.ctrl:
        data, _ = runs[ctrl]
        k_c = np.arange(len(data["t"]))
        for i, col in enumerate(("x1", "u")):
            ax[i].plot(k_c, data[col], color=CTRL_COLOR[ctrl], label=CTRL_LABEL[ctrl],
                       lw=LINE_WIDTH, **CTRL_STYLE[ctrl])
    ax[0].set_ylabel(r"$y$")
    ax[1].set_ylabel(r"$u$")

    shared_legend(fig, ax)
    ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
    ax[-1].set_xlim(k[0], k[-1])
    if args.save:
        fig.savefig(RESULTS_DIR / f"comparison_tracking_{tag}.pdf")
    plt.show()

    # --- Figure 2: phase plane (x1, x2) with the state constraint box ------
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")
    ax.add_patch(Rectangle((x_lb[0], x_lb[1]), x_ub[0] - x_lb[0], x_ub[1] - x_lb[1],
                           fill=False, edgecolor=BOUND_COLOR, lw=LINE_WIDTH,
                           label=r'$\mathcal{X}$', **BOUND_STYLE))
    for ctrl in args.ctrl:
        data, _ = runs[ctrl]
        ax.plot(data["x1"], data["x2"], color=CTRL_COLOR[ctrl], label=CTRL_LABEL[ctrl],
                lw=LINE_WIDTH, **CTRL_STYLE[ctrl])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    shared_legend(fig, ax)
    if args.save:
        fig.savefig(RESULTS_DIR / f"comparison_phase_{tag}.pdf")
    plt.show()

    # --- Figure 3 (only if "rampc" was selected): parameter estimation ----
    # One subplot per parameter delta_i: estimate delta_hat_i, projection of the
    # (shrinking) parameter set on the delta_i axis, true value delta_i. Data keep the
    # theta_* naming of the CSV/npz schema; only the plot labels use delta.
    if "rampc" in args.ctrl:
        rampc_data, rampc_bounds = runs["rampc"]
        p_true = rampc_bounds.get("theta_true")
        n_p = sum(1 for name in rampc_data if name.startswith("theta_hat_"))
        needed = [f"theta_{kind}_{i + 1}" for kind in ("up", "low") for i in range(n_p)]
        missing = [name for name in needed if name not in rampc_data]

        if n_p == 0 or missing or p_true is None:
            print(f"Skipping the parameter figure: results/{args.ref}/rampc_example.csv has no "
                  "theta_hat_i/theta_up_i/theta_low_i columns (or the _bounds.npz has no theta_true) "
                  "-- re-run rampc_example.py to regenerate it.")
        else:
            k_r = np.arange(len(rampc_data["t"]))
            fig, ax = plt.subplots(n_p, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
            ax = np.atleast_1d(ax)
            for i in range(n_p):
                ax[i].plot(k_r, rampc_data[f"theta_hat_{i + 1}"], color="#00FF00EA",
                           label=r'$\hat{\delta}_k$', marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)
                ax[i].set_ylabel(r"$\delta_{%d}$" % (i + 1))
                ax[i].fill_between(k_r, rampc_data[f"theta_low_{i + 1}"], rampc_data[f"theta_up_{i + 1}"],
                                    color="#006EFFEA", alpha=0.2, label=r'$\delta$ bounds')
                ax[i].plot(k_r, rampc_data[f"theta_up_{i + 1}"], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
                ax[i].plot(k_r, rampc_data[f"theta_low_{i + 1}"], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
                ax[i].hlines(p_true[i], k_r[0], k_r[-1], color="#FF0000EA", ls='--',
                             label=r'$\delta$ true value', lw=LINE_WIDTH)

            shared_legend(fig, ax)
            ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
            ax[-1].set_xlim(k_r[0], k_r[-1])
            if args.save:
                fig.savefig(RESULTS_DIR / "comparison_theta_rampc.pdf")
            plt.show()

    # --- Optional: zoom on MPC's longest contiguous constraint violation --
    if args.zoom:
        violation = find_mpc_violation(runs)
        if violation is not None:
            signal, i0, i1, length = violation
            lb, ub = signal_bounds(ref_bounds, signal)

            # Small, fixed time-axis tolerance around the violation itself
            # (not scaled with its length -- a long violation should not
            # zoom back out).
            TIME_TOL = 5
            lo, hi = max(0, i0 - TIME_TOL), min(len(k) - 1, i1 + TIME_TOL)

            name = f"zoom_{signal}_mpc_violation_{tag}"
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

            visible_values = []
            if signal == "x1":  # only the output has a reference
                ax.plot(k, ref_data["r"], color=REF_COLOR, label=r'$r_k$', lw=LINE_WIDTH, **REF_STYLE)
                visible_values.append(ref_data["r"][lo:hi + 1])
            ax.hlines([lb, ub], k[0], k[-1], color=BOUND_COLOR, label='bound', lw=LINE_WIDTH, **BOUND_STYLE)
            for ctrl in args.ctrl:
                data, _ = runs[ctrl]
                # markevery=8 (the default) can miss the zoomed-in window entirely
                # (only ~2*TIME_TOL+1 points wide) -- every point instead, here.
                style = {**CTRL_STYLE[ctrl], "markevery": 1} if "markevery" in CTRL_STYLE[ctrl] else CTRL_STYLE[ctrl]
                ax.plot(np.arange(len(data["t"])), data[signal], color=CTRL_COLOR[ctrl],
                         label=CTRL_LABEL[ctrl], lw=LINE_WIDTH, **style)
                visible_values.append(np.asarray(data[signal])[lo:hi + 1])

            # y-range from what is actually visible in [lo, hi] across every
            # plotted signal (reference + all controllers), not the whole
            # run -- the bound lines are allowed to extend beyond this.
            visible_values = np.concatenate(visible_values)
            y_lo, y_hi = visible_values.min(), visible_values.max()
            y_margin = 0.1 * max(y_hi - y_lo, 1e-6)
            ax.set_ylim(y_lo - y_margin, y_hi + y_margin)

            ax.set_xlim(lo, hi)
            ax.set_ylabel(ZOOM_SIGNALS[signal][2])
            ax.set_xlabel(r"$\mathrm{Step}\ k$")
            shared_legend(fig, ax, ncol=3)  # up to 5 entries here -- 1 row overflows the figure width
            if args.save:
                fig.savefig(RESULTS_DIR / f"{name}.pdf")
                print("Saved zoom figure to:", RESULTS_DIR / f"{name}.pdf")
            plt.show()

    # --- Optional: LaTeX RMSE table ---------------------------------------
    if args.table:
        write_rmse_table(RESULTS_DIR / f"rmse_table_{args.ref}_{tag}.tex",
                          runs, args.ctrl, args.ref)

    if args.save:
        print("Saved comparison figures to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
