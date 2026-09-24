"""
Closed-loop comparison across controllers on the two-mass mass-spring-damper plant
====================================================================================
Re-plots cached results from results/<ref>/{mpc,rmpc,rampc}_example.csv (each
produced by the matching *_example.py script) on the SAME axes, one color per
controller, instead of re-simulating. If "rampc" is among the selected
controllers, an extra figure shows its online parameter estimate (theta_hat)
against its true value and its shrinking polytopic bound over time.

Usage
-----
    python run_comparison.py --ref square --ctrl mpc rmpc rampc
    python run_comparison.py --ref harmonic --ctrl rmpc rampc --save
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from os import path
from pathlib import Path

import common as dc

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
# every figure (tracking, input, zoom) -- color alone is not enough to tell
# lines apart for color-blind readers, and since these are near-overlapping
# trajectories, distinguishing by marker alone (all solid) is not enough
# either, so each gets its own dash pattern; MPC keeps a plain solid line
# with no marker (solid is already visually distinct on its own).
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MPC/RMPC/RAMPC closed-loop results on the two-mass mass-spring-damper example."
    )
    parser.add_argument("--ref", required=True, choices=["square", "harmonic"],
                         help="Reference signal of the runs to compare (selects results/<ref>/).")
    parser.add_argument("--ctrl", required=True, nargs="+", choices=list(CTRL_COLOR),
                         help="Controllers to compare, e.g. --ctrl mpc rmpc rampc.")
    parser.add_argument("--save", action="store_true",
                         help="Also save each figure as PDF in results/<ref>/.")
    parser.add_argument("--zoom", action="store_true",
                         help="Extra zoomed-in figure on MPC's longest contiguous constraint "
                              "violation (any of x1, x2, u1, u2), if one exists.")
    parser.add_argument("--table", action="store_true",
                         help="Export a LaTeX table of tracking RMSE (x1, x2) per selected controller.")
    args = parser.parse_args()
    args.ctrl = list(dict.fromkeys(args.ctrl))  # de-duplicate, keep order
    return args


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


def find_mpc_violation(runs, x_lb, x_ub, u_lb, u_ub):
    """Return (signal, start, end, length) for MPC's longest contiguous
    bound violation across x1, x2, u1, u2, or None if "mpc" was not among
    the compared controllers or it never violates any bound.
    """
    if "mpc" not in runs:
        print('--zoom requires "mpc" among --ctrl (it is the one most likely to violate '
              'constraints); skipping the zoom figure.')
        return None
    data, _ = runs["mpc"]
    checks = (
        ("x1", data["x1"], x_lb[0], x_ub[0]),
        ("x2", data["x2"], x_lb[2], x_ub[2]),
        ("u1", data["u1"], u_lb[0], u_ub[0]),
        ("u2", data["u2"], u_lb[1], u_ub[1]),
    )
    best = None
    for name, values, lb, ub in checks:
        run = longest_violation_run(values, lb, ub)
        if run is not None and (best is None or run[2] > best[3]):
            best = (name, run[0], run[1], run[2])
    if best is None:
        print("--zoom: MPC never violates any bound (x1, x2, u1, u2) in this run; skipping the zoom figure.")
    return best


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def write_rmse_table(path_tex: Path, runs, ctrl_list, ref: str):
    """Write a booktabs-style LaTeX table of tracking RMSE (x1 vs r1, x2 vs
    r2) for each controller in ctrl_list."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Tracking RMSE, {ref} reference}}",
        rf"\label{{tab:rmse_{ref}_{'_'.join(ctrl_list)}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Controller & RMSE $x_1$ & RMSE $x_2$ \\",
        r"\midrule",
    ]
    for ctrl in ctrl_list:
        data, _ = runs[ctrl]
        rmse_x1 = rmse(data["x1"], data["r1"])
        rmse_x2 = rmse(data["x2"], data["r2"])
        lines.append(rf"{CTRL_LABEL[ctrl]} & {rmse_x1:.4f} & {rmse_x2:.4f} \\")
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

    # --- Figure 1: output tracking (y = [x1, x2] vs. reference) -----------
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    y_labels = (r"$x_1$", r"$x_2$")
    y_state_idx = (0, 2)  # x1, x2 live at state indices 0, 2 (x = [x1, v1, x2, v2])
    for i, (col, r_col) in enumerate((("x1", "r1"), ("x2", "r2"))):
        ax[i].plot(k, ref_data[r_col], color=REF_COLOR, label=r'$r_k$', lw=LINE_WIDTH, **REF_STYLE)
        ax[i].hlines([x_lb[y_state_idx[i]], x_ub[y_state_idx[i]]], k[0], k[-1],
                     color=BOUND_COLOR, label=r'$x$ bound', lw=LINE_WIDTH, **BOUND_STYLE)
        for ctrl in args.ctrl:
            data, _ = runs[ctrl]
            ax[i].plot(np.arange(len(data["t"])), data[col], color=CTRL_COLOR[ctrl],
                       label=CTRL_LABEL[ctrl], lw=LINE_WIDTH, **CTRL_STYLE[ctrl])
        ax[i].set_ylabel(y_labels[i])

    dc.shared_legend(fig, ax)
    ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
    ax[-1].set_xlim(k[0], k[-1])
    if args.save:
        fig.savefig(RESULTS_DIR / f"comparison_tracking_{'_'.join(args.ctrl)}.pdf")
    plt.show()

    # --- Figure 2: input effort (u1, u2) -----------------------------------
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    for i, col in enumerate(("u1", "u2")):
        ax[i].hlines([u_lb[i], u_ub[i]], k[0], k[-1], color=BOUND_COLOR,
                     label=r'$u$ bound', lw=LINE_WIDTH, **BOUND_STYLE)
        for ctrl in args.ctrl:
            data, _ = runs[ctrl]
            ax[i].plot(np.arange(len(data["t"])), data[col], color=CTRL_COLOR[ctrl],
                       label=CTRL_LABEL[ctrl], lw=LINE_WIDTH, **CTRL_STYLE[ctrl])
        ax[i].set_ylabel(f'$u_{i + 1}$')

    dc.shared_legend(fig, ax)
    ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
    ax[-1].set_xlim(k[0], k[-1])
    if args.save:
        fig.savefig(RESULTS_DIR / f"comparison_input_{'_'.join(args.ctrl)}.pdf")
    plt.show()

    # --- Figure 3 (only if "rampc" was selected): parameter estimation ----
    if "rampc" in args.ctrl:
        rampc_data, rampc_bounds = runs["rampc"]
        theta_true = rampc_bounds["theta_true"]
        n_p = len(theta_true)
        k_r = np.arange(len(rampc_data["t"]))
        needed = [f"theta_up_{i + 1}" for i in range(n_p)] + [f"theta_low_{i + 1}" for i in range(n_p)]
        missing = [name for name in needed if name not in rampc_data]

        if missing:
            print(f"Skipping the theta figure: results/{args.ref}/rampc_example.csv is missing {missing} "
                  "-- re-run rampc_example.py to regenerate it with the current column schema.")
        else:
            fig, ax = plt.subplots(n_p, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
            ax = np.atleast_1d(ax)
            for i in range(n_p):
                ax[i].plot(k_r, rampc_data[f"theta_hat_{i + 1}"], color="#00FF00EA",
                           label=r'$\hat{\theta}_k$', marker='o', markersize=MARKER_SIZE, lw=LINE_WIDTH)
                ax[i].set_ylabel(r"$\hat{\theta}_{%d}$" % (i + 1))
                ax[i].fill_between(k_r, rampc_data[f"theta_low_{i + 1}"], rampc_data[f"theta_up_{i + 1}"],
                                    color="#006EFFEA", alpha=0.2, label=r'$\theta$ bounds')
                ax[i].plot(k_r, rampc_data[f"theta_up_{i + 1}"], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
                ax[i].plot(k_r, rampc_data[f"theta_low_{i + 1}"], color="#006EFFEA", ls='--', lw=LINE_WIDTH)
                ax[i].hlines(theta_true[i], k_r[0], k_r[-1], color="#FF0000EA", ls='--',
                             label=r'$\theta$ true value', lw=LINE_WIDTH)

            dc.shared_legend(fig, ax)
            ax[-1].set_xlabel(r"$\mathrm{Step}\ k$")
            if args.save:
                fig.savefig(RESULTS_DIR / "comparison_theta_rampc.pdf")
            plt.show()

    # --- Optional: zoom on MPC's longest contiguous constraint violation --
    if args.zoom:
        violation = find_mpc_violation(runs, x_lb, x_ub, u_lb, u_ub)
        if violation is not None:
            signal, i0, i1, length = violation
            is_state = signal in ("x1", "x2")
            state_idx = {"x1": 0, "x2": 2}.get(signal)
            input_idx = {"u1": 0, "u2": 1}.get(signal)
            lb, ub = (x_lb[state_idx], x_ub[state_idx]) if is_state else (u_lb[input_idx], u_ub[input_idx])

            # Small, fixed time-axis tolerance around the violation itself
            # (not scaled with its length -- a long violation should not
            # zoom back out).
            TIME_TOL = 5
            lo, hi = max(0, i0 - TIME_TOL), min(len(k) - 1, i1 + TIME_TOL)

            name = f"zoom_{signal}_mpc_violation"
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, layout="constrained")

            visible_values = []
            if is_state:
                r_col = ref_data[f"r{signal[-1]}"]
                ax.plot(k, r_col, color=REF_COLOR, label=r'$r_k$', lw=LINE_WIDTH, **REF_STYLE)
                visible_values.append(r_col[lo:hi + 1])
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
            ax.set_ylabel(f'${signal[0]}_{signal[-1]}$')
            ax.set_xlabel(r"$\mathrm{Step}\ k$")
            # ax.set_title(f"MPC {signal} bound violation: steps {i0}-{i1} ({length} steps)")
            dc.shared_legend(fig, ax, ncol=3)  # up to 5 entries here -- 1 row overflows the figure width
            if args.save:
                fig.savefig(RESULTS_DIR / f"{name}.pdf")
                print("Saved zoom figure to:", RESULTS_DIR / f"{name}.pdf")
            plt.show()

    # --- Optional: LaTeX RMSE table ---------------------------------------
    if args.table:
        write_rmse_table(RESULTS_DIR / f"rmse_table_{args.ref}_{'_'.join(args.ctrl)}.tex",
                          runs, args.ctrl, args.ref)

    if args.save:
        print("Saved comparison figures to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
