# Two-mass mass-spring-damper example

Two masses connected only by a spring (`k`) and a damper (`c`), no wall,
each with its own actuator: state `x = [x1, v1, x2, v2]`, input `u = [u1, u2]`,
output `y = [x1, x2]` (both masses' positions). All four physical parameters
(`m1`, `m2`, `k`, `c`) are uncertain. `MPC`, `RMPC`, `RAMPC` (this repo's
`src/` controllers) are compared in closed loop tracking the same reference
signal, against the *true* plant (a specific, fixed parameter vector inside
the identified uncertainty set), not the nominal model the controllers see.

## Pipeline

```
msd_test.ipynb / modeling-uncertainty.ipynb   (build + verify the uncertainty model)
                |
                v
model_data/*.npz                              (exported model + uncertainty data)
                |
                v
{mpc,rmpc,rampc}_example.py  ---> results/<ref>/*.csv, *_bounds.npz
                |
                v
run_comparison.py            ---> results/<ref>/comparison_*.pdf, zoom_*.pdf, rmse_table_*.tex
```

### 1. Notebooks: modeling the uncertainty

- **`msd_test.ipynb`**: the full notebook -- plant + uncertainty representation
  (SVD/PCA cardinality reduction on `(m1, m2, k, c)` sampled increments),
  reduced affine model `(A(p), B(p))`, nested convex uncertainty sets
  (ellipsoid/polytope/box), offline robust gain synthesis, closed-loop
  tracking with `MPC`/`RMPC`/`RAMPC`.
- **`model_data/modeling-uncertainty.ipynb`**: a standalone copy covering only
  the modeling/uncertainty-representation sections (1-5), used to verify the
  model export independently of the closed-loop sections.

Both notebooks end with an export section (Section 5 in
`modeling-uncertainty.ipynb`) that writes the three `model_data/*.npz` files
below. The plant is defined as its **own Euler discretization**
(`Ad = I + Ts*Ac`, `Bd = Ts*Bc`) by design, not as an approximation of a
separate continuous-time system -- see the notebook's Section 1-2 markdown
for why this choice keeps the affine parametrization exact
(`n_p_exact = 6`, matching the combined ratios `k/m1, k/m2, c/m1, c/m2, 1/m1,
1/m2`) instead of approximate.

### 2. `model_data/*.npz`: exported model + uncertainty data

- **`ss_data.npz`**: the affine state-space family `A(p) = A[:,:,0] + sum_i
  p_i * A[:,:,i]` (same for `B`), `C`, `D`, `n_p`, `n_p_exact`, `Ts`,
  `p_coords` (the identified parameter point cloud), `PNOM` (nominal
  physical `[m1, m2, k, c]`). `A[:,:,0] == A_nominal`, `B[:,:,0] ==
  B_nominal` exactly (uncentered SVD: `p = 0 <=> (A_nominal, B_nominal)`).
- **`uncertainty_sets.npz`**: `Theta` in `R^n_p` as ellipsoid (`Q`, `r`,
  `p_c`), fixed-complexity polytope (`Gp`/`hp`, unit form `H_poly`), rotated
  box (`Hb`/`hb`, unit form `H_rect`), and the exact convex hull
  (`H_hull_raw`/`h_hull_raw`, unit form `H_hull`) -- plus `W_mismatch`
  (`W_A`, `W_b`, `W_dim`), the truncation-mismatch additive disturbance on
  `x(k+1)` in **state space** (dimension `W_dim = NX`), not parameter space.
- **`sys_param.npz`**: the single test/working point used to drive the
  closed-loop plant -- `p_real` (physical `[m1, m2, k, c]`, kept consistent
  with `msd_test.ipynb`'s `P_TRUE`), `p_basis` (that same point's coordinate
  in the reduced theta space, i.e. what RAMPC's `th_hat` estimates), `p_cheb`
  (the Chebyshev centre of the ellipsoid `Theta_ell` -- for an ellipsoid,
  just its own centre), `n_p`.
- **`{mpc,rmpc,rampc}_cache.npz`**: cached offline `(K, P)` gain-synthesis
  result per controller (recomputed automatically if missing/invalid).

### 3. `common.py`: shared utilities

- **Data loading**: `load_ss`, `load_uncertainty_set(set_type="polytope"|"box"|"hull")`,
  `load_mismatch_set`, `load_point`, `load_offline_cache`/`save_offline_cache`.
- **Plant**: `make_msd_plant(m1, m2, k, c, ts)` -- returns `(rhs, step)`; the
  plant *is* its own Euler discretization (matches the notebook), so `step`
  takes a single Euler step, not RK4; `rhs` (continuous-time dynamics) is
  exposed for reference only.
- **Reference/tracking**: `make_reference(ts, kind, amp, n_signals, period,
  phase)` -- returns an `(n_signals,)`-valued `reference(k)` ("square" wave
  or a "harmonic" multi-tone), `ref_option`, `reference_arg` (flattens the
  window into the flat list `solve()`/`ca.vertcat` expects).
- **Disturbance**: `make_disturbance(W, rng)` (rejection sampling in a
  polytope).
- **RAMPC/RMPC bookkeeping**: `sys_vertex`, `lift_output_matrix`,
  `theta_vertex_bounds`.
- **Results I/O**: `save_results_csv`/`load_results_csv` (schema: `t, r1,
  r2, x1, v1, x2, v2, u1, u2, xa1..xa4`, plus any `extra` columns --
  `rampc_example.py` adds `theta_hat_i`/`theta_up_i`/`theta_low_i`),
  `save_bounds_npz`/`load_bounds_npz` (constraint levels + any `extra`
  constant data, e.g. RAMPC's `theta_true`).
- **Plotting**: `shared_legend(fig, axes, loc="outside upper center", ncol=None)`
  -- one fig-level legend deduplicated by label, for stacked `sharex`
  subplots that reuse the same color per signal role across panels (state /
  reference / bound, or one color per controller) instead of repeating an
  identical legend in every subplot.

### 4. `{mpc,rmpc,rampc}_example.py`: single-controller closed-loop runs

Each script: loads `model_data/*.npz`, builds/loads the cached `(K, P)`,
constructs the controller with physical bounds (`xBound = ±[2.22, 6, 2.22,
6]`, `uBound = ±[25, 25]`, matching the MATLAB reference `Complete_5ag.m`),
runs `T = 200` closed-loop steps against the *true* plant
(`make_msd_plant(*p_real, Ts)`), and saves `results/<TRACK_SIGNAL>/
{mpc,rmpc,rampc}_example.csv` + `..._bounds.npz`.

- `mpc_example.py`: plain (non-robust) MPC on the nominal model
  (`A_full[:,:,0]`, `B_full[:,:,0]`) -- the one most likely to violate
  constraints, since it only sees the nominal plant.
- `rmpc_example.py`: tube-based robust MPC on the full affine family,
  certified against the vertices of `Theta`.
- `rampc_example.py`: robust *adaptive* MPC -- additionally estimates
  `theta` online (`par_filter`: `"lms"` or `"rls"`) and shrinks its
  polytopic parameter set over time; also plots/saves the estimate
  (`theta_hat`) against its true value (`p_basis`) and bound
  (`theta_up`/`theta_low`).

Key config knobs at the top of each script: `TRACK_MODE`
(`"trajectory"|"setpoint"`), `TRACK_SIGNAL` (`"square"|"harmonic"`),
`UNCERTAINTY_MODEL` (`"polytope"|"box"|"hull"`, RMPC/RAMPC only), `SEED`.

Run e.g.:

```bash
python mpc_example.py
python rmpc_example.py
python rampc_example.py
```

### 5. `run_comparison.py`: cross-controller comparison

Re-plots cached `results/<ref>/*.csv` (no re-simulation) for a chosen subset
of controllers, in the same layout/font/line style as the individual
scripts.

```bash
python run_comparison.py --ref square --ctrl mpc rmpc rampc
python run_comparison.py --ref harmonic --ctrl rmpc rampc --save
python run_comparison.py --ref harmonic --ctrl mpc rmpc rampc --save --zoom --table
```

- `--ref {square,harmonic}` (required): selects `results/<ref>/`.
- `--ctrl {mpc,rmpc,rampc} [...]` (required, repeatable): controllers to
  overlay, one fixed color each (MPC blue, RMPC orange, RAMPC green) shared
  across every figure. If `rampc` is included, an extra figure shows its
  parameter estimate vs. true value and bound (rebuilt from the CSV's
  `theta_hat_i`/`theta_up_i`/`theta_low_i` columns, no re-simulation).
- `--save`: also writes each figure as a PDF into `results/<ref>/`
  (`comparison_tracking_*.pdf`, `comparison_input_*.pdf`,
  `comparison_theta_rampc.pdf`, and the zoom figure if `--zoom` is set).
- `--zoom`: an extra figure zoomed on **MPC's** longest contiguous bound
  violation (across `x1`, `x2`, `u1`, `u2` -- MPC specifically, since it is
  the controller most likely to violate constraints). Requires `mpc` among
  `--ctrl`; if MPC never violates any bound in that run, the figure is
  skipped (printed message, not an error). The violated window is shaded
  and padded with context steps on both sides; the file name encodes which
  signal violated (e.g. `zoom_x1_mpc_violation.pdf`).
- `--table`: writes a booktabs-style LaTeX table of tracking RMSE (`x1` vs
  `r1`, `x2` vs `r2`) for every selected controller to
  `results/<ref>/rmse_table_<ref>_<ctrl...>.tex` (independent of `--save`,
  which only controls figure export).

## Known loose ends

- `working_points.npz` at the top of this folder (not inside `model_data/`)
  is a stray duplicate of `model_data/sys_param.npz`'s data, from an earlier
  notebook run with a different working directory. Nothing reads it; safe
  to delete once confirmed unneeded.
- The example scripts currently run with the additive disturbance
  (`make_disturbance`) disabled (commented out in the closed-loop loop) and
  a slower reference (`period=[5.0, 6.8]`) -- both easy to re-enable, but
  worth knowing when comparing runs against earlier results.
