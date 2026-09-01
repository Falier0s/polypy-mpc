# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-01

First public release. Companion software of the paper *"Design and validation of
Robust Adaptive MPC for LPV Trajectory tracking of Wheeled Differential Drive
Robot with guarantees"* (under review, Control Engineering Practice).

### Added

- `MPC` — nominal / contractive Model Predictive Control.
- `RMPC` — robust tube MPC for systems with bounded additive disturbances.
- `RAMPC` — robust adaptive MPC with online set-membership parameter-set update
  and projected RLS estimation for parametric / LPV uncertainty.
- Artificial-reference tracking extended from a steady-state target to a full
  time-varying reference trajectory over the prediction horizon.
- Polytopic representation of constraint, disturbance and parameter-uncertainty
  sets; polytope-based constraint tightening and tube construction.
- Offline design helpers: pre-stabilizing gain / terminal cost synthesis
  (`GainSynthesis`) and robust invariant / contractive set computation
  (`InvariantSet`).
- Runnable examples in `examples/`, including the TurtleBot 4 case study in
  `examples/turtlebot4/`.
- Sphinx documentation site.

### Changed

- Project renamed to **PolyPY-MPC**; the import package is now `polypympc`
  (was `rampc`) and the repository is `polypy-mpc`.
- Public API renamed to PEP 8 conventions: modules `cMPC`/`cRMPC`/`cRAMPC` →
  `mpc`/`rmpc`/`rampc`, `polytopesystem`/`statesystem` →
  `polytope_system`/`state_system`; classes `CMPC`/`CRMPC`/`CRAMPC` →
  `MPC`/`RMPC`/`RAMPC`. No backward-compatible aliases.
- Packaging metadata (`pyproject.toml`) modernized; project version is
  single-sourced from `polypympc.__version__`.
- License set to MIT (replacing an incorrect placeholder license file).

### Fixed

- `MPC` cost and system building collapsed single-input / single-output nominal
  system matrices to 1-D via `numpy.squeeze()`, breaking the CasADi expressions;
  the nominal slice is now selected explicitly.
- `SetUpdater.update` now degrades gracefully when the CVXPY parameter-set solve
  raises a `SolverError`, instead of aborting the closed-loop run.

[Unreleased]: https://github.com/Falier0s/polypy-mpc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Falier0s/polypy-mpc/releases/tag/v0.1.0
