# TODO — pycvxset-free (0.2.0)

Living checklist for replacing `pycvxset` with an internal, `pycddlib`-backed
set layer. Companion to [`pycvxset-removal.md`](pycvxset-removal.md).

Decisions taken:
- **cddlib / pycddlib as the core backend** (double-description in C: H↔V,
  vertex/facet enumeration, redundancy removal).
- **License — option 1**: accept that the combined work is **GPL-2.0-or-later**
  (pycddlib is GPL-2.0); relicense the project accordingly.
- **Module split**: pure geometry (`sets/`) vs set-theoretic control algorithms
  (`invariance/`).

---

## 0. Design freeze

- [ ] Write **§9 "Design direction"** in `pycvxset-removal.md` (backend, module
      split, Blanchini–Miani operation map, license checklist).
- [ ] Record which extra operation the earlier prototype added that pycvxset
      lacked (candidate: `pre_A` / one-step set / Pontryagin `⊖`) and slot it in.
- [ ] Fix the public API surface for `Polytope` (freeze method names before
      migrating call sites).
- [ ] Decide number type policy: float DD + tolerance by default; optional exact
      (`Fraction`) path — or drop exact for now.
- [ ] Pin `pycddlib>=3,<4` and target its 3.x API (big break vs 2.x).

## 1. Geometry layer — `src/sets/`

### `polytope.py` — `Polytope`
- [ ] `__init__(A=None, b=None, V=None)` — store H- and/or V-rep, lazy convert.
- [ ] Properties: `A`, `b`, `V`, `dim`, `is_empty`, `is_bounded`.
- [ ] `contains(x, tol)` — point membership.
- [ ] `contains(other: Polytope)` — set inclusion (`P ⊆ Q`).
- [ ] `intersection(other)` — stack halfspaces (+ optional redundancy removal).
- [ ] `minimize()` / `minimize_h_rep()` — drop redundant inequalities (cdd).
- [ ] `normalize()` — row-normalize `A` by `‖Aᵢ‖`; `with_unit_b()` — the
      recurring `A / b[:,None]` rescale.
- [ ] `projection(axes)` — coordinate projection (cdd block elimination / FM).
- [ ] `chebyshev_centering()` — returns `(centre, radius)` (LP). Keep the tuple.
- [ ] `volume()` — exact via V-rep + `ConvexHull.volume` (optional / reporting).
- [ ] `copy()`.
- [ ] `plot(ax=None, **patch_args)` — 2D only, guarded matplotlib import.
- [ ] `__eq__` (optional) = mutual `contains`.
- [ ] Degenerate / empty handling: `Polytope().dim == 0` sentinel replicated
      exactly (several `if poly.dim:` guards depend on it).

### `backend_cdd.py`
- [ ] H→V, V→H, redundancy removal, facet enumeration via pycddlib 3.x.
- [ ] Wrap cdd number-type + `RepType` details; float DD path.
- [ ] Adjacency / incidence (needed by `adaptive_tools._extract_bases`).

### `operations.py`
- [ ] `image(M, S)` — `M·S` (V-rep map + hull).
- [ ] `preimage(M, S)` — `{x : M x ∈ S}` (H-rep: `S.A @ M x ≤ S.b`).
- [ ] `minkowski_sum(S, W)` — V-rep sum + hull (guard vertex blow-up).
- [ ] `pontryagin_diff(S, W)` — `Aᵢ x ≤ bᵢ − h_W(Aᵢ)` per row.
- [ ] `support(S, eta)` — `h_S(η)` (max over vertices / LP).
- [ ] `gauge(S, x)` — Minkowski functional (C-sets).
- [ ] `cartesian_product(*S)` — replaces `tools/cartesian_product.py`.

## 2. Set-theoretic control layer — `src/invariance/`

- [ ] `reach.py`:
  - [ ] `one_step_set(S, A, B, Z)` — `Q(S) = {x : ∃u, (x,u)∈Z, Ax+Bu∈S}`.
  - [ ] `pre(S, A)` — autonomous one-step (`{x : Ax ∈ S}`).
  - [ ] `n_step_reachable(...)`, `n_step_controllable(...)`.
- [ ] `invariant.py` — port `InvariantSet` / `myPolytope.invariant_set`:
  - [ ] maximal RPI in constraint set (Gilbert–Tan iteration).
  - [ ] maximal robust **λ-contractive** set (`A_K x ⊕ W ⊆ λ S ∩ constraints`).
  - [ ] convergence test via `contains` + per-iteration `minimize()`.
  - [ ] keep `project_on_axis` helper (project *away from* given axes).
- [ ] `mrpi.py` (optional) — minimal RPI outer approximation (Raković et al.).
- [ ] `GainSynthesis` stays where it is (cvxpy LMIs) — only its `Polytope`
      inputs change.

## 3. Migrate existing call sites

Order = lowest risk first. Regression-test the named example after each.

- [ ] `src/tools/polytope_system.py` — 3× box constructors → `sets.Polytope`.
- [ ] `src/tools/cartesian_product.py` → `sets.operations.cartesian_product`.
- [ ] `src/tools/invariance_tools.py` → `src/invariance/` (biggest change) —
      regression `contractive_example`.
- [ ] `src/tools/adaptive_tools.py` — `.V`, `Polytope(V=…).A/.b`,
      `chebyshev_centering`, `contains`, `dim` → new layer — regression
      `setmembership_example`, `filter_example`.
- [ ] `src/mpc.py` — `W`, `zc`, `zs`, `x0_poly.intersection(...)`,
      `xa_1.minimize_h_rep()`.
- [ ] `src/rmpc.py` — `self.V` from invariant set + unit-`b` rescale, `theta.V`.
      Watch: facet count `self.na` and `Hbar`/`HCbar` shapes — regression
      `rmpc_example` + turtlebot (RMSE + solver time vs 0.1.0 baseline).
- [ ] `src/rampc.py` — `E`, θ-set reconstruction (`Polytope()` empty path).
- [ ] Remove every `from pycvxset import Polytope` and
      `from pycvxset.common import constants`.
- [ ] Delete `src/tools/sm_tool.py` (untracked, superseded by
      `adaptive_tools.py`) — confirm first.

## 4. Example scripts — polytope & set operations

New directory `examples/sets/`, each script a sphinx-gallery item (docstring
header), each also usable as a smoke/doc fixture. Keep them small, 2–3D, with a
plot. They double as the "how the library builds its ingredients" tutorial.

- [ ] `01_polytope_basics.py` — construct from H-rep, V-rep, box helper, empty;
      inspect `.A/.b/.V/.dim`; H↔V round-trip; `copy`; plot a 2D polytope.
- [ ] `02_set_operations.py` — `intersection`, Minkowski `⊕`, Pontryagin `⊖`,
      `contains` (point and set), `minimize()` (redundant halfspaces removed),
      `normalize` / `with_unit_b`.
- [ ] `03_affine_and_functionals.py` — `image(M, S)`, `preimage(M, S)`,
      `projection(axes)`, `support(S, η)`, `chebyshev_centering`, `volume`.
- [ ] `04_reachability.py` — `pre(S, A)`, `one_step_set(S, A, B, Z)`,
      N-step reachable/controllable sets for a 2D linear system, nested plot.
- [ ] `05_invariant_sets.py` — maximal RPI in a constraint box, maximal
      λ-contractive set, (optional) minimal RPI; overlay the iterates.
- [ ] `06_tube_mpc_ingredients.py` — compose the above into the terminal set +
      constraint tightening used by `RMPC`; cross-link to `rmpc_example`.
- [ ] Wire `examples/sets/` into `docs/conf.py` sphinx-gallery
      (`examples_dirs`, add a `README.rst`).
- [ ] Add `01`–`03` to the `smoke` workflow (fast, no solver stack beyond cdd).

## 5. Tests — `tests/`

- [ ] Property tests: H↔V round-trip; `S ⊆ preimage(M, image(M, S))`;
      `(S ⊖ W) ⊕ W ⊆ S`; invariance fixed-point `pre(Ω) ∩ X ⊇ Ω`;
      `contains` reflexive / transitive.
- [ ] Numeric regression vs **pycvxset** on fixtures (constraint boxes, θ-sets,
      the turtlebot invariant set) — run **before** removing pycvxset; store
      expected `A/b/V`, Chebyshev centre, volume.
- [ ] Closed-loop regression: `rmpc_example`, `rampc_example`, turtlebot
      scenarios — assert RMSE within tolerance of the 0.1.0 numbers.
- [ ] `pytest` config in `pyproject.toml`; add `tests` to `dev` extra.

## 6. Packaging & license (option 1 → GPL)

- [ ] `pyproject.toml`: `license = "GPL-2.0-or-later"`, drop MIT classifier;
      `dependencies`: remove `pycvxset @ git+…`, add `pycddlib>=3,<4`, ensure
      `scipy` present; keep `cvxpy`.
- [ ] `LICENSE`: replace MIT text with GPL-2.0-or-later.
- [ ] `README` *License* section: state GPL, remove the AGPL/pycvxset caveat;
      simplify install (single path, `pip install -e .` self-contained).
- [ ] `CHANGELOG` `[0.2.0]`: relicense note + "pycvxset replaced by an internal
      cdd-backed polytope/invariance layer".
- [ ] `src/__init__.py`: bump `__version__ = "0.2.0"`; export the new
      `sets` / `invariance` public names if any.
- [ ] Check headers/notices required by GPL (per-file notice optional but
      conventional).

## 7. Docs

- [ ] `docs/conf.py`: drop `pycvxset` from `autodoc_mock_imports`; add `cdd`
      only if it can't be imported on the runner (prefer real install).
- [ ] `docs/installation.md` + `README`: remove pycvxset/`pycddlib`-from-source
      steps; `pip install pycddlib` from wheels.
- [ ] New API pages: `docs/api.rst` add `polypympc.sets`, `polypympc.invariance`.
- [ ] `docs/user_guide.md`: new "Working with sets" section referencing
      `examples/sets/`.
- [ ] Consider flipping sphinx-gallery `plot_gallery` to `"True"` for
      `examples/sets/` (cheap, cdd-only) while keeping the controller examples
      source-only.

## 8. CI

- [ ] `smoke.yml`: remove the editable-`pycvxset` clone; `pip install -e ".[viz]"`
      becomes self-contained; run `examples/sets/01..03`.
- [ ] `docs.yml`: unchanged (verify build still green without pycvxset mock).

## 9. Release 0.2.0

- [ ] Merge `pycvxset-free` → `main` via PR.
- [ ] `CHANGELOG` finalized; `git tag -a v0.2.0`.
- [ ] GitHub Release from the tag (body = `## [0.2.0]` section).
- [ ] Verify docs redeploy and `pip install` from a clean env (source + wheels).
- [ ] (Optional) first real PyPI upload now that there is no direct-URL dep.
