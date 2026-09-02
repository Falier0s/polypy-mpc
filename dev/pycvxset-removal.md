# pycvxset removal — analysis and plan

Working notes for the `pycvxset-free` branch (target: **0.2.0**).

## 1. Goal

Drop the `pycvxset` runtime dependency and replace it with a small internal
polytope handler built directly on **numpy / scipy** (plus `cvxpy`, already a
dependency, for LP-based operations; `pycddlib` optional for fast exact vertex
enumeration).

Motivation:

- **PyPI**: `pycvxset` is only a direct URL dependency
  (`pycvxset @ git+https://…`); PyPI rejects packages with direct-URL deps, so a
  real `pip install polypympc` is blocked until this is removed.
- **Licensing**: `pycvxset` is **AGPL-3.0-or-later**; removing it also removes
  the redistribution caveat currently noted in the README.
- **CI fragility**: the smoke workflow has to `git clone` + editable-install
  `pycvxset` because a non-editable install ships an incomplete package.

## 2. What pycvxset provides today

Only one symbol is imported: `from pycvxset import Polytope`. Examples also touch
`pycvxset.common.constants` to pick default solvers. **No** Minkowski sum,
Pontryagin difference, support functions, or affine-map *methods* are used
(affine maps are done by hand as `A @ x`).

## 3. API surface actually used

### Constructors
| form | meaning | reimplementation |
|---|---|---|
| `Polytope(A=H, b=k)` | half-space rep `{x : H x ≤ k}` | store `A`, `b` directly |
| `Polytope(V=verts)` | vertex rep (convex hull of rows) | store `V`; `.A`/`.b` via `scipy.spatial.ConvexHull` facets |
| `Polytope()` | empty / zero-dimensional | `dim == 0`, empty `A`/`b`; used as a sentinel behind `if poly.dim:` |

### Attributes
| attr | meaning | reimplementation |
|---|---|---|
| `.A`, `.b` | H-rep matrices | direct; lazy convert from V-rep |
| `.V` | **vertex enumeration** from H-rep | hard part — `pycddlib` (cdd) *or* `scipy.spatial.HalfspaceIntersection` (needs an interior point → Chebyshev LP) |
| `.dim` | ambient dimension, `0` if empty | `A.shape[1]` (or `V.shape[1]`), `0` if empty |

### Methods
| method | meaning | reimplementation |
|---|---|---|
| `.projection(axes)` / `.projection(i)` | project onto a coordinate subset | Fourier–Motzkin (pure numpy; dims here are 2–7 so OK) *or* vertex-enum → drop coords → hull |
| `.intersection(other)` | `{x : A₁x≤b₁} ∩ {x : A₂x≤b₂}` | stack `[A₁;A₂]`, `[b₁;b₂]`; redundancy removal optional |
| `.contains(point)` | membership | `np.all(A @ x <= b + tol)` |
| `.contains(polytope)` | `P ⊆ Q` | all vertices of `P` in `Q`, *or* an LP per facet of `Q` |
| `.chebyshev_centering()` | `(centre, radius)` of largest inscribed ball | one LP: `max r s.t. Aᵢx + r‖Aᵢ‖ ≤ bᵢ` (`scipy.optimize.linprog` / `cvxpy`) — **keep the tuple return** |
| `.volume()` | exact volume | vertex-enum + `ConvexHull.volume`; only used for reporting in two examples → could be approximate/optional |
| `.minimize_H_rep()` | drop redundant inequalities (in place) | one small LP per row; can start as a no-op, but matters for the invariant-set loop's termination/perf |
| `.normalize()` | row-normalize `A` (and `b`) by `‖Aᵢ‖` | trivial numpy (only referenced in a commented line) |
| `.copy()` | deep copy | copy the arrays |
| `.plot(ax, patch_args)` / `.projection(…).plot()` | 2D matplotlib patch | only in examples + one commented debug block — provide a tiny 2D helper or move plotting into the examples |

### Subclassing
`class myPolytope(Polytope)` in `invariance_tools.py` adds `invariant_set` and
`project_on_axis`. The new handler must be subclassable, or refactor to
composition.

### Solver config
Examples set `pycvxset.common.constants.DEFAULT_{LP,SOCP,SDP}_SOLVER_STR`.
After removal: internal LP/solve calls take an explicit `solver=` argument
(default e.g. `CLARABEL` via cvxpy or `highs` via scipy); the three
`constants.*` lines and the import are removed from every example.

## 4. Per-file usage map

### Leaf helpers (low risk — do first)
- **`src/tools/polytope_system.py`** — 3× box polytopes
  `Polytope(A=[I;-I], b=[ub;-lb])` for the X / Y / U constraint sets. Trivial.
- **`src/tools/cartesian_product.py`** — `[p.V for p in polytopes]` then
  `Polytope(V=<itertools.product of the vertex lists>)`. Needs input V-rep +
  hull of the product vertices; inputs are small-dim boxes / θ-sets.

### `src/tools/invariance_tools.py` — heaviest consumer
`InvariantSet` / `myPolytope.invariant_set`: iterative preimage
```
next = Polytope(A=[prev.A ;  (prev.A/prev.b) @ A_prop**k],
                b=[prev.b ;  λ**k · 1])
next.minimize_H_rep()
if next.contains(prev): converged
```
Also: `W.V` (disturbance vertex list), `Z_bnd.A/.b`, `.dim`, `.copy()`,
`.projection(2).plot()` (debug only). Needs H-rep construct,
`.contains(polytope)`, redundancy removal, and the vertex list of `W`.

### `src/tools/adaptive_tools.py`
- `_extract_bases`: `Polytope(A, b).V` (H→V) and
  `Polytope(V=verts).A / Polytope(V=verts).b[:,None]` (V→H, `b`-normalized).
- `initial_point_estimate`: `.chebyshev_centering()`, `.dim`,
  `.contains(point)`.
Needs vertex enumeration, hull, and the Chebyshev LP.

### `src/mpc.py`
`Polytope(A, b)` for `W` and `polys.zc`; `Polytope(V=zeros)` for `polys.zs`;
`x0_poly = Polytope(A, b).intersection(cartesian_product(...))`;
`xa_1 = Polytope(A, b); xa_1.minimize_H_rep()`. Otherwise `.A` / `.b` used as
plain arrays.

### `src/rmpc.py`
`self.V` is the tube facet polytope from
`invariant_set.compute_invariant_set()`, then rescaled to unit `b` via
`Polytope(A=V.A / V.b[:,None], b=1)`. `theta.V` vertex enumeration. Downstream
(`Hbar`, `HCbar`, tube constraints) is pure numpy on `self.V.A`.
**Attention:** a different-but-equivalent facet set changes `self.na` and the
shapes of `Hbar`/`HCbar` — fine mathematically, but regression-test the closed
loop.

### `src/rampc.py`
`Polytope(A, b)` for `E`; `Polytope(A=theta.A, b=theta_b)` / `Polytope()`
reconstruction of θ-sets; `.b`.

### Examples
Constructors, `.A` / `.b` / `.V`, `.volume()`, `.projection(...).plot()`,
`.intersection(...).plot()`, and the `constants.*` solver lines.

## 5. Proposed internal module

`src/polytope.py` (import as `polypympc.polytope`) exposing `Polytope`:

- `__init__(self, A=None, b=None, V=None)` — store H- and/or V-rep, lazy convert.
- properties: `A`, `b`, `V`, `dim`, `is_empty`
- methods: `contains(x_or_poly, tol=…)`, `intersection(other)`,
  `projection(axes)`, `chebyshev_centering()`, `minimize_h_rep()`,
  `normalize()`, `with_unit_b()` (the recurring `A / b[:,None]` pattern),
  `copy()`, `volume()` (optional), `plot(ax=None, **patch_args)` (guarded
  matplotlib import)
- backend: `numpy` + `scipy.spatial` (`ConvexHull`, `HalfspaceIntersection`) +
  `scipy.optimize.linprog`; optional `pycddlib` fast path for `.V`.

### Dependency changes
- `pyproject.toml`: remove `pycvxset @ git+…` from `dependencies`; `scipy` stays;
  add optional `cdd = ["pycddlib"]` extra (or drop pycddlib entirely if
  `HalfspaceIntersection` is enough).
- `docs/conf.py`: drop `pycvxset` from `autodoc_mock_imports`.
- `docs/requirements.txt`, `docs/installation.md`, `README.md`: remove pycvxset
  install steps; delete the AGPL caveat in the License section; simplify the
  now-single install path.
- `.github/workflows/smoke.yml`: drop the editable-`pycvxset` clone step;
  `pip install -e ".[viz]"` becomes self-contained.

## 6. Risks / attention points

1. **Vertex enumeration correctness & speed** in the invariant-set iteration
   (rows accumulate each step). `minimize_h_rep()` is what makes it terminate in
   reasonable time — do not ship it as a no-op for `invariance_tools`.
2. **Tube facet matrix** (`rmpc.V`): downstream shapes depend on the facet
   count. Any valid H-rep of the same set is acceptable, but add a closed-loop
   regression test (RMSE + solver time) on `rmpc_example` and the turtlebot
   scenarios.
3. **Empty-polytope semantics**: `Polytope().dim == 0` is a sentinel used in
   several `if poly.dim:` guards — replicate exactly.
4. **`chebyshev_centering()` returns `(centre, radius)`** — keep the tuple.
5. **Tolerances**: pick an explicit `tol` (e.g. `1e-9`) and a fixed LP solver;
   pycvxset's behaviour is solver-dependent.
6. **Dead file**: `src/tools/sm_tool.py` is untracked / git-ignored and appears
   superseded by `adaptive_tools.py`. Confirm, then delete rather than port.
7. `__eq__` between polytopes is only in a commented line — not needed
   (`a.contains(b) and b.contains(a)` gives set equality if ever wanted).

## 7. Suggested work order

1. Implement `polytope.py` + unit tests (H↔V round-trip, `contains`,
   `intersection`, `projection`, `chebyshev_centering`, `volume`) validated
   against `pycvxset` on fixtures.
2. Swap `tools/polytope_system.py`, `tools/cartesian_product.py`.
3. Swap `tools/invariance_tools.py`; regression `contractive_example`.
4. Swap `tools/adaptive_tools.py`; regression `setmembership_example`,
   `filter_example`.
5. Swap `mpc.py` / `rmpc.py` / `rampc.py`; regression `rmpc_example`,
   `rampc_example`, turtlebot examples (compare RMSE / solver time against the
   `0.1.0` baseline).
6. Update packaging, docs, workflows, README; drop the AGPL note.
7. Bump to `0.2.0`, update `CHANGELOG.md`.

## 8. Raw reference — every `pycvxset` / `Polytope` site

```
src/mpc.py:8      from pycvxset import Polytope
src/mpc.py:167    isinstance(self.options.W, Polytope)
src/mpc.py:170    self.W = Polytope(A=…, b=…)                 # disturbance set
src/mpc.py:196    self.polys.zc = Polytope(A=…, b=…)          # tightened constraint set
src/mpc.py:206    self.polys.zs = Polytope(V=np.zeros(...))   # steady-state set (V-rep)
src/mpc.py:285    isinstance(h_constraint, Polytope)
src/mpc.py:381    x0_poly = Polytope(A=…, b=…)
src/mpc.py:393    x0_poly.intersection( cartesian_product(...) )
src/mpc.py:397    x0_poly = Polytope(A=…, b=…)
src/mpc.py:860    xa_1 = Polytope(A=…, b=…)
src/mpc.py:883    xa_1.minimize_H_rep()
src/mpc.py:900    xa_1 = Polytope(A=…, b=…)
src/mpc.py:911    xa_1.minimize_H_rep()

src/rmpc.py:7     from pycvxset import Polytope
src/rmpc.py:55    Polytope(V=np.eye(self.q))                  # default theta
src/rmpc.py:68    Polytope(V=np.eye(self.q_c))               # default theta_c
src/rmpc.py:364   self.V = invariant_set.compute_invariant_set(check_mode=True)
src/rmpc.py:366   self.na = self.V.A.shape[0]
src/rmpc.py:367   self.V = Polytope(A = V.A / V.b[:,None], b = 1)   # unit-b rescale
src/rmpc.py:955   theta_vertices = hstack((ones, theta.V))          # vertex enumeration
src/rmpc.py:979   A_mat = self.V.A.copy()
src/rmpc.py:982   return Polytope(A=A_mat, b=b_vec)

src/rampc.py:7    from pycvxset import Polytope
src/rampc.py:58   isinstance(self.options.E, Polytope)
src/rampc.py:61   self.E = Polytope(A=…, b=…)                # measurement-noise set
src/rampc.py:133  Polytope(A=self.theta.A, b=theta_b)  /  Polytope()
src/rampc.py:140  Polytope(A=self.theta_c.A, b=theta_c_b)  /  Polytope()

src/tools/adaptive_tools.py:9     from pycvxset import Polytope
src/tools/adaptive_tools.py:48    Theta.A , Theta.dim
src/tools/adaptive_tools.py:283   Polytope(A=A, b=b).V                     # H -> V
src/tools/adaptive_tools.py:285   Polytope(V=vertices).A / .b[:,None]      # V -> H (unit-b)
src/tools/adaptive_tools.py:332   theta_set.chebyshev_centering()
src/tools/adaptive_tools.py:340   theta_set.contains(point)

src/tools/cartesian_product.py:5  from pycvxset import Polytope
src/tools/cartesian_product.py:26 v_rep = [p.V for p in polytopes]
src/tools/cartesian_product.py:34 return Polytope(V=np.array(combinations))

src/tools/invariance_tools.py:2   from pycvxset import Polytope
src/tools/invariance_tools.py:32  self.noise_vertices = W.V.copy()
src/tools/invariance_tools.py:219 Polytope(A = (Z_bnd.A @ _fmt)/Z_bnd.b[:,None], b = ones)
src/tools/invariance_tools.py:257 poly.projection(poly_axes) / poly.copy()
src/tools/invariance_tools.py:286 next_poly = Polytope(A=vstack(...), b=concat(...))
src/tools/invariance_tools.py:290 next_poly.projection(2).plot(...)          # debug
src/tools/invariance_tools.py:293 next_poly.minimize_H_rep()
src/tools/invariance_tools.py:295 next_poly.contains(prev_poly)
src/tools/invariance_tools.py:304 class myPolytope(Polytope)
src/tools/invariance_tools.py:352 _boundary_polytope = Polytope(A=…, b=…)
src/tools/invariance_tools.py:397 next_poly = Polytope(A=vstack(...), b=concat(...))
src/tools/invariance_tools.py:400 next_poly.contains(prev_poly)
src/tools/invariance_tools.py:428 list(range(poly.dim))

src/tools/polytope_system.py:3    from pycvxset import Polytope
src/tools/polytope_system.py:13   self.x = Polytope(A=[I;-I], b=[…])
src/tools/polytope_system.py:15   self.y = Polytope(A=[I;-I], b=[…])
src/tools/polytope_system.py:17   self.u = Polytope(A=[I;-I], b=[…])

src/tools/sm_tool.py               (UNTRACKED / git-ignored — superseded by adaptive_tools.py)

examples/*.py                      Polytope(A,b) / Polytope(V) / Polytope(),
                                   .A .b .V .volume() .projection().plot()
                                   .intersection().plot(),
                                   pycvxset.common.constants.DEFAULT_*_SOLVER_STR
```
