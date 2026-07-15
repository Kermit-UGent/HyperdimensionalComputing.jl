# JuliaCon Readiness TODO — HyperdimensionalComputing.jl

Audit date: 2026-07-13. All bugs below were **reproduced on the current `main`**
(tests pass: 268/268, but coverage gaps hide every one of these).
Ordered by priority: fix §1–§2 before anything else, §3–§5 before the talk,
§6–§7 as time allows.

---

## 0. Architecture (2026-07-14): the `encode` interface

The package now has an explicit layer taxonomy: **primitives** (`operations.jl`)
→ **combinators** (`encoding.jl`, hypervectors in/out — unchanged) → **encoders**
(`encode.jl`, raw data in, hypervector out). Key points:

- `encode(HV, x)` is the canonical, deterministic token path; `HV(x)` is sugar.
- Constructors have one meaning each: `HV(n::Number)` **throws** (the old
  one-time `@warn` and its testset are gone), invalid data arrays throw instead
  of silently token-encoding (this killed the `BinaryHV([1, 0])` trapdoor), and
  data constructors validate each type's element domain.
- Sequence strategies dispatch on `AbstractEncoding`: `KMer(k)` **resolves issue
  #53** (windows as atomic hashed tokens — genuinely different from `NGram(n)`,
  which shift-binds symbol encodings via `ngrams`); also `Sequence()` and
  `BagOfSymbols()`. Extension point: one struct + one `encode` method.
- [x] Stateful encoders as an `AbstractEncoder{HV}` hierarchy with
  `encode`/`decode` — **`LevelEncoder` landed (2026-07-15)**: builds its level
  set once at construction (fixes §1.4b by construction), replaces and deletes
  the `level`/`encodelevel`/`decodelevel`/`convertlevel` family. Ladder
  (perturbation, all types, `bandwidth` keyword) + fractional power encoding
  (FHRR, `β`), selected by dispatch. `encode` deliberately keeps its first
  argument free for encoder instances.
- [x] `RandomProjection` as the next `AbstractEncoder{HV}` — **landed
  (2026-07-15)**: fixed `D × d` projection matrix (`:gaussian`/`:bipolar`/
  `:sparse_ternary`), per-type nonlinearities, parametric scalar-or-vector
  ternary threshold `θ` (+ data-driven `target_sparsity` constructor,
  `rethreshold` sharing `R`), supplied-matrix constructor, FHRR = random
  Fourier features via the shared internal `phase_encode(z, β)` helper (also
  now used by `LevelEncoder`'s fractional power path — single source of
  truth). Closes the colour/embedding (RGB / ESM-2-style feature-vector)
  random-projection gap.
- [ ] Shared item-memory/codebook abstraction (labelled HV collection +
  nearest-neighbour lookup): now seeded by `LevelEncoder(levels, values)` — its
  `decode` is built on that stored collection. `RandomProjection`'s
  `decode(rp, hv, references)` clean-up is the **second consumer** of the same
  shape (currently via raw `nearest_neighbor` over a user-supplied reference
  set); the tutorial's colour reverse-lookup and a future `train`/`predict`
  need it too. Factor it out.

---

## 1. Confirmed bugs (release blockers)

### 1.1 UnicodePlots extension fails to precompile — FIXED (2026-07-14)
- [x] `show` now lives only in `src/representations.jl` and delegates to the
  extension's `_show_rich` via `Base.get_extension` when UnicodePlots is loaded
  (plain Base array display otherwise, and always under `:compact`). The plain
  header comes from per-type `Base.summary` methods; Base handles elements and
  `⋮` truncation. The extension defines no `Base.show` methods and precompiles
  cleanly; `unicodeheatmap`/`unicodehistogram` are exported, user-callable stubs
  implemented in the extension.

### 1.2 Displaying a `BipolarHV` with the extension loaded throws — FIXED (2026-07-14)
- [x] `getindex` restricted to `i::Integer` plus a non-scalar
  `I::AbstractVector` method on all types (`ifelse.(hv.v[I], 1, -1)` for
  `BipolarHV`). Decision, documented in the `AbstractHV` docstring: non-scalar
  indexing returns a plain `Vector` of element values, never a new hypervector.
  Covered by getindex tests in `test/representations.jl`.

### 1.3 `perturbate` is broken for FHRR — FIXED (2026-07-14)
FHRR-specific `perturbate!` methods resample phases (`exp(2πi·rand())`) at the
selected positions, keeping unit modulus; `level(::FHRR, ...)` dispatch verified.
Locked by FHRR perturbate tests in `test/operations.jl`. Original notes:
`eldist(::Type{<:FHRR})` is not defined, so `perturbate(FHRR(), 10)` throws a
`MethodError` (`src/operations.jl:290` calls `eldist(hv)`; `src/types.jl` defines
`eldist` for every type except FHRR). Even with an `eldist`, the generic byte-vec
path would need unit-modulus draws.
- [x] Implement FHRR perturbation properly (resample phases, e.g.
  `exp(2πi·rand())`, or add phase noise) and add a test.
- [x] Consequence: `level(::FHRR, ...)` correctly has its own `^`-based method,
  but generic `level()` on FHRR via the `AbstractHV` fallback would also break — verify dispatch.

### 1.4 `decodelevel` / `convertlevel` instance path is broken — FIXED (2026-07-14)
The vector `decodelevel` methods now accept (and ignore) `testbound`, so the
generic instance-path forwarding no longer mis-dispatches. Locked by tests in
`test/encoding.jl`. **See the new §1.4b found while verifying.** Original notes:
`decodelevel(BipolarHV(), 0:0.1:1)` throws
`MethodError: no method matching level(::Vector{BipolarHV}, ::Int64)`.
Cause: `src/encoding.jl:595` forwards `; testbound` to
`decodelevel(hvlevels::AbstractVector{<:AbstractHV}, numvalues)` (`:583`) which
accepts **no kwargs**, so the call re-enters the generic method with the level
vector as first argument. `convertlevel(hv, vals)` for non-FHRR types is broken
the same way. (Same family as the already-fixed `convertlevel` kwargs bug.)
- [x] Accept (and use or ignore) `testbound` in the vector method, or stop
  forwarding it; add tests for the instance-based `encodelevel`/`decodelevel`/`convertlevel` paths.

### 1.5 `bundle` silently drops custom distributions — FIXED (2026-07-14)
`bundle` and `bind` now propagate the first operand's `distr` for
RealHV/GradedHV/GradedBipolarHV; locked by a testset asserting `.distr === x.distr`
and that `normalize!(x + y)` rescales to the original spread. Original notes:
For `RealHV`/`GradedHV`/`GradedBipolarHV` with a non-default `distr`, bundling
returns a result carrying the **default** distribution
(`src/operations.jl:130–152` construct `RealHV(r)` etc. without passing `distr`).
`normalize!(::RealHV)` rescales by `std(hv.distr)`, so this changes numerics, not
just metadata. Same bug class as the fixed `GradedHV.similar` issue.
- [x] Propagate `first(hdvs).distr` in the three `bundle` methods; check `bind`
  (`operations.jl:176–178`) for the same issue; add tests.

### 1.6 `shift!` returns the wrapped vector, not the hypervector — FIXED (2026-07-14)
Generic `shift!` and the three clamp!-based `normalize!` methods now return the
hypervector; locked by a testset asserting `op!(hv) === hv` for all types and all
in-place ops. Original notes:
Generic `shift!(hv, k) = circshift!(hv.v, k)` (`src/operations.jl:201`) returns a
raw `Vector`, while the `BinaryHV`/`BipolarHV` methods (`:209`) return the HV.
The README even shows `shift!(x, 2)` printing a `Vector{Float64}` as if normal.
- [x] Make all in-place ops return the hypervector; same check for `ρ!`,
  `normalize!` (mostly OK) and `perturbate!`.

### 1.7 `δ` is a non-`const` global — FIXED (2026-07-14)
`const δ = similarity`; locked by an `isconst` test. Original notes:
`src/inference.jl:84`: `δ = similarity` — an exported, untyped, non-const
binding (type-unstable at call sites, and `isconst(HDC, :δ) == false`).
- [x] Change to `const δ = similarity`.

### 1.2b BipolarHV bit↦value convention flipped — COMPLETED (2026-07-14)
All follow-ups landed via a systematic audit of every `.v`-touching site (25 sites,
each given an explicit verdict): sign-based `BipolarHV(v::AbstractVector{<:Real})`
constructor that **throws on zero elements** (pointing to `TernaryHV`); summary
labels corrected; docstring rewritten (XOR IS the ±1 product, `x * x` = all-`+1`
identity, Bool vectors = raw stored bits); doctests + README regenerated;
polarity-locking tests added (`all(x * x .== 1)`, construction/indexing
round-trip, zero-throw, summary-count checks) since all prior property tests were
polarity-blind. Verified correct without change: bundle majority vote (mapping-
invariant), the bipolar `dot` formula (algebraically matches the new mapping),
`isapprox` (element-based), hash/isequal (storage-based, self-consistent),
shift/perturbate (bit-level, symmetric). Noted under issue #15: tie-break
outcomes for identical inputs flipped sign with the mapping (still a fair coin).

Original notes below for the record:
Michiel changed the mapping to `Base.getindex(hv::BipolarHV, i::Integer) = hv.v[i] ? -1 : 1`
(stored bit `true ↦ -1`, `false ↦ +1`). **This is deliberate**: with `v = (-1)^bit`,
XOR on the stored bits is now *exactly* the elementwise `±1` product, so bind-as-XOR
is the true MAP multiply (the old "negated product" wart is gone). `sum` was updated
accordingly; `dot` is symmetric under the flip and needed no change. Verified:
`x * y == x .* y` elementwise, `sum`/`dot` consistent with element values.

Remaining inconsistencies found by verification (not yet fixed):
- [x] `BipolarHV(v::AbstractVector{<:Integer}) = BipolarHV(v .> 0)` (`src/types.jl:213`)
  now **negates the input**: `BipolarHV([-1, 0, 1])` gives values `[1, 1, -1]`.
  Should be `v .< 0`; decide what `0` maps to (it flips from `-1` to `+1` semantics).
  The test `BipolarHV([-1, 0, 1]) == BipolarHV([false, false, true])` in
  `test/types.jl` currently locks in the inverted behaviour — update it with the fix.
- [x] `Base.summary(io, hv::BipolarHV)` (`src/representations.jl:18`) still labels
  `count(hv.v)` as "positives", but true bits are now `-1`: `[-1, -1, 1]` prints as
  "2 positives and 1 negatives". Swap the labels (the display test only checks the
  pattern, so it passes silently).
- [x] BipolarHV docstring: delete the now-false "*negated* elementwise product" note
  (it IS the product now — say so), fix "positive entries become `+1`" once the
  integer constructor is fixed, and regenerate its doctest outputs
  (`doctest(fix = true)`) — element signs flipped, so **the doctest CI job fails
  until this is done**.

### 1.4b Instance-path `convertlevel` builds encoder and decoder over DIFFERENT ladders — FIXED (2026-07-15)
Fixed structurally by the `LevelEncoder` refactor (§0): the level set is built
once in the constructor and shared by every `encode`/`decode` call, and `seed`
makes the whole ladder deterministic. The old function family (including the
broken instance path) is deleted, no deprecation shims. Locked by the
"one shared level set" testset in `test/encoding.jl` (exact grid round-trips +
`!isdefined` checks for the removed names). Original notes:
Found (by execution) while fixing §1.4, deliberately NOT fixed in that PR.
`convertlevel(hv::AbstractHV, numvals)` calls `encodelevel(hv, ...)` and
`decodelevel(hv, ...)`, each of which builds its own `level(hv, m)` ladder — and
`level` perturbation is unseeded, so the two ladders share only the base vector.
Measured: `decode(encode(x))` errors up to 1.0 (mean 0.41) on the instance path vs
exactly 0.0 when both are built from one shared `level(...)` ladder.
- [x] Fix: build the ladder once (now: in the `LevelEncoder` constructor), then
  assert the roundtrip in tests.

### 1.5c `perturbate` resamples from the TYPE-default distribution, not `hv.distr` — FIXED (2026-07-14)
Instance-level `eldist(hv) = hv.distr` methods added for RealHV/GradedHV/
GradedBipolarHV (the type-level defaults stay for constructors); `level()` ladders
built from custom-distr hypervectors are fixed by the same dispatch. Locked by a
resampled-element statistics testset in `test/operations.jl`. Original notes:
Found (by execution) during the §1.5 pattern sweep, deliberately not fixed in that
PR. `eldist(hv::AbstractHV) = eldist(typeof(hv))` (`src/types.jl`) ignores the
instance's `distr`, and the byte-vec `perturbate!` methods draw replacement
elements from it. Measured: `perturbate(RealHV(distr = Normal(0, 5)), 5000)`
resamples with std ≈ 1.01 (should be ≈ 5); `GradedHV(distr = Beta(10, 2))` (mean
0.833) resamples with mean ≈ 0.498. The result *carries the right `distr`
metadata but wrong-distribution elements* — the same silent-numerics family as
§1.5. Also affects `level()` ladders built from custom-distr hypervectors.
- [x] Fix: `eldist(hv::RealHV) = hv.distr` (and GradedHV/GradedBipolarHV);
  lock with a resampled-element statistics test like the ones above.

### 1.5b `unbind` / `/` for RealHV — RESOLVED by design decision (2026-07-14)
Was an accidental `MethodError` (the old `Union{RealHV, FHRR}` division method
constructed via the concrete type). **Decision (Michiel): real-valued MAP binding
is not exactly invertible, so `unbind(::RealHV, ::RealHV)` now throws an explicit
`ArgumentError`** pointing users to `similarity` against candidate hypervectors,
or to `FHRR`/`BipolarHV` for exact unbinding. FHRR kept its exact elementwise
division. `unbind`, `bind` and `RealHV` docstrings updated accordingly; covered
by the new `unbind` testset (see §5).

### 1.7b TernaryHV pretty-printing branch is dead code — FIXED (2026-07-14)
- [x] Resolved by the `Base.summary` refactor (§1.1): the ternary header
  dispatches on `::TernaryHV` and now actually shows positives/zeros/negatives
  (visible in the regenerated TernaryHV doctest).

### 1.8 Cross-type equality is wrong — FIXED (2026-07-14, strengthened later same day)
First pass: same-type storage fast path for `isequal`, one-arg `hash` removed
(hashing element-based via the AbstractArray fallback). Strengthened after the
polarity flip made the remaining hole real: Base's numeric fallback let an
all-true BinaryHV equal an all-+1 BipolarHV (`true == 1`) even though their
stored bits are opposite. `==`/`isequal` between DIFFERENT hypervector types are
now strictly `false`; same-family cross-parameter (TernaryHV{Int8} vs {Int64})
compares by value; comparisons against plain vectors stay elementwise, which is
also why hashing must remain the unsalted element-based fallback
(`isequal(hv, ::Vector)` can be true and must imply equal hashes).
Locked by an equality/hashing testset (same bits, different type ⇒ not equal).
Original notes:
`Base.isequal(v::AbstractHV, u::AbstractHV) = v.v == u.v` (`src/operations.jl:225`)
makes `isequal(BinaryHV([1,0]), BipolarHV([1,0])) == true` — a binary vector
"equals" a bipolar one because the underlying BitVectors match. Also
`Base.hash(hv) = hash(hv.v)` (`src/types.jl:28`) is a one-arg overload that is
inconsistent with the two-arg `hash(hv, h)` fallback for `BipolarHV`
(elements hash as ±1, storage hashes as bits).
- [x] Restrict `isequal`/`==` to same HV type; define `hash(hv, h::UInt)`
  consistently with equality instead of the one-arg form.

---

## 2. API design decisions (breaking — settle *before* registering)

### 2.1 The positional-constructor trap (biggest UX footgun)

> **Note (2026-07-13, Michiel):** this is a *deliberate*, already-decided breaking
> change — the intended interface is `HV(this::Any; D::Int = 10_000, seed/rng)`,
> where the positional argument is always the thing to encode (seeded by its hash),
> and some types add special constructors depending on the datatype. The design was
> documented once, but the commits were lost. So the task below is **not** to
> re-decide the API but to (re)document the convention prominently (docstrings on
> every constructor + a docs section) and bring README/docstring examples/tests in
> line with it.

`HV(x)` for any non-vector `x` means "deterministic vector seeded by `hash(x)`",
**including integers**:
- `BipolarHV(6)` → a 10,000-dim vector seeded by `hash(6)` — but the README
  claims it creates a 6-element vector and shows fabricated output.
- `[BinaryHV(10) for _ in 1:10]` → ten **identical** vectors. Every docstring
  example in `src/encoding.jl` uses this idiom and shows made-up 10-element
  outputs that the code cannot produce (GitHub issue #36).
- Even the test suite fell into it: `test/operations.jl:25,45` bundle/bind five
  identical `HV(N)` vectors, weakening the tests.
- [x] Re-document the convention: canonical statement + warning admonition on the
  `AbstractHV` docstring; constructor docstrings on all 7 types (2026-07-14).
- [x] Fix README (rewritten Usage section with real outputs), all `encoding.jl`
  docstring examples (regenerated with `HV.(tokens; D = 10)` and real outputs,
  including a filled-in `graph` example), and the tests (`HV(i; D = N)` in
  `test/operations.jl`; new "constructor convention" testset in `test/types.jl`
  locking `length(HV(42)) == 10_000`, determinism, and `D` kwarg for all types).
- [x] Runtime guard added (2026-07-14): all token constructors call
  `warn_integer_token`, which emits a one-time-per-session `@warn` (via
  `maxlog = 1`) when an `Integer` is passed positionally, pointing the user to
  `D = n`. `Bool` tokens are exempt; behavior is unchanged (still encodes the
  integer). Covered by `@test_logs` tests in `test/types.jl`.

### 2.2 Extending `Base.bind`
`bind` already exists in `Base` (for Channels/Sockets); the package extends and
re-exports it (`src/operations.jl:172`). Legal (own types), but surprising and
flagged by tooling.
- [ ] Decide: own `bind` function (breaks `Base.bind` extension) vs keep. Run
  Aqua.jl piracy/ambiguity checks either way (see §5).

### 2.3 `similar` returns a *random* vector
`Base.similar(hv)` (`src/types.jl:29`) violates the Base contract (uninitialized
container) by returning a fresh random hypervector, and does so with the global RNG.
- [ ] Rename the internal helper (e.g. `randlike(hv)`) and stop overloading `Base.similar`,
  or document the deviation prominently.

### 2.4 `normalize` name clash
`normalize`/`normalize!` are package-own functions (only `norm`/`dot` are imported
from LinearAlgebra), so `using LinearAlgebra, HyperdimensionalComputing` gives
users an export conflict.
- [ ] Extend `LinearAlgebra.normalize(!)` instead of defining new functions.

### 2.5 Smaller decisions to document (or fix)
- [x] ~~`TernaryHV()` random constructor never generates 0 — document~~ (documented in the TernaryHV docstring, 2026-07-14).
- [x] ~~No `setindex!`: document immutability on `AbstractHV`~~ (documented in the `# Indexing` section, 2026-07-14).
- [ ] FHRR `unbind` uses elementwise `/`; conjugate multiplication is the idiomatic,
  numerically safer choice for unit-modulus vectors.
- [ ] FHRR `bundle` does `r ./= abs.(r)` — NaN if elements cancel exactly; guard or document.
- [x] ~~Default RNG is `MersenneTwister`~~ — RNG handling modernized (2026-07-14):
  `Xoshiro` throughout, `Random.GLOBAL_RNG` → `Random.default_rng()`, the `rng`
  keyword now takes an `AbstractRNG` *instance* (`HV(; D, seed, rng = Random.default_rng())`,
  `seed` builds a fresh `Xoshiro(seed)`), and the deterministic constructor is
  `HV(this; D)` — no `rng` keyword; always `Xoshiro(hash(this))`. Breaking in
  output: every seeded/token vector changed; README + example outputs regenerated.

### 2.6 `RandomProjection(TernaryHV, ::AbstractMatrix)` positional collision (found 2026-07-15)

The merged ternary constructor reads the positional matrix as **training data**
when `target_sparsity` is given and as a **supplied projection matrix**
otherwise — same positional shape, opposite meanings, disambiguated only by
keyword presence. The "ternary constructor: positional collision" testset in
`test/encoding.jl` pins what tests *can* pin: each path's documented reading,
and that a data-shaped (non-square) matrix misread as a projection matrix
cannot encode the intended features (immediate `DimensionMismatch` at first
use). What tests cannot make safe, deliberately not patched in that test-only
pass:

- A **square** matrix is genuinely ambiguous: both readings yield a working
  encoder, so a forgotten `target_sparsity` silently produces a projection
  encoder built from data (and vice versa is undetectable).
- The misread's error is a downstream feature-length `DimensionMismatch`; it
  never names the actual mistake (matrix misread as R instead of X).
- The ternary method's signature carries the data-path keywords, so the
  supplied-matrix path **silently accepts and ignores** `D`, `matrix` and
  `seed` (`RandomProjection(TernaryHV, R; D = 500)` returns a
  `size(R, 1)`-dimensional encoder); the generic-type equivalent throws a
  `MethodError`. Inconsistent and unlockable without blessing it.
- [ ] Fix by renaming one path rather than patching: a keyword-only or
  distinctly named data-driven constructor (e.g. `fit_sparsity(TernaryHV, X;
  target_sparsity, ...)` or `RandomProjection(TernaryHV, d; from_data = X)`),
  and reject inapplicable kwargs on the supplied-matrix path.

---

## 3. README (the first thing JuliaCon attendees will open)

- [x] ~~All usage examples are stale~~ — Usage section rewritten (2026-07-14):
  constructor convention explained with real outputs, `D` kwarg shown, operations
  and similarity blocks regenerated; the misleading `shift!` output block dropped.
- [ ] Broken CI badge: points to `MichielStock/...` and a workflow named `CI`;
  the repo is `Kermit-UGent/...` and the workflow is `Test suite`.
- [x] ~~Truncated sentence "…For each VSA"~~ (completed 2026-07-14).
- [ ] Claims "Basic functionality for fitting a k-NN like classifier is also
  supported" — no `train`/`predict` exists (see §7.1). Remove the claim or ship the feature.
- [ ] Update installation instructions once registered (`Pkg.add("HyperdimensionalComputing")`).
- [ ] Unify org capitalization (`KERMIT-UGent` vs `Kermit-UGent`) in links.
- [ ] Add a docs badge that actually resolves + a Codecov badge once §5 lands.

---

## 4. Documentation

### Structural
- [ ] `docs/src/index.md` `@contents` references non-existent `examples.md`
  (actual pages live in `examples/`).
- [ ] `docs/src/developers.md` is built but missing from the `pages` nav in `docs/make.jl:35–42`.
- [ ] `makedocs(repo = "...string...")` triggers a navbar warning — pass
  `Remotes.GitHub("Kermit-UGent", "HyperdimensionalComputing.jl")`.
- [ ] Committed generated files `docs/src/examples/*.md` are stale Literate output
  from the **old `BipolarHDV` API** (incl. a `Handcalcs` reference). They're
  regenerated at build time — delete and gitignore them.
- [ ] `docs/Project.toml`: `Handcalcs` is no longer used by any tutorial — remove.
  `CairoMakie`/`Colors` are only needed by `logo.jl` — consider a separate env so
  docs CI doesn't compile Makie. Add compat bounds (esp. Documenter).
- [ ] Restructure docs per issue #32: pages for Types/VSAs, Operations,
  Comparison & inference, then examples.

### Content
- [x] ~~Missing docstrings on type names~~: all 7 types + `AbstractHV` rewritten to a
  shared template with seeded `jldoctest` examples (2026-07-14); `bundle` and `bind`
  got docstrings too. Still missing: `shift`/`ρ`, `perturbate(!)`, `^` for FHRR.
  Note: the type docstrings contain literal `<<<DECIDE>>>` markers for the documented
  RNG default and the binary/bipolar element-set notation — resolve before release.
- [x] ~~`# Example` blocks in `src/encoding.jl` use the `BinaryHV(10)` trap with
  fabricated outputs~~ — rewritten with token-based construction and real outputs
  (2026-07-14). Type docstrings now use `jldoctest` with outputs generated by
  `doctest(fix = true)` and enforced in CI (doctests job in `test.yml` + the docs
  build). Follow-up: migrate the `encoding.jl` examples to `jldoctest` too.
- [x] ~~`graph` docstring: `# Example` section is empty~~ (filled in 2026-07-14).
- [ ] Second `isapprox` docstring (`src/operations.jl:248–256`): header line shows
  the wrong kwargs (`atol`/`ptol` instead of `ptol`/`N_bootstrap`) and says
  `N_bootstrap=200` while the code default is `500`.
- [ ] `similarity(u, v; method)` docstring says "When no method is given, a default
  is used" but `method` is a mandatory kwarg for plain vectors (`src/inference.jl:38`).
  Also `:hamming` returns a match *count*, not a similarity in [0,1] — document or normalize.
- [ ] Typo `src/inference.jl:71`: "and `u`` " (stray backtick).
- [x] ~~`level` docstring: document the perturbation-based correlation mechanism and the FHRR `^`-based variant~~ (obsolete: the family was replaced by `LevelEncoder`, whose docstring documents both mechanisms with doctests).
- [ ] Issue #36: the intro tutorial's ngrams example yields the wrong result — re-derive it.
- [ ] Developer docs: how to add a new HV type (required methods: constructor,
  `eldist`, `empty_vector`, `bundle`, `bind`, `similarity`, traits…).

---

## 5. Tests & QA

Coverage gaps (each hid a §1 bug):
- [x] ~~`unbind` / `/`: zero tests package-wide~~ — `unbind` testset added to
  `test/operations.jl` (2026-07-14): exact roundtrips for Binary/Bipolar/Ternary,
  approximate fuzzy recovery for the graded types, exact FHRR division, and the
  explicit `ArgumentError` for RealHV.
- [ ] `perturbate` on FHRR (currently broken, §1.3).
- [x] ~~Instance-path `encodelevel`/`decodelevel`/`convertlevel` (currently broken, §1.4)~~ (family deleted; `LevelEncoder` has its own testset: all 7 types, round-trips, bandwidth, FPE, precomputed constructor, bounds).
- [ ] `bundle`/`bind` preserving custom `distr` (§1.5).
- [ ] `similarity(...; method = :jaccard / :hamming)` — only `:cosine` is tested.
- [ ] `isapprox` bootstrap path — the whole similarity testset is skipped for
  `TernaryHV`/`GradedHV`/`GradedBipolarHV`/`RealHV` (`test/operations.jl:71`).
- [x] ~~`show` methods and the UnicodePlots extension untested~~ — added
  `test/representations.jl` (plain display for all 7 types + getindex) and
  `test/ext_display.jl` (rich display, run in a separate Julia process since
  extension loading is irreversible) (2026-07-14).
- [ ] Semantic property tests, not just type checks: bundle preserves similarity
  to inputs; bind produces dissimilar output and is invertible via unbind;
  shift preserves distance. Currently most tests only assert `isa`.

Test hygiene:
- [x] ~~`test/operations.jl`: `HV(N)` creates identical seeded vectors~~ (fixed to `HV(i; D = N)`, 2026-07-14).
- [ ] Cross-file leakage: `n`, `s`, `hash_s` consts from `test/types.jl` are used
  by `test/operations.jl` (`:90`) — scope per file or move to `runtests.jl`.
- [ ] Declare `Distributions` and `Random` in `[extras]`/test target (currently
  only `Test`); works today via the sandbox but is fragile and blocks test/Project.toml migration.
- [ ] Add **Aqua.jl** (ambiguities, piracy — will flag `Base.bind`, unbound args,
  stale exports, compat) and run **doctests** in CI.

---

## 6. CI / infrastructure / registration

- [ ] `test.yml`: `setup-julia@v1` → `v2`; test a matrix — `min` (1.11), `1`,
  `pre` — and at least ubuntu + one of macOS/Windows.
- [ ] Add coverage (`julia-processcoverage` + Codecov) — closes the workflow TODO.
- [ ] Add `CompatHelper.yml` and `TagBot.yml` (required plumbing for a registered package).
- [ ] **Register in the General registry** (issue #9) — the JuliaCon audience must be
  able to `] add HyperdimensionalComputing`. Compat bounds and license are already
  in place; do this *after* the breaking decisions in §2.
- [ ] `docs.yml` warning: deploy config fine, but fix the `repo` remote (§4).
- [x] ~~Drop the `StatsBase` strong dependency~~ — removed from `[deps]` and
  `[compat]` (2026-07-14); the extension no longer needs it (`mean`/`std` in
  summaries come via the existing Distributions re-export).

---

## 7. Features & polish (nice-to-have for the talk)

- [ ] **Classification workflow** (`train`/`predict`, prototype = bundle of class
  examples + retraining on misclassified) — HDC's headline use case, promised by
  the README, and the natural JuliaCon demo (language identification / MNIST, issue #32).
- [ ] Dead code in `src/operations.jl`: `aggfun`, `bindfun`, `neutralbind`,
  `noisy_and`, `elementreduce!`, `offsetcombine(!)` and `Base.zeros(::GradedHV)`
  (`src/types.jl:271`) are never called — delete or wire in.
- [ ] `crossproduct` TODO: should bundle without normalizing (`src/encoding.jl:373`).
- [ ] `SparseHV` (long-standing TODO in `src/types.jl`) — fine to skip for JuliaCon.
- [ ] Triage open issues for the roadmap slide: #53 k-mer encoder, #50 VSA aliases,
  #42 Makie recipe, #25 concatenation, #16 level-encoding strategies,
  #15 tie-breaking strategies, #14 naming conventions, #10 stateful encoding.

---

## Suggested order of attack (updated 2026-07-14, end of session)

Done so far: constructor convention documented + runtime guard; RNG modernized
(Xoshiro, instance `rng`, `HV(this; D)`); display/extension architecture fixed and
tested; all 8 type docstrings templated with CI-enforced doctests; `unbind`
tested everywhere and made an explicit error for RealHV; README rewritten;
StatsBase dropped. Test suite grew 268 → ~450 assertions plus doctests.

1. **§1.2b BipolarHV mapping-flip follow-ups — do this first, the doctest CI job
   is red until it lands**: fix the integer-vector constructor (`v .> 0` →
   `v .< 0`, decide zero semantics), swap the summary labels, refresh the
   BipolarHV docstring text, re-run `doctest(fix = true)`, update the locked-in
   test in `test/types.jl`.
2. Remaining §1 bugs, one small PR each with a locking test: `perturbate` on FHRR
   (§1.3), instance-path `decodelevel`/`convertlevel` (§1.4), `bundle` dropping
   `distr` (§1.5), `shift!` return value (§1.6), `const δ` (§1.7),
   cross-type `isequal`/`hash` (§1.8).
3. §2 API decisions before registration: `Base.bind` extension (run Aqua.jl and
   decide), `similar` returning a random vector, `normalize` name clash.
4. Registration track (§6): CI matrix (min/1/pre, macOS/Windows), coverage +
   Codecov, CompatHelper + TagBot, Aqua in tests → register in General (issue #9).
5. Docs polish (§3/§4): README badge + k-NN claim, docs restructure per issue #32,
   fix the intro tutorial ngrams example (issue #36), remaining function docstrings
   (`shift`/`ρ`, `perturbate`, `level`, `^` for FHRR).
6. §7.1 classification workflow (`train`/`predict`) — the JuliaCon demo.
