# CLAUDE.md - HyperdimensionalComputing.jl

## What is this package?

HyperdimensionalComputing.jl is a Julia package implementing Vector Symbolic Architectures (VSA) for hyperdimensional computing. HDC represents information as high-dimensional vectors (typically D=10,000) where three core operations — **bundling** (aggregation/superposition), **binding** (association/conjunction), and **permutation** (shifting) — are composed to encode arbitrary data structures. The high dimensionality guarantees that random vectors are quasi-orthogonal, enabling robust distributed representations.

## Authors

Carlos Vigil-Vásquez, Dimi Boeckaerts, Michiel Stock, Steff Taelman

## Project structure

```
src/
  HyperdimensionalComputing.jl  # Main module, exports
  types.jl                      # AbstractHV and 7 concrete vector types
  operations.jl                 # bundle, bind, unbind, shift, perturbate
  encoding.jl                   # Combinators (ngrams, hashtable, ...): hypervectors in, hypervector out
  encode.jl                     # Encoder layer: raw data in, hypervector out (encode + strategies)
  inference.jl                  # similarity, nearest_neighbor
  representations.jl            # Custom show/display methods
ext/
  UnicodePlotting.jl            # Extension for histogram/heatmap display via UnicodePlots
test/
  runtests.jl                   # Test runner
  types.jl, operations.jl, encoding.jl, inference.jl, representations.jl
  ext_display.jl                # rich-display tests, run in a separate process (extension loading is irreversible)
docs/
  src/examples/                 # Literate.jl tutorials (intro-to-hdc, dollar-of-mexico)
```

## Supported vector types (all subtypes of `AbstractHV{T} <: AbstractVector{T}`)

| Type | Elements | Algebra | Key trait |
|------|----------|---------|-----------|
| `BinaryHV` | {false, true} (`Bool`, displays as 0/1) | BSC (Binary Spatter Codes) | BitVector storage |
| `BipolarHV` | {-1, +1} | MAP (Multiply-Add-Permute) | BitVector storage; bit `true ↦ -1`, `false ↦ +1`, so XOR on stored bits IS the ±1 product. Data must be exactly ±1; **zero elements throw** (use `TernaryHV`); Bool vectors are raw bits |
| `TernaryHV` | {-1, 0, +1} | MAP variant | Vector{Int} |
| `RealHV` | ℝ | Continuous MAP | Configurable distribution |
| `GradedHV` | [0, 1] | Fuzzy logic | Beta distribution default |
| `GradedBipolarHV` | [-1, 1] | Fuzzy logic bipolar | Scaled Beta default |
| `FHRR` | Complex unit circle | Fourier HRR | e^(iθ) elements, supports `^` |

Default dimension is 10,000 for all types. Layer taxonomy: **primitives** (operations.jl: bundle/bind/shift/perturbate) → **combinators** (encoding.jl: multiset, ngrams, ... — hypervectors in, hypervector out) → **encoders** (encode.jl: raw data in, hypervector out).

Constructor convention (settled; each form has exactly one meaning, documented on `AbstractHV`):

- `HV(; D = 10_000, seed = nothing, rng = Random.default_rng())` — random; `rng` takes an `AbstractRNG` **instance**, `seed` builds a fresh `Xoshiro(seed)`
- `HV(v::AbstractVector{<:Real})` — wrap element data, **validated** against the type's domain (BinaryHV {0,1}; BipolarHV ±1, zero errors pointing to TernaryHV; TernaryHV {-1,0,+1}; Graded types range-checked; FHRR unit modulus). Invalid arrays throw — they never silently token-encode
- `HV(x)` — token shorthand for `encode(HV, x)`; `HV(n::Number)` **throws an ArgumentError** naming both alternatives (`D = n` / `encode(HV, n)`) — this replaced the old one-time `@warn`
- Tuples of reals read as data, like vectors

`encode` is the canonical interface: `encode(HV, x; D)` is the deterministic token path (hash → seed), `encode(HV, x, strategy)` dispatches on `AbstractEncoding` strategies: `KMer(k)` (windows as atomic hashed tokens — resolves issue #53), `NGram(n)` (symbol-level shift-binding via `ngrams`), `Sequence()` (bundlesequence), `BagOfSymbols()` (multiset). New strategies = one struct + one `encode` method.

Stateful encoders live in the `AbstractEncoder{HV}` hierarchy (encode.jl): instances hold hypervector state built once at construction and slot in as the first argument of `encode`, with `decode` as the inverse. `LevelEncoder` is the first — it encodes numeric values against a shared level set (see below); `RandomProjection` is the planned sibling.

Scalar `getindex` returns the element type `T`; non-scalar indexing returns a plain `Vector` of values, never a hypervector. Hypervectors are immutable (no `setindex!`).

## Core operations

| Operation | Function | Operator | Effect |
|-----------|----------|----------|--------|
| Bundle | `bundle(hvs)` | `+` | Superposition; result similar to all inputs |
| Bind | `bind(hv1, hv2)` | `*` | Association; result dissimilar to inputs |
| Unbind | `unbind(hv1, hv2)` | `/` | Inverse of bind (throws for `RealHV`: real MAP binding is not exactly invertible) |
| Shift | `shift(hv, k)` / `ρ(hv, k)` | — | Circular shift by k positions |
| Perturbate | `perturbate(hv, n_or_p)` | — | Flip n positions or fraction p |

In-place variants: `shift!`, `ρ!`, `perturbate!`.

## Encoding strategies (compose primitives into structured representations)

- `multiset(hvs)` — bundle a set of vectors
- `multibind(hvs)` — bind a set of vectors
- `bundlesequence(hvs)` / `bindsequence(hvs)` — ordered sequences via shift
- `hashtable(keys, values)` — key-value pairs via bind+bundle
- `crossproduct(U, V)` — cross product of two sets
- `ngrams(hvs, n)` — n-gram statistics for text/sequence encoding
- `graph(sources, targets)` — directed/undirected graph encoding

Numeric values are encoded with `LevelEncoder` (encode.jl), which replaced the old `level`/`encodelevel`/`decodelevel`/`convertlevel` function family. It builds its level set **once** at construction (the old family rebuilt it per call, producing incomparable encodings). Two mechanisms, selected by dispatch: a perturbation **ladder** for all types (`LevelEncoder(HV, range, n; bandwidth = 2/n)`, flip fraction per step) and **fractional power encoding** for FHRR (`LevelEncoder(FHRR, values; β)`, continuous via `base^(β·x)`; pass an explicit level count `n` to get a ladder instead). `encode(lvl, x)` maps a number to a hypervector; `decode(lvl, hv)` maps back via nearest-neighbour similarity (FHRR also supports `method = :analytic` for continuous, clean-vector decoding). `LevelEncoder(levels, values)` wraps precomputed hypervectors — this labelled-codebook form is the seam for a future shared item-memory abstraction.

## Similarity and inference

- `similarity(u, v)` / `δ(u, v)` — type-dispatched similarity (cosine for bipolar/real, Jaccard for binary/graded, complex dot for FHRR)
- `nearest_neighbor(u, collection)` — find closest match
- `nearest_neighbor(u, collection, k)` — k-nearest neighbors

## Dependencies

Core: Distributions, LinearAlgebra, Random
Optional: UnicodePlots (extension for REPL visualization)
Julia compat: >= 1.11

## Code style

Uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) formatter. Contributions follow GitHub Flow.

## Running tests

```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## Landscape: related packages

**Python**: Torchhd (PyTorch-based, GPU-accelerated, most popular), OpenHD, hdlib, HDTorch, PyBHV, NengoSPA (HRR-based)
**MATLAB**: VSA_Toolbox (TU Chemnitz)
**Java**: Semantic Vectors

## Key HDC/VSA concepts for contributors

- **Blessing of dimensionality**: random vectors in D~10,000 are quasi-orthogonal with high probability
- **Binding is self-inverse** for BSC/MAP (x * x = identity); FHRR unbinds exactly via elementwise complex division; RealHV binding is **not** exactly invertible (`unbind` throws)
- **Bundle preserves similarity** to inputs; bind produces dissimilar output
- **Shift encodes position** — used for sequences and n-grams
- All encodings are compositions of these three primitives
- The specific vector algebra matters less than the high dimensionality

---

## Issues and improvement backlog

### 1. Code style issues

- [x] ~~Typo "Represenetations" in FHRR comment and docstring~~ (fixed)
- [x] ~~Missing blank line between TernaryHV and BinaryHV sections~~ (fixed)
- [x] ~~TODO block references "complex HDC" which is now implemented (FHRR)~~ (fixed)
- [x] ~~BinaryHV docstring says "A ternary hypervector type"~~ (fixed)
- [x] ~~Stale comment about LazyArrays in `operations.jl`~~ (fixed)
- [x] ~~Typo "Measurures" in `isapprox` docstrings~~ (fixed)
- [x] ~~Typo "N_bootstap" in `isapprox` docstring~~ (fixed)
- [x] ~~`predictors.jl` contains Dutch comments and dead code~~ (deleted)
- [x] ~~Unused `counts` variable in `representations.jl` show method~~ (fixed)
- [x] ~~Unused `counts` variable in `ext/UnicodePlotting.jl` show method~~ (fixed)
- [x] ~~`docs/src/index.md` placeholder text "provides..."~~ (filled in)
- [x] ~~`docs/src/index.md` and README link to wrong repo owner~~ (fixed)
- [x] ~~`introduction-to-hdc.jl` uses undeclared `Handcalcs` dependency~~ (removed)
- [x] ~~Typos "constituyents" / "contituyent" in introduction tutorial~~ (fixed)
- [x] ~~`scripts/concept.jl` uses entirely obsolete API~~ (deleted)
- [x] ~~`test/benchmarking.jl` uses entirely obsolete API~~ (deleted)
- [x] ~~TernaryHV docstring says elements in `(-1, 1)` (open interval)~~ (fixed to `{-1, +1}`)
- [x] ~~Graph docstring uses `\otimes` for outer op instead of `\oplus`~~ (fixed)
- [x] ~~Typo "incoding" in `convertlevel` docstring~~ (fixed)

### 2. Missing documentation

- [x] ~~`AbstractHV`: no docstring on the abstract type itself~~ (added, documents the `HV(this; D, seed/rng)` constructor convention)
- [x] ~~`BipolarHV`: no docstring~~ (added)
- [x] ~~`RealHV`: no docstring~~ (added)
- [x] ~~`GradedHV`: no docstring~~ (added)
- [x] ~~`GradedBipolarHV`: no docstring~~ (added)
- [x] ~~`FHRR`: no docstring~~ (added)
- [x] ~~`bundle`: no docstring for any of the bundle methods~~ (added on the collection entry point)
- [x] ~~`bind`: no docstring (only `unbind` has one)~~ (added)
- [ ] `shift` / `ρ`: no docstrings
- [ ] `perturbate` / `perturbate!`: no docstrings
- [x] ~~`level`: docstring is minimal; does not explain the perturbation-based correlation mechanism~~ (obsolete: the family was replaced by `LevelEncoder`, whose docstring covers both mechanisms)
- [ ] `three_pi` and `fuzzy_xor` helper functions: no docstrings
- [ ] Internal functions (`aggfun`, `bindfun`, `neutralbind`, `noisy_and`, `elementreduce!`, `offsetcombine`, `empty_vector`, `eldist`, `vectype`): none documented
- [ ] `docs/make.jl`: Contents block references `examples.md` but actual pages are in `examples/` subdirectory
- [ ] No docstring for `^` on FHRR — exponentiation is an important FHRR feature
- [ ] No developer docs on how to add a new HV type (what methods to implement, traits, etc.)

### 3. Missing or incomplete features

- [x] ~~`learning.jl`: commented out, uses old API~~ (deleted)
- [x] ~~`predictors.jl`: uses undefined `cosine_dist`, empty Naive Bayes stub~~ (deleted)
- [x] ~~`Distances` dependency declared but never used~~ (removed)
- [x] ~~`convertlevel` passes `kwargs...` as positional to `decodelevel`~~ (fixed)
- [x] ~~`BipolarHV` seed type inconsistent (`Number` vs `Integer`)~~ (fixed)
- [x] ~~`GradedHV.similar` loses custom distribution~~ (fixed)
- [ ] `SparseHV`: listed as TODO in `types.jl` — never implemented
- [x] ~~`TernaryHV` constructor generates only ±1 — undocumented~~ (documented in docstring)
- [x] ~~No `setindex!` on `AbstractHV` — undocumented~~ (documented on `AbstractHV`)
- [ ] FHRR `unbind` uses element-wise `/` instead of idiomatic complex conjugate multiplication (works and is tested; conjugate variant is a possible refinement)
- [x] ~~`empty_vector` not defined for FHRR — `bundle` would error~~ (stale: the generic `zero(hv.v)` fallback covers FHRR; bundle works and is tested)
- [x] ~~`perturbate` not implemented for FHRR~~ (phase-resampling methods added)
- [x] ~~`graph` docstring has empty `# Example` section~~ (example added)
- [ ] No CI coverage analysis (`test.yml` has a TODO for this)
- [x] ~~`StatsBase` dependency unused in `src/`~~ (removed from deps; `mean`/`std` come from the Distributions re-export)
- [ ] No classification/learning workflow (`train`/`predict`) — HDC's main practical use case
- [ ] Package not registered in Julia General registry

### 4. Explicit TODOs in codebase

- [ ] `src/types.jl`: `TODO: SparseHV`
- [ ] `src/encoding.jl`: `# TODO: This should be bundled without normalizing` in `crossproduct`
- [ ] `.github/workflows/test.yml`: `# TODO: Add coverage analysis and style checker`
