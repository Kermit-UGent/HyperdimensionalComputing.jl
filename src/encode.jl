#=
encode.jl

The encoder layer: turning raw data into hypervectors.

Layer taxonomy of the package:
- primitives   (operations.jl): bundle, bind, shift, perturbate
- combinators  (encoding.jl):   multiset, ngrams, hashtable, ... — hypervectors
                                in, hypervector out
- encoders     (this file):     raw data in, hypervector out

`encode(HV, x)` is the canonical token path; sequence strategies compose the
token path with the existing combinators. Stateful encoders live in the
`AbstractEncoder{HV}` hierarchy: their instances hold hypervector state built
once at construction and slot in as the first argument of `encode` (and support
the inverse map `decode`). `LevelEncoder` (scalars, shared level set) and
`RandomProjection` (feature vectors, shared projection matrix) are the current
members — nothing here may claim `encode(::SomeState, ...)` for other purposes.
=#

"""
    encode(HV::Type{<:AbstractHV}, x; D = 10_000, kwargs...)
    encode(HV::Type{<:AbstractHV}, x, strategy::AbstractEncoding; D = 10_000, kwargs...)

Encode raw data `x` as a `D`-dimensional hypervector of type `HV`.

Without a strategy, this is the **token path**: one object, one hash, one
hypervector. `x` is hashed and the hash seeds the hypervector, so encoding the
same object twice yields the same hypervector, and distinct objects give
quasi-orthogonal hypervectors. This holds for *any* `x`, including strings and
collections: `encode(HV, "ACGT")` hashes the whole string as a single token.

To encode a string (or any iterable of symbols) as a *sequence*, say so with a
strategy: [`KMer`](@ref), [`NGram`](@ref), [`Sequence`](@ref) or
[`BagOfSymbols`](@ref). Strategies are thin compositions of the token path with
the combinators in `encoding.jl` (`multiset`, `ngrams`, `bundlesequence`).

`HV(x)` is shorthand for `encode(HV, x)` for token-like `x`; every non-trivial
encoding goes through `encode`. Extra keyword arguments (`distr`, `T`, ...) are
forwarded to the `HV` constructor.

# Examples

```jldoctest
julia> encode(BipolarHV, "cat") == encode(BipolarHV, "cat")   # deterministic
true

julia> encode(BipolarHV, "cat") == BipolarHV("cat")           # HV(x) is sugar
true

julia> length(encode(BinaryHV, 42; D = 100))   # numbers are fine as encode tokens
100
```

Sequence strategies compose the token path with the combinators:

```jldoctest
julia> kmer = encode(BinaryHV, "ACGTAC", KMer(3); D = 64);

julia> kmer == multiset([encode(BinaryHV, s; D = 64) for s in ["ACG", "CGT", "GTA", "TAC"]])
true

julia> kmer != encode(BinaryHV, "ACGTAC", NGram(3); D = 64)   # a different encoding
true
```

# See also

[`AbstractEncoding`](@ref), [`KMer`](@ref), [`NGram`](@ref), [`Sequence`](@ref),
[`BagOfSymbols`](@ref)
"""
encode(HV::Type{<:AbstractHV}, x; kwargs...) = HV(; seed = hash(x), kwargs...)

"""
    AbstractEncoding

Supertype of sequence-encoding strategies for [`encode`](@ref):
[`KMer`](@ref), [`NGram`](@ref), [`Sequence`](@ref) and [`BagOfSymbols`](@ref).

# Extending

Adding a new strategy requires only a struct and one `encode` method:

    struct EveryOther <: AbstractEncoding end

    function HyperdimensionalComputing.encode(
            HV::Type{<:AbstractHV}, x, ::EveryOther; kwargs...
        )
        return multiset([encode(HV, s; kwargs...) for s in collect(x)[1:2:end]])
    end
"""
abstract type AbstractEncoding end

"""
    KMer(k)

Sequence-encoding strategy: slide a window of length `k` over the sequence,
treat every k-mer **substring as one atomic token**, hash it, and bundle the
results with [`multiset`](@ref). This is the standard genomics/text encoding
(k-mer profile) and resolves issue #53.

Not the same operation as [`NGram`](@ref): `KMer` hashes each window as a
whole, so `"AC"` and `"CA"` get unrelated hypervectors; `NGram` encodes the
*symbols* and composes windows by shift-binding, so windows that share symbols
share structure. The two produce different hypervectors with different
properties — pick deliberately.

# Examples

```jldoctest
julia> hv = encode(BinaryHV, "ACGT", KMer(2); D = 64);

julia> hv == multiset([encode(BinaryHV, s; D = 64) for s in ["AC", "CG", "GT"]])
true
```
"""
struct KMer <: AbstractEncoding
    k::Int
    function KMer(k::Integer)
        k ≥ 1 || throw(ArgumentError("k-mer length must be ≥ 1, got $k"))
        return new(k)
    end
end

"""
    NGram(n)

Sequence-encoding strategy: encode each **symbol** to a hypervector, compose
every window of `n` consecutive symbols by shift-binding, and bundle the
windows — i.e. the existing [`ngrams`](@ref) combinator applied to
token-encoded symbols. See [`KMer`](@ref) for how this differs from k-mer
hashing.

# Examples

```jldoctest
julia> hv = encode(BinaryHV, "ACGT", NGram(2); D = 64);

julia> hv == ngrams([encode(BinaryHV, c; D = 64) for c in "ACGT"], 2)
true
```
"""
struct NGram <: AbstractEncoding
    n::Int
    function NGram(n::Integer)
        n ≥ 1 || throw(ArgumentError("n-gram size must be ≥ 1, got $n"))
        return new(n)
    end
end

"""
    Sequence()

Sequence-encoding strategy: encode each symbol to a hypervector and superpose
them position-aware via [`bundlesequence`](@ref) (symbol `i` is shifted `i - 1`
times). Similar sequences map to similar hypervectors; use [`KMer`](@ref) or
[`NGram`](@ref) for local-window statistics instead.
"""
struct Sequence <: AbstractEncoding end

"""
    BagOfSymbols()

Sequence-encoding strategy: encode each symbol to a hypervector and bundle them
orderlessly with [`multiset`](@ref) — the position-free counterpart of
[`Sequence`](@ref) (and the `n = 1` corner of [`NGram`](@ref)).
"""
struct BagOfSymbols <: AbstractEncoding end

# The window tokens: SubStrings for strings (hash-compatible with the equal
# String, and unicode-safe), tuples of symbols for everything else.
function windows(s::AbstractString, k::Int)
    idx = collect(eachindex(s))
    n = length(idx)
    k ≤ n || throw(ArgumentError("cannot take $k-mers of a sequence of length $n"))
    return [SubString(s, idx[i], idx[i + k - 1]) for i in 1:(n - k + 1)]
end

function windows(x, k::Int)
    v = collect(x)
    n = length(v)
    k ≤ n || throw(ArgumentError("cannot take $k-mers of a sequence of length $n"))
    return [Tuple(v[i:(i + k - 1)]) for i in 1:(n - k + 1)]
end

function encode(HV::Type{<:AbstractHV}, x, enc::KMer; kwargs...)
    return multiset([encode(HV, w; kwargs...) for w in windows(x, enc.k)])
end

function encode(HV::Type{<:AbstractHV}, x, enc::NGram; kwargs...)
    return ngrams([encode(HV, s; kwargs...) for s in collect(x)], enc.n)
end

function encode(HV::Type{<:AbstractHV}, x, ::Sequence; kwargs...)
    return bundlesequence([encode(HV, s; kwargs...) for s in collect(x)])
end

function encode(HV::Type{<:AbstractHV}, x, ::BagOfSymbols; kwargs...)
    return multiset([encode(HV, s; kwargs...) for s in collect(x)])
end

# Stateful encoders
# -----------------

"""
    AbstractEncoder{HV <: AbstractHV}

Supertype of **stateful encoders**: objects that hold hypervector state built
once at construction (a level set, a projection matrix, ...) and are then used
through [`encode`](@ref) and its inverse [`decode`](@ref). Where an
[`AbstractEncoding`](@ref) strategy is a pure recipe (`encode(HV, x, strategy)`),
an `AbstractEncoder` instance *is* the shared state: encoding a value and
decoding a hypervector are only consistent against the same state, so the state
lives in the object, not in the call. Instances slot in as the first argument
of `encode`.

Currently implemented: [`LevelEncoder`](@ref) (scalars) and
[`RandomProjection`](@ref) (feature vectors).
"""
abstract type AbstractEncoder{HV <: AbstractHV} end

"""
    phase_encode(z::AbstractVector{<:Real}, β::Real = 1.0)

Map real scores `z` to an [`FHRR`](@ref) phasor hypervector `exp.(im * β * z)`,
with `β` a bandwidth: the shared phase-encoding primitive behind
[`LevelEncoder`](@ref)'s fractional power path (`z` = the base's phases, scaled
by the encoded value) and [`RandomProjection`](@ref)'s FHRR nonlinearity
(`z = R * x`, random Fourier features). Internal — both encoders must route
FHRR phase encoding through this one function.
"""
phase_encode(z::AbstractVector{<:Real}, β::Real = 1.0) = FHRR(cis.(β .* z))

"""
    LevelEncoder{HV} <: AbstractEncoder{HV}

Encoder for numeric values: maps a number to a hypervector such that **close
values get similar hypervectors** and distant values get quasi-orthogonal ones,
via a set of level-correlated hypervectors built once at construction and
shared by every [`encode`](@ref)/[`decode`](@ref) call.

Two mechanisms, selected by dispatch on the hypervector type:

- **Ladder** (every hypervector type): start from a random base hypervector and
  repeatedly [`perturbate`](@ref) a fraction `bandwidth` of the positions to
  obtain successive levels. Adjacent levels are similar; the first and last are
  quasi-orthogonal. Values are quantized to the nearest level.
- **Fractional power encoding** ([`FHRR`](@ref) only, the default for `FHRR`):
  `encode(lvl, x) = base^(β * x)` using complex exponentiation — continuous, no
  quantization. `β` plays the same bandwidth role as the ladder's flip
  fraction: smaller values give slower similarity decay.

# Constructors

    LevelEncoder(HV, values;   D = 10_000, bandwidth = 2/length(values), seed, rng)
    LevelEncoder(HV, range, n; D = 10_000, bandwidth = 2/n, seed, rng)
    LevelEncoder(FHRR, values; D = 10_000, β = 1/(max - min), seed, rng)
    LevelEncoder(levels::AbstractVector{<:AbstractHV}, values)

The first two build a ladder: over the given `values` (any vector or range of
numbers), or `n` evenly spaced levels over `range` (anything `minimum`/`maximum`
accept: a range, a vector, a 2-tuple). For `FHRR` the two-argument form builds a
fractional power encoder whose `values` serve as the decoding grid; ask for a
ladder explicitly by passing a level count `n`. The last form wraps precomputed
level hypervectors with their values. Passing `seed` makes the whole encoder
deterministic.

# Examples

```jldoctest levelencoder
julia> lvl = LevelEncoder(BipolarHV, (0, 1), 20; seed = 42)
LevelEncoder{BipolarHV}: 20 levels over [0.0, 1.0] (ladder, bandwidth = 0.1)

julia> decode(lvl, encode(lvl, 0.25))   # round-trips within one grid step
0.2631578947368421

julia> similarity(encode(lvl, 0.3), encode(lvl, 0.35)) >
           similarity(encode(lvl, 0.3), encode(lvl, 0.9))
true
```

Fractional power encoding for `FHRR` is continuous:

```jldoctest levelencoder
julia> fpe = LevelEncoder(FHRR, 0:0.1:10; seed = 1)
LevelEncoder{FHRR}: 101 levels over [0.0, 10.0] (fractional power, β = 0.1)

julia> decode(fpe, encode(fpe, 3.7); method = :analytic) ≈ 3.7
true
```

# See also

[`encode`](@ref), [`decode`](@ref), [`AbstractEncoder`](@ref),
[`perturbate`](@ref)
"""
struct LevelEncoder{HV <: AbstractHV, V <: AbstractVector{<:Real}, B <: Union{AbstractHV, Nothing}} <: AbstractEncoder{HV}
    levels::Vector{HV}
    values::V
    base::B             # FPE base hypervector; `nothing` for ladder/precomputed
    bandwidth::Float64  # ladder: flip fraction per step; FPE: β; NaN: precomputed
    function LevelEncoder(levels::Vector{HV}, values::V, base::B, bandwidth::Real) where {HV <: AbstractHV, V <: AbstractVector{<:Real}, B <: Union{AbstractHV, Nothing}}
        length(levels) == length(values) ||
            throw(ArgumentError("number of levels ($(length(levels))) must match number of values ($(length(values)))"))
        isempty(levels) && throw(ArgumentError("a LevelEncoder needs at least one level"))
        return new{HV, V, B}(levels, values, base, Float64(bandwidth))
    end
end

LevelEncoder(levels::AbstractVector{<:AbstractHV}, values::AbstractVector{<:Real}) =
    LevelEncoder(collect(levels), values, nothing, NaN)

# The ladder builder itself, shared by the constructor methods below: the
# two-argument FHRR method dispatches to fractional power encoding instead, so
# the explicit-level-count form must reach the ladder without re-dispatching.
function ladder(
        HV::Type{<:AbstractHV}, values::AbstractVector{<:Real};
        D::Int = 10_000, bandwidth::Real = 2 / length(values),
        seed = nothing, rng::AbstractRNG = Random.default_rng()
    )
    n = length(values)
    n ≥ 2 || throw(ArgumentError("a level encoding needs at least 2 levels, got $n"))
    0 < bandwidth ≤ 1 ||
        throw(ArgumentError("bandwidth must be a flip fraction in (0, 1], got $bandwidth"))
    rng = seed === nothing ? rng : Xoshiro(seed)
    levels = [HV(; D, rng)]
    while length(levels) < n
        push!(levels, perturbate(last(levels), float(bandwidth); rng))
    end
    return LevelEncoder(levels, values, nothing, bandwidth)
end

LevelEncoder(HV::Type{<:AbstractHV}, values::AbstractVector{<:Real}; kwargs...) =
    ladder(HV, values; kwargs...)

LevelEncoder(HV::Type{<:AbstractHV}, range, n::Integer; kwargs...) =
    ladder(HV, Base.range(minimum(range), maximum(range), n); kwargs...)

function LevelEncoder(
        F::Type{<:FHRR}, values::AbstractVector{<:Real};
        β::Real = 1 / (maximum(values) - minimum(values)),
        D::Int = 10_000, seed = nothing, rng::AbstractRNG = Random.default_rng()
    )
    length(values) ≥ 2 ||
        throw(ArgumentError("a level encoding needs at least 2 values, got $(length(values))"))
    rng = seed === nothing ? rng : Xoshiro(seed)
    base = F(; D, rng)
    phases = angle.(base.v)
    levels = [phase_encode(phases, β * x) for x in values]
    return LevelEncoder(levels, values, base, β)
end

"""
    encode(lvl::LevelEncoder, x::Number; testbound = false)

Encode the number `x` as a hypervector using the encoder's shared level set:
the level whose value is nearest to `x` for a ladder encoder, or the continuous
`base^(β * x)` for a fractional power ([`FHRR`](@ref)) encoder. Out-of-range
values snap to the nearest level by default; pass `testbound = true` to throw a
`DomainError` instead. See [`LevelEncoder`](@ref).
"""
function encode(lvl::LevelEncoder, x::Number; testbound::Bool = false)
    if testbound
        lo, hi = extrema(lvl.values)
        lo ≤ x ≤ hi || throw(DomainError(x, "value outside the encoder's range [$lo, $hi]"))
    end
    lvl.base === nothing || return phase_encode(angle.(lvl.base.v), lvl.bandwidth * x)
    (_, ind) = findmin(v -> abs(x - v), lvl.values)
    return lvl.levels[ind]
end

"""
    decode(lvl::LevelEncoder, hv::AbstractHV; method = :nearest)

Decode a hypervector back to the numeric value of the most similar level in the
encoder's shared level set (nearest-neighbour by [`similarity`](@ref) — robust,
works for noisy hypervectors such as bundles, but quantized to the encoder's
value grid).

For a fractional power ([`FHRR`](@ref)) encoder, `method = :analytic` instead
inverts the phases directly (`mean(real(log(hv) / log(base)) / β)`), giving a
continuous, grid-free estimate — but it assumes a *clean* encoded vector and
degrades on noisy ones; the default `:nearest` is the robust choice. See
[`LevelEncoder`](@ref).
"""
function decode(lvl::LevelEncoder, hv::AbstractHV; method::Symbol = :nearest)
    if method === :nearest
        (_, ind) = findmax(v -> similarity(v, hv), lvl.levels)
        return lvl.values[ind]
    elseif method === :analytic
        lvl.base === nothing && throw(
            ArgumentError(
                "analytic decoding requires a fractional power (FHRR) encoder; " *
                    "this encoder has no base hypervector — use `method = :nearest`"
            )
        )
        return mean(@. real(log(hv.v) / log(lvl.base.v)) / lvl.bandwidth)
    else
        throw(ArgumentError("unknown decoding method $(repr(method)); expected :nearest or :analytic"))
    end
end

function Base.show(io::IO, lvl::LevelEncoder{HV}) where {HV}
    lo, hi = extrema(lvl.values)
    mechanism = if lvl.base !== nothing
        "fractional power, β = $(lvl.bandwidth)"
    elseif isnan(lvl.bandwidth)
        "precomputed levels"
    else
        "ladder, bandwidth = $(lvl.bandwidth)"
    end
    return print(io, "LevelEncoder{$(nameof(HV))}: $(length(lvl.levels)) levels over [$lo, $hi] ($mechanism)")
end

"""
    RandomProjection{HV, T} <: AbstractEncoder{HV}

Encoder for **fixed-length real feature vectors** (an RGB triple, an
embedding): `x ∈ ℝᵈ` is projected as `z = R * x` through a `D × d` projection
matrix drawn once at construction, then mapped into the hypervector domain by a
per-type nonlinearity. Nearby feature vectors get similar hypervectors; the
Johnson–Lindenstrauss lemma guarantees the projection approximately preserves
distances (Kleyko et al., §3.2.3). Not for sequences (use [`KMer`](@ref) /
[`NGram`](@ref)) and not for scalars (use [`LevelEncoder`](@ref)).

`R` is the shared state: two hypervectors are comparable **only** when encoded
through the same `R`, which is why this is a stateful [`AbstractEncoder`](@ref)
— exactly like `LevelEncoder`'s level set. The struct is immutable and has no
`fit` method; re-thresholding returns a new encoder via [`rethreshold`](@ref).

!!! warning "Standardize your features first"
    `R * x` is dominated by whichever feature has the largest scale. RGB
    channels share a scale; embedding dimensions and mixed tabular features
    generally do not. Center and scale each feature (z-scores) before
    encoding, or the projection silently encodes only the loudest feature.

# Nonlinearities (dispatched on the output type)

| Output type | Map from `z = R * x` |
|:------------|:---------------------|
| `BipolarHV` | `z > 0 ↦ +1`, else `-1` |
| `BinaryHV` | `z .> 0` |
| `TernaryHV` | `sign.(z) .* (abs.(z) .> θ)` (`θ` scalar or per-component vector) |
| `RealHV` | `β .* z` |
| `GradedHV` | `logistic.(β .* z)` |
| `GradedBipolarHV` | `tanh.(β .* z)` |
| `FHRR` | `exp.(im * β .* z)` (shared `phase_encode` helper) |

The sign-like maps are scale-invariant; for the others `β` (default `1/√d`)
scales `z` to order one for standardized features, and can be tuned.

# FHRR = random Fourier features

For `FHRR` with a `:gaussian` matrix this is exactly the Rahimi–Recht random
Fourier feature map: `similarity(encode(rp, x), encode(rp, y))` approximates
the Gaussian kernel `exp(-β² ‖x - y‖² / 2)` with bandwidth `β` — the same
phase-encoding math as `LevelEncoder`'s fractional power path, and the
strongest bridge between hyperdimensional computing and kernel methods.

# Constructors

    RandomProjection(HV, d::Int; D = 10_000, matrix = :gaussian, θ = 0, β = 1/√d, seed, rng)
    RandomProjection(HV, R::AbstractMatrix; θ = 0, β = 1/√size(R, 2))
    RandomProjection(TernaryHV, X::AbstractMatrix; target_sparsity, D = 10_000, matrix, seed, rng)

The first draws a fresh `D × d` matrix: `:gaussian` (`N(0,1)`, default),
`:bipolar` (`{-1, +1}`) or `:sparse_ternary` (`{-1, 0, +1}` with nonzero
density `1/√d`, Li–Hastie–Church very sparse projections — the HDC-idiomatic,
scalable choice). All three satisfy Johnson–Lindenstrauss.

The second wraps a **supplied** matrix (`d` and `D` are read off its size), for
reproducible, saved or structured projections.

The third is the data-driven ternary constructor: it reads the `d × n` data
matrix `X` (columns are observations) once at construction and solves for the
global scalar `θ` such that encoded training columns have a fraction
`target_sparsity` of zero elements. This is construction from data, not
fitting: the returned encoder is as immutable as any other. (For `TernaryHV`
the positional matrix is interpreted as data `X` when `target_sparsity` is
given, and as a supplied projection matrix otherwise.)

# Examples

```jldoctest randomprojection
julia> rp = RandomProjection(BipolarHV, 3; seed = 42)
RandomProjection{BipolarHV}: 3 features → 10000-dimensional BipolarHV

julia> x = [0.9, -0.2, 0.4];

julia> encode(rp, x) == encode(rp, x)   # same R, comparable and deterministic
true

julia> similarity(encode(rp, x), encode(rp, x .+ 0.05)) >
           similarity(encode(rp, x), encode(rp, -x))
true
```

The FHRR similarity approximates a Gaussian kernel:

```jldoctest randomprojection
julia> rff = RandomProjection(FHRR, 3; β = 0.5, seed = 1)
RandomProjection{FHRR}: 3 features → 10000-dimensional FHRR (β = 0.5)

julia> y = [0.7, -0.2, 0.4];   # ‖x - y‖ = 0.2

julia> similarity(encode(rff, x), encode(rff, y)) ≈ exp(-(0.5 * 0.2)^2 / 2) atol = 0.02
ERROR: ParseError:
# Error @ none:1:69
similarity(encode(rff, x), encode(rff, y)) ≈ exp(-(0.5 * 0.2)^2 / 2) atol = 0.02
#                                                                   └──────────┘ ── extra tokens after end of expression
Stacktrace:
 [1] top-level scope
   @ none:1
```

# See also

[`encode`](@ref), [`decode`](@ref), [`rethreshold`](@ref),
[`AbstractEncoder`](@ref), [`LevelEncoder`](@ref)
"""
struct RandomProjection{HV <: AbstractHV, T} <: AbstractEncoder{HV}
    R::Matrix{Float64}   # D × d, drawn (or supplied) once
    θ::T                 # ternary threshold; scalar or per-component vector
    bandwidth::Float64   # scale β for RealHV/graded/FHRR (RFF bandwidth)
    function RandomProjection{HV}(R::Matrix{Float64}, θ::T, bandwidth::Real) where {HV <: AbstractHV, T <: Union{Real, AbstractVector{<:Real}}}
        θ isa AbstractVector && length(θ) != size(R, 1) && throw(
            ArgumentError("per-component θ must have length D = $(size(R, 1)), got $(length(θ))")
        )
        bandwidth > 0 || throw(ArgumentError("β must be positive, got $bandwidth"))
        return new{HV, T}(R, θ, Float64(bandwidth))
    end
end

# The projection matrix distributions; all Johnson–Lindenstrauss-guaranteed.
function projection_matrix(
        matrix::Symbol, D::Integer, d::Integer;
        rng::AbstractRNG = Random.default_rng()
    )
    d ≥ 1 || throw(ArgumentError("feature dimension d must be ≥ 1, got $d"))
    D ≥ 1 || throw(ArgumentError("hypervector dimension D must be ≥ 1, got $D"))
    return if matrix === :gaussian
        randn(rng, D, d)
    elseif matrix === :bipolar
        rand(rng, (-1.0, 1.0), D, d)
    elseif matrix === :sparse_ternary
        # very sparse random projections (Li, Hastie & Church 2006):
        # ±1 with probability 1/(2√d) each, 0 otherwise
        p = 1 / (2 * sqrt(d))
        u = rand(rng, D, d)
        @. ifelse(u < p, 1.0, ifelse(u > 1 - p, -1.0, 0.0))
    else
        throw(
            ArgumentError(
                "unknown projection matrix distribution $(repr(matrix)); " *
                    "expected :gaussian, :bipolar or :sparse_ternary"
            )
        )
    end
end

function RandomProjection(
        HV::Type{<:AbstractHV}, d::Integer;
        D::Integer = 10_000, matrix::Symbol = :gaussian,
        θ = 0.0, β::Real = 1 / sqrt(d),
        seed = nothing, rng::AbstractRNG = Random.default_rng()
    )
    rng = seed === nothing ? rng : Xoshiro(seed)
    return RandomProjection{HV}(projection_matrix(matrix, D, d; rng), θ, β)
end

# Supplied-matrix path, shared between the generic method and the ternary
# method below (which cannot reach the generic one: dispatch is positional).
supplied_projection(HV::Type{<:AbstractHV}, R::AbstractMatrix{<:Real}, θ, β) =
    RandomProjection{HV}(Matrix{Float64}(R), θ, something(β, 1 / sqrt(size(R, 2))))

RandomProjection(HV::Type{<:AbstractHV}, R::AbstractMatrix{<:Real}; θ = 0.0, β = nothing) =
    supplied_projection(HV, R, θ, β)

function RandomProjection(
        HV::Type{<:TernaryHV}, X::AbstractMatrix{<:Real};
        target_sparsity = nothing, θ = 0.0, β = nothing,
        D::Integer = 10_000, matrix::Symbol = :gaussian,
        seed = nothing, rng::AbstractRNG = Random.default_rng()
    )
    # without a target sparsity, the matrix is a supplied projection matrix
    target_sparsity === nothing && return supplied_projection(HV, X, θ, β)
    0 < target_sparsity < 1 ||
        throw(ArgumentError("target_sparsity must lie in (0, 1), got $target_sparsity"))
    rng = seed === nothing ? rng : Xoshiro(seed)
    d = size(X, 1)
    R = projection_matrix(matrix, D, d; rng)
    # the global threshold zeroing exactly a target_sparsity fraction of the
    # projected training data
    θ_data = quantile(vec(abs.(R * X)), target_sparsity)
    return RandomProjection{HV}(R, θ_data, something(β, 1 / sqrt(d)))
end

"""
    rethreshold(rp::RandomProjection, θ)

Return a new [`RandomProjection`](@ref) with threshold `θ` (a scalar or a
length-`D` vector), **sharing** the original's projection matrix — encodings
from both encoders remain comparable. This is the immutable counterpart of
mutating the threshold.
"""
rethreshold(rp::RandomProjection{HV}, θ) where {HV} =
    RandomProjection{HV}(rp.R, θ, rp.bandwidth)

# The per-type nonlinearities mapping the projection z = R * x into each
# hypervector domain. sign-like maps are scale-invariant; the rest scale by
# the encoder's bandwidth β. `z > 0 ↦ +1, else -1` (not `sign`): a bipolar
# element has no zero state, and Bool vectors would be read as raw bits.
nonlinearity(::Type{<:BipolarHV}, z, rp) = BipolarHV(ifelse.(z .> 0, 1, -1))
nonlinearity(::Type{<:BinaryHV}, z, rp) = BinaryHV(z .> 0)
nonlinearity(::Type{<:TernaryHV}, z, rp) = TernaryHV(sign.(z) .* (abs.(z) .> rp.θ))
nonlinearity(::Type{<:RealHV}, z, rp) = RealHV(rp.bandwidth .* z)
nonlinearity(::Type{<:GradedHV}, z, rp) = GradedHV(@. 1 / (1 + exp(-rp.bandwidth * z)))
nonlinearity(::Type{<:GradedBipolarHV}, z, rp) = GradedBipolarHV(tanh.(rp.bandwidth .* z))
nonlinearity(::Type{<:FHRR}, z, rp) = phase_encode(z, rp.bandwidth)

"""
    encode(rp::RandomProjection, x::AbstractVector{<:Real})
    encode(rp::RandomProjection, X::AbstractMatrix{<:Real})

Encode the feature vector `x` (length `d`) as a hypervector: project through
the encoder's fixed matrix (`z = R * x`) and apply the output type's
nonlinearity. A `d × n` matrix is encoded per column, returning a vector of
`n` hypervectors — all comparable, having passed through the same `R`. See
[`RandomProjection`](@ref).
"""
function encode(rp::RandomProjection{HV}, x::AbstractVector{<:Real}) where {HV}
    d = size(rp.R, 2)
    length(x) == d ||
        throw(DimensionMismatch("expected a feature vector of length $d, got $(length(x))"))
    return nonlinearity(HV, rp.R * x, rp)
end

encode(rp::RandomProjection, X::AbstractMatrix{<:Real}) = [encode(rp, x) for x in eachcol(X)]

"""
    decode(rp::RandomProjection, hv::AbstractHV, references; method = :nearest)

Clean up `hv` against a set of `references` via
[`nearest_neighbor`](@ref) — returning its `(similarity, index, neighbor)`
(index becomes the key when `references` is a `Dict`).

This is **clean-up, not inversion**: the projection nonlinearity (sign,
threshold, tanh, ...) discards magnitudes, so a random projection has no
analytic inverse and `method = :analytic` is deliberately not offered (unlike
[`LevelEncoder`](@ref) for FHRR). Calling `decode` without `references` throws:
recovering anything from a lossy encoding needs a codebook to search.
"""
function decode(rp::RandomProjection, hv::AbstractHV, references; method::Symbol = :nearest)
    method === :nearest || throw(
        ArgumentError(
            method === :analytic ?
                "random projections have no analytic inverse: the nonlinearity " *
                "(sign, threshold, tanh, ...) discards magnitudes, so encoding is " *
                "lossy. Only `method = :nearest` clean-up is available." :
                "unknown decoding method $(repr(method)); expected :nearest"
        )
    )
    return nearest_neighbor(hv, references)
end

decode(rp::RandomProjection, hv::AbstractHV; kwargs...) = throw(
    ArgumentError(
        "a random projection is lossy and cannot be inverted; decoding is " *
            "nearest-neighbour clean-up against a codebook. Supply a reference " *
            "set: `decode(rp, hv, references)`."
    )
)

function Base.show(io::IO, rp::RandomProjection{HV}) where {HV}
    D, d = size(rp.R)
    detail = if HV <: TernaryHV
        " (θ = $(rp.θ isa Real ? rp.θ : "per-component"))"
    elseif HV <: FHRR
        " (β = $(rp.bandwidth))"
    elseif HV <: Union{RealHV, GradedHV, GradedBipolarHV}
        " (scale β = $(rp.bandwidth))"
    else
        ""
    end
    return print(io, "RandomProjection{$(nameof(HV))}: $d features → $D-dimensional $(nameof(HV))$detail")
end
