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
the inverse map `decode`). `LevelEncoder` is the first; `RandomProjection` is
the planned sibling — nothing here may claim `encode(::SomeState, ...)` for
other purposes.
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

Currently implemented: [`LevelEncoder`](@ref). A `RandomProjection` encoder is
the planned next sibling.
"""
abstract type AbstractEncoder{HV <: AbstractHV} end

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
    levels = [base^(β * x) for x in values]
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
    lvl.base === nothing || return lvl.base^(lvl.bandwidth * x)
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
