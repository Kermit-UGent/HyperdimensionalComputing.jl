#=
encode.jl

The encoder layer: turning raw data into hypervectors.

Layer taxonomy of the package:
- primitives   (operations.jl): bundle, bind, shift, perturbate
- combinators  (encoding.jl):   multiset, ngrams, hashtable, ... — hypervectors
                                in, hypervector out
- encoders     (this file):     raw data in, hypervector out

`encode(HV, x)` is the canonical token path; sequence strategies compose the
token path with the existing combinators. Stateful encoders (random projection,
level encoders) are a planned follow-up as an `AbstractEncoder{HV}` hierarchy
whose instances will slot in as the first argument of `encode` — nothing here
may claim `encode(::SomeState, ...)` for other purposes.
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
