#=
types.jl

Implements the basic types for the different hypervectors (wrappers for ordinary vectors)

Contains:
- AbstractHV
- BinaryHV
- BipolarHV
- TernaryHV
- RealHV
- GradedHV
- GradedBipolarHV

Every hypervector HV has the following basic functionality
- random generation using the Constructor ()
- norm/sum/normalize...

TODO:
- [ ] SparseHV
=#

# ----------------------------------------------------------------------------------- AbstractHV
"""
    AbstractHV{T} <: AbstractVector{T}

Abstract supertype of all hypervector types: [`BinaryHV`](@ref), [`BipolarHV`](@ref),
[`TernaryHV`](@ref), [`RealHV`](@ref), [`GradedHV`](@ref), [`GradedBipolarHV`](@ref)
and [`FHRR`](@ref).

A hypervector is a high-dimensional vector (10,000 dimensions by default) that
carries information holographically: meaning is distributed over the whole vector
rather than located in individual elements. Hypervectors are composed with
[`bundle`](@ref) (`+`), [`bind`](@ref) (`*`) and `shift` (`ρ`), and compared with
[`similarity`](@ref).

# Constructors and `encode`

All concrete hypervector types `HV <: AbstractHV` share the same constructor
interface, and each constructor form has exactly one meaning:

    HV(; D = 10_000, seed = nothing, rng = default_rng())   # fresh random hypervector
    HV(v::AbstractVector{<:Real})                           # wrap element data, validated per type
    HV(x)                                                   # token shorthand for `encode(HV, x)`

[`encode`](@ref) is the canonical way to turn raw data into hypervectors — `HV(x)`
is shorthand for its token path only. `HV(n::Number)` throws, because a number is
ambiguous between a token and the dimensionality (use `D = n`, or `encode(HV, n)`
for number tokens); an array that is not valid element data for the type throws
instead of silently token-encoding. Tuples of reals read as data, like vectors.

Some types extend this interface with type-specific keywords, e.g. `distr` for
[`RealHV`](@ref), [`GradedHV`](@ref) and [`GradedBipolarHV`](@ref).

# Indexing

`hv[i]` with an integer returns the element value. Non-scalar indexing —
`hv[1:3]`, `hv[[1, 4]]`, logical masks — returns a plain `Vector` of element
values, *not* a new hypervector: information in a hypervector is distributed over
all `D` dimensions, so a slice is not itself a meaningful hypervector.
Hypervectors are immutable; there is no `setindex!`. Equality (`==`/`isequal`)
holds only between hypervectors of the same type — a `BinaryHV` never equals a
`BipolarHV` — while comparison against plain vectors is elementwise.

# Examples

```jldoctest
julia> BinaryHV(:cat) == BinaryHV(:cat)   # encoding the same object twice
true

julia> BinaryHV(:cat) == BinaryHV(:dog)   # different objects, different vectors
false

julia> length(BinaryHV(:cat; D = 100))    # dimensionality is set with the keyword D
100
```

# See also

[`encode`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)

# Extended help

## References

- Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors. Cognitive Computation, 1(2), 139–159.
"""
abstract type AbstractHV{T} <: AbstractVector{T} end

Base.copy(hv::HV) where {HV <: AbstractHV} = HV(copy(hv.v))
# Scalar indexing returns the element value; non-scalar indexing (ranges, index
# vectors, logical masks) returns a plain Vector of element values, NOT a new
# hypervector — a slice of a hypervector is not a meaningful hypervector.
Base.getindex(hv::AbstractHV, i::Integer) = hv.v[i]
Base.getindex(hv::AbstractHV, I::AbstractVector) = hv.v[I]
Base.similar(hv::HV) where {HV <: AbstractHV} = HV(; D = length(hv))
Base.size(hv::AbstractHV) = size(hv.v)
Base.sum(hv::AbstractHV) = sum(hv.v)

LinearAlgebra.norm(hv::AbstractHV) = norm(hv.v)

"""
    normalize!(hv::AbstractHV)

Normalize the input hypervector.

# Examples

```julia-repl
julia> v = bundle([encode(TernaryHV, i; D = 8) for i in 1:100])
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  12
  -2
 -16
  12
  -4
   6
  12
 -12

julia> normalize!(v)
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  1
 -1
 -1
  1
 -1
  1
  1
 -1

julia> v
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  1
 -1
 -1
  1
 -1
  1
  1
 -1
"""
normalize!(hv::AbstractHV) = hv

"""
    normalize(hv::AbstractHV)

Return a copy of the normalized version of the input hypervector.

# Example

```julia-repl
julia> v = bundle([encode(TernaryHV, i; D = 8) for i in 1:100])
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  12
  -2
 -16
  12
  -4
   6
  12
 -12

julia> normalize(v)
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  1
 -1
 -1
  1
 -1
  1
  1
 -1

julia> v
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  12
  -2
 -16
  12
  -4
   6
  12
 -12
```
"""
normalize(hv::AbstractHV) = (c = copy(hv); normalize!(c); c)

# Types that carry an element distribution (`distr` field) override the
# instance method so that e.g. `perturbate` resamples from the vector's own
# distribution, not the type default.
eldist(hv::AbstractHV) = eldist(typeof(hv))
empty_vector(hv::AbstractHV) = zero(hv.v)

# Constructor sugar and guardrails. `HV(x)` is shorthand for `encode(HV, x)` —
# the token path. A `Number` is irreducibly ambiguous with the dimensionality
# and throws. An array that reaches the generic method is not valid element
# data for the type and throws rather than silently token-encoding
# (constructing from an array always means data; use `encode` for array
# tokens). A tuple of reals reads as data, like a vector.
(HV::Type{<:AbstractHV})(x; kwargs...) = encode(HV, x; kwargs...)
(HV::Type{<:AbstractHV})(t::Tuple{Vararg{Real}}; kwargs...) = HV(collect(t); kwargs...)
function (HV::Type{<:AbstractHV})(n::Number; kwargs...)
    throw(
        ArgumentError(
            "`$HV($n)` is ambiguous. For a $n-dimensional random hypervector, use " *
                "`$HV(; D = $n)`. To encode the number $n as a token, use `encode($HV, $n)`."
        )
    )
end
function (HV::Type{<:AbstractHV})(v::AbstractArray; kwargs...)
    throw(
        ArgumentError(
            "`$HV` cannot be constructed from a $(typeof(v)): it is not valid element " *
                "data for this type. Use `encode($HV, x)` to token-encode arbitrary objects."
        )
    )
end


# ------------------------------------------------------------------------------------ BipolarHV
"""
    BipolarHV(; D = 10_000, seed = nothing, rng = default_rng())
    BipolarHV(x)
    BipolarHV(v::AbstractVector{<:Real})
    BipolarHV(v::AbstractVector{Bool})

A bipolar hypervector in the style of the Multiply-Add-Permute (MAP) vector symbolic
architecture (Gayler, 1998). Elements are `±1`, stored compactly as a `BitVector`
with bit `true ↦ -1` and `false ↦ +1`, so that XOR on the stored bits is exactly
the elementwise `±1` product.

Constructing from a real vector requires every element to be exactly `±1`; a zero
element throws an `ArgumentError`, since a bipolar hypervector has no zero state —
use [`TernaryHV`](@ref) for elements in `{-1, 0, +1}` — and anything else is
rejected rather than coerced. A `Bool` vector is the exception: it is interpreted
as the **raw stored bits** (`true ↦ -1`), not as values.

Under this architecture, `bind` is the elementwise product (XOR on the stored
bits) and self-inverse — `x * x` is the all-`+1` identity — `bundle` is a
majority vote across inputs with deterministic tie-breaking, and `similarity`
defaults to cosine.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> BipolarHV(; D = 8, rng = Xoshiro(42))
8-element BipolarHV with 5 positives and 3 negatives:
  1
  1
  1
  1
 -1
 -1
 -1
  1

julia> BipolarHV("apple") == BipolarHV("apple")   # deterministic from hash
true
```

Random hypervectors are quasi-orthogonal at the default `D = 10_000`:

```jldoctest
julia> x = BipolarHV(; rng = Xoshiro(1)); y = BipolarHV(; rng = Xoshiro(2));

julia> similarity(x, y)
-0.0028
```

# See also

[`AbstractHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)

# Extended help

## References

- Gayler, R. W. (1998). Multiplicative Binding, Representation Operators & Analogy. In Advances in Analogy Research: Integration of Theory and Data from the Cognitive, Computational, and Neural Sciences, 1–4.
"""
struct BipolarHV <: AbstractHV{Int}
    v::BitVector
    BipolarHV(v::AbstractVector{Bool}) = new(v)
end

# Outer constructors
function BipolarHV(;
        D::Integer = 10_000,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return BipolarHV(bitrand(rng_instance, D))
end

# Data constructor: elements must be exactly ±1. Zero has no bipolar state and
# points to TernaryHV; anything else is rejected rather than coerced (silent
# sign-taking is how the polarity bug family started). Note that Bool vectors
# do NOT take this path: they hit the inner constructor and are treated as the
# raw stored bits.
function BipolarHV(v::AbstractVector{<:Real})
    any(iszero, v) && throw(
        ArgumentError(
            "bipolar hypervectors have no zero state; a zero element cannot be " *
                "mapped to ±1. Use `TernaryHV` for hypervectors with elements in {-1, 0, +1}."
        )
    )
    all(x -> x == 1 || x == -1, v) || throw(
        ArgumentError("BipolarHV elements must be -1 or +1")
    )
    return BipolarHV(v .< 0)
end

# Helpers
Base.getindex(hv::BipolarHV, i::Integer) = hv.v[i] ? -1 : 1
Base.getindex(hv::BipolarHV, I::AbstractVector) = ifelse.(hv.v[I], -1, 1)
Base.sum(hv::BipolarHV) = length(hv.v) - 2sum(hv.v)
LinearAlgebra.norm(hv::BipolarHV) = sqrt(length(hv))
empty_vector(hv::BipolarHV) = zeros(Int, length(hv))
eldist(::Type{BipolarHV}) = 2Bernoulli(0.5) - 1


# ------------------------------------------------------------------------------------ TernaryHV
"""
    TernaryHV(; D = 10_000, seed = nothing, rng = default_rng())
    TernaryHV(x)
    TernaryHV(v::AbstractVector{<:Real})
    TernaryHV{T}(...)

A ternary hypervector implementing the Multiply-Add-Permute (MAP) vector symbolic
architecture (Gayler, 1998). Elements are integers, stored as a `Vector{T}` with
`T <: Integer`; the random constructors generate only `±1` entries, and zeros arise
from operations such as unnormalized bundling. All constructor forms also exist with
an explicit element type, `TernaryHV{T}(...)`.

Under MAP, `bind` is elementwise multiplication and self-inverse, `bundle` is
elementwise addition *without* normalization by default (so counts accumulate;
`normalize` clamps the result back to `{-1, 0, +1}`), and `similarity` defaults to
cosine.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> TernaryHV(; D = 8, rng = Xoshiro(42))
8-element TernaryHV{Int64} with 4 positives, 0 zeros, and 4 negatives:
  1
  1
 -1
 -1
 -1
 -1
  1
  1

julia> TernaryHV("apple") == TernaryHV("apple")   # deterministic from hash
true
```

Bundling accumulates counts; `normalize` clamps back to `{-1, 0, +1}`:

```jldoctest
julia> x = TernaryHV(; D = 8, rng = Xoshiro(1)); y = TernaryHV(; D = 8, rng = Xoshiro(2));

julia> x + y
8-element TernaryHV{Int64} with 3 positives, 2 zeros, and 3 negatives:
 -2
  0
 -2
 -2
  0
  2
  2
  2

julia> normalize(x + y)
8-element TernaryHV{Int64} with 3 positives, 2 zeros, and 3 negatives:
 -1
  0
 -1
 -1
  0
  1
  1
  1
```

# See also

[`AbstractHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)

# Extended help

## References

- Gayler, R. W. (1998). Multiplicative Binding, Representation Operators & Analogy. In Advances in Analogy Research: Integration of Theory and Data from the Cognitive, Computational, and Neural Sciences, 1–4.
"""
struct TernaryHV{T <: Integer} <: AbstractHV{T}
    v::Vector{T}

    # Inner constructor for same type
    TernaryHV{T}(v::AbstractVector{T}) where {T <: Integer} = new{T}(v)
    # Inner constructor for type conversion
    TernaryHV{T}(v::AbstractVector{<:Integer}) where {T <: Integer} = new{T}(convert(Vector{T}, v))
end

# Data constructors validate the ternary domain. The inner `TernaryHV{T}`
# constructors stay permissive on purpose: operations legitimately exceed the
# domain (unnormalized bundling accumulates counts).
function TernaryHV(v::AbstractVector{T}) where {T <: Integer}
    all(x -> -1 ≤ x ≤ 1, v) || throw(ArgumentError("TernaryHV elements must be -1, 0, or +1"))
    return TernaryHV{T}(v)
end
function TernaryHV(v::AbstractVector{<:Real})
    all(x -> x == -1 || x == 0 || x == 1, v) || throw(ArgumentError("TernaryHV elements must be -1, 0, or +1"))
    return TernaryHV{Int}(convert(Vector{Int}, v))
end

function TernaryHV(;
        D::Integer = 10_000,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return TernaryHV{Int}(rand(rng_instance, (-1, 1), D))
end

function TernaryHV{T}(;
        D::Integer = 10_000,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    ) where {T <: Integer}
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return TernaryHV{T}(convert(Vector{T}, rand(rng_instance, (-1, 1), D)))
end

# Helpers
Base.copy(hv::TernaryHV{T}) where {T} = TernaryHV{T}(copy(hv.v))
Base.similar(hv::TernaryHV{T}) where {T} = TernaryHV{T}(; D = length(hv))
normalize!(hv::TernaryHV) = (clamp!(hv.v, -1, 1); hv)
eldist(::Type{<:TernaryHV}) = 2Bernoulli(0.5) - 1

# ------------------------------------------------------------------------------------ BinaryHV
"""
    BinaryHV(; D = 10_000, seed = nothing, rng = default_rng())
    BinaryHV(x)
    BinaryHV(v::AbstractVector{<:Real})

A binary hypervector implementing the Binary Spatter Code (BSC) vector symbolic
architecture (Kanerva, 1994–1997). Elements are `{false,true}`,
stored compactly as a `BitVector`.

Under BSC, `bind` is elementwise XOR and self-inverse (`x * x` is the identity
element), `bundle` is a majority vote across inputs with
deterministic tie-breaking, and `similarity` defaults to Jaccard.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> BinaryHV(; D = 8, rng = Xoshiro(42))
8-element BinaryHV with 3 true and 5 false:
 0
 0
 0
 0
 1
 1
 1
 0

julia> BinaryHV("apple") == BinaryHV("apple")   # deterministic from hash
true
```

Binding is self-inverse:

```jldoctest
julia> x = BinaryHV(; D = 8, rng = Xoshiro(1)); y = BinaryHV(; D = 8, rng = Xoshiro(2));

julia> x * y * y == x
true
```

# See also

[`AbstractHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)

# Extended help

## References

- Kanerva, P. (1994). The Spatter Code for Encoding Concepts at Many Levels. ICANN, 226–229.
- Kanerva, P. (1995). A Family of Binary Spatter Codes. ICANN, 517–522.
- Kanerva, P. (1996). Binary Spatter-Coding of Ordered K-tuples. ICANN, LNCS 1112, 869–873.
- Kanerva, P. (1997). Fully Distributed Representation. RWC, 358–365.
"""
struct BinaryHV <: AbstractHV{Bool}
    v::BitVector

    BinaryHV(v::AbstractVector{Bool}) = new(v)
end

function BinaryHV(;
        D::Integer = 10_000,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return BinaryHV(bitrand(rng_instance, D))
end

# Data constructor: elements must be Boolean-valued (0/1); Bool vectors hit
# the inner constructor directly.
function BinaryHV(v::AbstractVector{<:Real})
    all(x -> iszero(x) || isone(x), v) || throw(ArgumentError("BinaryHV elements must be 0 or 1"))
    return BinaryHV(BitVector(v .== 1))
end

# Helpers
empty_vector(hv::BinaryHV) = zeros(Int, length(hv))
eldist(::Type{BinaryHV}) = Bernoulli(0.5)


# --------------------------------------------------------------------------------------- RealHV
"""
    RealHV(; D = 10_000, distr = Normal(), seed = nothing, rng = default_rng())
    RealHV(x)
    RealHV(v::AbstractVector{<:Real}[, distr])

A real-valued hypervector (continuous Multiply-Add-Permute architecture). Elements
are drawn from a configurable distribution `distr` (standard normal by default),
which the vector carries along so that `normalize!` can rescale a result back to
the original spread.

Under this architecture, `bind` is elementwise multiplication, `bundle` is
elementwise addition rescaled by `√m` for `m` inputs, and `similarity` defaults
to cosine. Real-valued MAP binding is not exactly invertible, so [`unbind`](@ref)
**throws** for this type: recover bound information with [`similarity`](@ref)
against candidate hypervectors, or use [`FHRR`](@ref) or [`BipolarHV`](@ref) if
you need exact unbinding.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> RealHV(; D = 8, rng = Xoshiro(42))
8-element RealHV{Float64} with μ ± σ = -0.222 ± 0.736:
 -0.36335748145177754
  0.2517372155742292
 -0.31498797116895605
 -0.31125240132442067
  0.8163067649323273
  0.47673837983187795
 -0.8595553820616212
 -1.4692882055065464

julia> RealHV("apple") == RealHV("apple")   # deterministic from hash
true
```

The `distr` keyword controls the element distribution:

```jldoctest
julia> RealHV(; D = 8, distr = Normal(0, 5), rng = Xoshiro(1))
8-element RealHV{Float64} with μ ± σ = 0.214 ± 4.17:
  0.30966370157040063
  1.392029070820001
 -2.9791220768202606
  0.2332969478669087
  5.428970107716381
 -7.88282461292992
  0.8796999565053736
  4.326904027046626
```

# See also

[`AbstractHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)
"""
struct RealHV{T <: Real} <: AbstractHV{T}
    v::Vector{T}
    distr::Distribution

    RealHV(
        v::AbstractVector{T},
        distr::Distribution = eldist(RealHV)
    ) where {T <: Real} = new{T}(v, distr)
end

# Constructors
function RealHV(;
        distr::Distribution = eldist(RealHV),
        D::Integer = 10_000,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return RealHV(rand(rng_instance, distr, D), distr)
end

# Helpers
Base.copy(hv::RealHV) = RealHV(copy(hv.v), hv.distr)
Base.similar(hv::RealHV) = RealHV(; D = length(hv), distr = hv.distr)
function normalize!(hv::RealHV)
    hv.v .*= std(hv.distr) / std(hv.v)
    return hv
end
eldist(::Type{<:RealHV}) = Normal()
eldist(hv::RealHV) = hv.distr


# -------------------------------------------------------------------------------------- GradedHV
"""
    GradedHV(; D = 10_000, distr = Beta(1, 1), seed = nothing, rng = default_rng())
    GradedHV(x)
    GradedHV(v::AbstractVector{<:Real}[, distr])

A graded hypervector with elements in the fuzzy-membership interval `[0, 1]`.
Elements are drawn from a distribution with support in `[0, 1]` (uniform
`Beta(1, 1)` by default, via the `distr` keyword); values passed as data are
clamped to `[0, 1]`.

Operations follow fuzzy logic: `bind` is the fuzzy XOR
`(1 - x) * y + x * (1 - y)`, `bundle` uses the three-valued π aggregation, and
`similarity` defaults to Jaccard.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> GradedHV(; D = 8, rng = Xoshiro(42))
8-element GradedHV{Float64} with μ ± σ = 0.532 ± 0.247:
 0.8023279156644033
 0.6042216741680727
 0.5612409791764235
 0.8514212832811604
 0.5873457677614401
 0.4663593857124599
 0.13521235508492413
 0.24655411787892703

julia> GradedHV("apple") == GradedHV("apple")   # deterministic from hash
true
```

Binding is fuzzy XOR, so binding with certainty (`1.0`) negates the membership:

```jldoctest
julia> GradedHV([1.0, 0.0, 0.5]) * GradedHV([1.0, 1.0, 1.0])
3-element GradedHV{Float64} with μ ± σ = 0.5 ± 0.5:
 0.0
 1.0
 0.5
```

# See also

[`AbstractHV`](@ref), [`GradedBipolarHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)
"""
struct GradedHV{T <: Real} <: AbstractHV{T}
    v::Vector{T}
    distr::Distribution

    function GradedHV(
            v::AbstractVector{T},
            distr::Distribution = eldist(GradedHV)
        ) where {T <: Real}
        all(x -> 0 ≤ x ≤ 1, v) || throw(ArgumentError("GradedHV elements must lie in [0, 1]"))
        return new{T}(v, distr)
    end
end

# Constructors
function GradedHV(;
        D::Integer = 10_000,
        distr::Distribution = eldist(GradedHV),
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    @assert 0 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [0,1]"
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return GradedHV(rand(rng_instance, distr, D), distr)
end

# Helpers
Base.copy(hv::GradedHV) = GradedHV(copy(hv.v), hv.distr)
Base.similar(hv::GradedHV) = GradedHV(; D = length(hv), distr = hv.distr)
Base.zeros(hv::GradedHV) = fill!(similar(hv.v), one(eltype(hv.v)) / 2)
normalize!(hv::GradedHV) = (clamp!(hv.v, 0, 1); hv)
eldist(::Type{<:GradedHV}) = Beta(1, 1)
eldist(hv::GradedHV) = hv.distr
empty_vector(hv::GradedHV) = fill!(zero(hv.v), 0.5)


# -------------------------------------------------------------------------------- GradedBipolarHV
"""
    GradedBipolarHV(; D = 10_000, distr = 2Beta(1, 1) - 1, seed = nothing, rng = default_rng())
    GradedBipolarHV(x)
    GradedBipolarHV(v::AbstractVector{<:Real}[, distr])

A graded bipolar hypervector with elements in `[-1, 1]`, the bipolar counterpart of
[`GradedHV`](@ref). Elements are drawn from a distribution with support in `[-1, 1]`
(the scaled uniform `2Beta(1, 1) - 1` by default, via the `distr` keyword); values
passed as data are clamped to `[-1, 1]`.

Operations are the fuzzy-logic operations of [`GradedHV`](@ref) mapped to the
bipolar interval: `bind` is fuzzy XOR and `bundle` the three-valued π aggregation,
both applied after rescaling `[-1, 1]` to `[0, 1]` and mapping back; `similarity`
defaults to cosine.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> GradedBipolarHV(; D = 8, rng = Xoshiro(42))
8-element GradedBipolarHV{Float64} with μ ± σ = 0.064 ± 0.494:
  0.6046558313288066
  0.20844334833614542
  0.12248195835284692
  0.7028425665623208
  0.1746915355228802
 -0.06728122857508023
 -0.7295752898301517
 -0.506891764242146

julia> GradedBipolarHV("apple") == GradedBipolarHV("apple")   # deterministic from hash
true
```

Binding with full certainty (`1.0`) mirrors a value across the interval:

```jldoctest
julia> GradedBipolarHV([-1.0, 0.0, 1.0]) * GradedBipolarHV([1.0, 1.0, 1.0])
3-element GradedBipolarHV{Float64} with μ ± σ = 0.0 ± 1.0:
  1.0
  0.0
 -1.0
```

# See also

[`AbstractHV`](@ref), [`GradedHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`similarity`](@ref)
"""
struct GradedBipolarHV{T <: Real} <: AbstractHV{T}
    v::Vector{T}
    distr::Distribution

    function GradedBipolarHV(
            v::AbstractVector{T},
            distr::Distribution = eldist(GradedBipolarHV)
        ) where {T <: Real}
        all(x -> -1 ≤ x ≤ 1, v) || throw(ArgumentError("GradedBipolarHV elements must lie in [-1, 1]"))
        return new{T}(v, distr)
    end
end

function GradedBipolarHV(;
        D::Integer = 10_000,
        distr::Distribution = eldist(GradedBipolarHV),
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    @assert -1 ≤ minimum(distr) < maximum(distr) ≤ 1 "Provide `distr` with support in [-1,1]"
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return GradedBipolarHV(rand(rng_instance, distr, D), distr)
end

# Helpers
Base.copy(hv::GradedBipolarHV) = GradedBipolarHV(copy(hv.v), hv.distr)
Base.similar(hv::GradedBipolarHV) = GradedBipolarHV(; D = length(hv), distr = hv.distr)
normalize!(hv::GradedBipolarHV) = (clamp!(hv.v, -1, 1); hv)
eldist(::Type{<:GradedBipolarHV}) = 2Beta(1, 1) - 1
eldist(hv::GradedBipolarHV) = hv.distr

# Fourier Holographic Reduced Representations
# --------------------------------------------

"""
    FHRR(; D = 10_000, T = Float64, seed = nothing, rng = default_rng())
    FHRR(x)
    FHRR(v::AbstractVector{<:Complex})

A Fourier Holographic Reduced Representation hypervector (Plate, 1995). Elements are
complex numbers on the unit circle, `e^(iθ)` with random phase `θ`, stored as a
`Vector{Complex{T}}` (`T = Float64` by default, via the `T` keyword).

Under FHRR, `bind` is elementwise complex multiplication (phases add), inverted by
[`unbind`](@ref) (elementwise division), `bundle` is phasor addition renormalized to
unit modulus, and `similarity` is the normalized real part of the complex dot
product. In addition, `hv ^ x` raises every phase to the power `x`, which enables
fractional-power (level) encoding of continuous values.

`HV(x)` is shorthand for [`encode`](@ref)`(HV, x)`, the deterministic token path;
a `Number` argument throws — use `D = n` for dimensionality, or `encode` for
number tokens. See [`AbstractHV`](@ref) for the full convention.

Indexing with a scalar returns a single element; indexing with a range or vector
returns a plain `Vector`, not a hypervector.

# Examples

```jldoctest
julia> FHRR(; D = 4, rng = Xoshiro(42))
4-element FHRR{ComplexF64}:
 -0.6875407989187119 - 0.7261457497102214im
 -0.9517124499338168 + 0.3069908999318585im
 -0.9899412825080958 + 0.14147882239482543im
 -0.2902555218408553 - 0.9569491794452267im

julia> FHRR("apple") == FHRR("apple")   # deterministic from hash
true
```

Fractional powers encode continuous values: nearby exponents stay similar.

```jldoctest
julia> x = FHRR(; rng = Xoshiro(1));

julia> similarity(x^1.0, x^1.05) > similarity(x^1.0, x^2.0)
true
```

# See also

[`AbstractHV`](@ref), [`bundle`](@ref), [`bind`](@ref), [`unbind`](@ref), [`similarity`](@ref)

# Extended help

## References

- Plate, T. A. (1995). Holographic Reduced Representations. IEEE Transactions on Neural Networks, 6(3), 623–641.
"""
struct FHRR{T <: Complex} <: AbstractHV{T}
    v::Vector{T}

    function FHRR{T}(v::Vector{T}) where {T <: Complex}
        all(z -> abs(z) ≈ 1, v) || throw(
            ArgumentError("FHRR elements must lie on the complex unit circle (unit modulus)")
        )
        return new{T}(v)
    end
end

FHRR(v::AbstractVector{T}) where {T <: Complex} = FHRR{T}(convert(Vector{T}, v))

function FHRR(;
        D::Integer = 10_000,
        T::Type = Float64,
        seed::Union{Integer, Nothing} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )
    rng_instance = isnothing(seed) ? rng : Xoshiro(seed)
    return FHRR(exp.(2π * im .* rand(rng_instance, T, D)))
end

Base.similar(hv::FHRR{<:Complex{R}}) where {R} = FHRR(exp.(2π * im .* rand(R, length(hv))))

"""
    normalize!(hv::FHRR)

A Fourier Holographic Reduced Representation is normalized by setting the norm of each complex element to 1.
"""
function normalize!(hv::FHRR)
    hv.v ./= abs.(hv.v)
    return hv
end

# ---------------------------------------------------------------------------------------  Traits
abstract type HVTraits end

struct HVByteVec <: HVTraits end
struct HVBitVec <: HVTraits end

vectype(::AbstractHV) = HVByteVec
vectype(::BinaryHV) = HVBitVec
vectype(::BipolarHV) = HVBitVec
