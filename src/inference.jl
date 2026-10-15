#=
This file implements all similarity metrics + modules for finding the closest match.
=#

using LinearAlgebra

LinearAlgebra.dot(u::AbstractHV, v::AbstractHV) = dot(u.v, v.v)

LinearAlgebra.dot(u::BipolarHV, v::BipolarHV) = 4dot(u.v, v.v) - 2sum(u.v) - 2sum(v.v) + length(u)

sim_cos(u::AbstractVector, v::AbstractVector) = dot(u, v) / (norm(u) * norm(v))
sim_cos(u::AbstractVector{<:Integer}, v::AbstractVector{<:Integer}) = begin
    u64 = Int64.(u)
    v64 = Int64.(v)
    dot(u64, v64) / (norm(u64) * norm(v64))
end

sim_jacc(u::AbstractVector, v::AbstractVector) = dot(u, v) / sum(ui + vi - ui * vi for (ui, vi) in zip(u, v))

dist_hamming(u::AbstractVector, v::AbstractVector) = sum(abs(ui - vi) for (ui, vi) in zip(u, v))

# The `similarity` methods below are single fused passes over the stored
# representation, without converting or copying the operands.

# Bits: popcounts on the packed 64-bit words of the `BitVector`s (its `chunks`, whose
# bits past the end are always zero). A bipolar dot product is the number of
# agreeing minus disagreeing positions, and both norms are `sqrt(length(u))`.
function popcount(op, u::BitVector, v::BitVector)
    length(u) == length(v) || throw(DimensionMismatch("hypervectors must have equal length: got $(length(u)) and $(length(v))"))
    s = 0
    for (a, b) in zip(u.chunks, v.chunks)
        s += count_ones(op(a, b))
    end
    return s
end
similarity(u::BinaryHV, v::BinaryHV) = popcount(&, u.v, v.v) / popcount(|, u.v, v.v)
similarity(u::BipolarHV, v::BipolarHV) = (length(u) - 2popcount(⊻, u.v, v.v)) / length(u)

# Accumulates in floating point, which for integer elements stays exact below 2^53
# and, unlike Int64, vectorizes on targets without SIMD 64-bit integer multiplication
# (e.g. AArch64 NEON).
function sim_cos_fused(u::AbstractVector, v::AbstractVector)
    R = float(promote_type(eltype(u), eltype(v)))
    d = nu = nv = zero(R)
    @simd for i in eachindex(u, v)
        x = R(u[i])
        y = R(v[i])
        d = muladd(x, y, d)
        nu = muladd(x, x, nu)
        nv = muladd(y, y, nv)
    end
    return d / (sqrt(nu) * sqrt(nv))
end

function sim_jacc_fused(u::AbstractVector, v::AbstractVector)
    R = float(promote_type(eltype(u), eltype(v)))
    d = s = zero(R)
    @simd for i in eachindex(u, v)
        x = R(u[i])
        y = R(v[i])
        d = muladd(x, y, d)
        s += x + y - x * y
    end
    return d / s
end

# Real part of the Hermitian inner product, `Σ re(conj(x) * y)`, divided by the length.
function sim_phasor_fused(u::AbstractVector{<:Complex}, v::AbstractVector{<:Complex})
    R = float(real(promote_type(eltype(u), eltype(v))))
    d = zero(R)
    @simd for i in eachindex(u, v)
        x, y = u[i], v[i]
        d = muladd(real(x), real(y), d)
        d = muladd(imag(x), imag(y), d)
    end
    return d / length(u)
end

similarity(u::TernaryHV, v::TernaryHV) = sim_cos_fused(u.v, v.v)
similarity(u::RealHV, v::RealHV) = sim_cos_fused(u.v, v.v)
similarity(u::GradedBipolarHV, v::GradedBipolarHV) = sim_cos_fused(u.v, v.v)
similarity(u::GradedHV, v::GradedHV) = sim_jacc_fused(u.v, v.v)
similarity(u::FHRR, v::FHRR) = sim_phasor_fused(u.v, v.v)

"""
    similaritymetric(HV)
    similaritymetric(hv::AbstractHV)

Which metric [`similarity`](@ref) uses for hypervectors of type `HV`, as a `Symbol`.

`:jaccard` for [`BinaryHV`](@ref) and [`GradedHV`](@ref), whose elements are
non-negative, and `:cosine` for the types that can take negative values --
[`BipolarHV`](@ref), [`TernaryHV`](@ref), [`RealHV`](@ref),
[`GradedBipolarHV`](@ref) and [`FHRR`](@ref). (For `FHRR` the normalized real part
of the Hermitian inner product *is* the cosine similarity, since every element has
unit modulus.)

Use together with [`chancesimilarity`](@ref), which tells you what an
*unrelated* pair scores under that metric -- the number you need to judge whether a
given similarity is large.

# Examples

```jldoctest
julia> similaritymetric(BinaryHV), similaritymetric(BipolarHV)
(:jaccard, :cosine)

julia> similaritymetric(BinaryHV(; D = 10))   # also works on an instance
:jaccard
```

# See also

[`similarity`](@ref), [`chancesimilarity`](@ref)
"""
similaritymetric(::Type{<:AbstractHV}) = :cosine
similaritymetric(::Type{BinaryHV}) = :jaccard
similaritymetric(::Type{<:GradedHV}) = :jaccard
similaritymetric(hv::AbstractHV) = similaritymetric(typeof(hv))

"""
    chancesimilarity(HV)
    chancesimilarity(hv::AbstractHV)

The expected [`similarity`](@ref) between two *independent random* hypervectors of
type `HV`: the value that means "unrelated".

This is the baseline you need to interpret a similarity score, and it is **not
always zero**. Under cosine (see [`similaritymetric`](@ref)) unrelated hypervectors
score `0.0`, but under Jaccard they score `1/3` -- so a similarity of `0.35` is
strong evidence of a relationship for a [`BipolarHV`](@ref) and no evidence at all
for a [`BinaryHV`](@ref).

!!! note
    The value assumes hypervectors built with the type's default element
    distribution. Passing a custom `distr` to [`RealHV`](@ref), [`GradedHV`](@ref)
    or [`GradedBipolarHV`](@ref) can shift the baseline: a distribution that is not
    centred makes random vectors point in a common direction, raising the chance
    level above zero.

# Examples

```jldoctest
julia> chancesimilarity(BipolarHV)
0.0

julia> chancesimilarity(BinaryHV)      # Jaccard: unrelated is 1/3, not 0
0.3333333333333333
```

# See also

[`similarity`](@ref), [`similaritymetric`](@ref)
"""
chancesimilarity(::Type{<:AbstractHV}) = 0.0
chancesimilarity(::Type{BinaryHV}) = 1 / 3
chancesimilarity(::Type{<:GradedHV}) = 1 / 3
chancesimilarity(hv::AbstractHV) = chancesimilarity(typeof(hv))

"""
    similarity(u::AbstractVector, v::AbstractVector; method::Symbol)

Computes similarity between two (hyper)vectors using a `method` ∈
`[:cosine, :jaccard, :hamming]`. When no method is given, a default is used
(cosine for vectors that can have negative elements and Jaccard for those that
only have positive elements).
"""
function similarity(u::AbstractVector, v::AbstractVector; method::Symbol)
    @assert length(u) == length(v) "Vectors have to be of the same length"
    methods = [:cosine, :jaccard, :hamming]
    @assert method ∈ methods "`method` has to be one of $methods"
    if method == :cosine
        return sim_cos(u, v)
    elseif method == :jaccard
        return sim_jacc(u, v)
    elseif method == :hamming
        return length(u) - dist_hamming(u, v)
    end
end

"""
    similarity(hvs::AbstractVector{<:AbstractHV}; [method])

Computes the similarity matrix for a vector of hypervectors using
the similarity metrics defined by the pairwise version of `similarity`.
"""
function similarity(hvs::AbstractVector{<:AbstractHV}; kwargs...)
    n = length(hvs)
    S = zeros(n, n)
    for i in 1:n
        for j in i:n
            S[i, j] = S[j, i] = similarity(hvs[i], hvs[j]; kwargs...)
        end
    end
    return S
end

"""
    similarity(u::AbstractHV; [method])

Create a function that computes the similarity between its argument and `u`` 
using `similarity`, i.e. a function equivalent to `v -> similarity(u, v)`.
"""
similarity(u::AbstractHV; kwargs...) = v -> similarity(u, v; kwargs...)


"""
    δ(u::AbstractHV, v::AbstractHV; [method])
    δ(u::AbstractHV; [method])
    δ(hvs::AbstractVector{<:AbstractHV}; [method])

Alias for `similarity`. See `similarity` for the main documentation.
"""
const δ = similarity


nearest_neighbor(u::AbstractHV, collection; kwargs...) =
    maximum(
    (similarity(u, xi; kwargs...), i, xi)
        for (i, xi) in enumerate(collection)
)

nearest_neighbor(u::AbstractHV, collection::Dict; kwargs...) =
    maximum((similarity(u, xi; kwargs...), k, xi) for (k, xi) in collection)

"""
    nearest_neighbor(u::AbstractHV, collection[, k::Int]; kwargs...)

Returns the element of `collection` that is most similar to `u`. 

Function outputs `(τ, i, xi)` with `τ` the highest similarity value,
`i` the index (or key if `collection` is a dictionary) of the closest 
neighbor and `xi` the closest vector. `kwargs` is an optional argument
for the similarity search.

If a number `k` is given, the `k` closest neighbor are returned, as a sorted
list of `(τ, i)`.
"""
function nearest_neighbor(u::AbstractHV, collection, k::Int; kwargs...)
    sims = [
        (similarity(u, xi; kwargs...), i)
            for (i, xi) in enumerate(collection)
    ]
    return partialsort!(sims, 1:k, rev = true)
end

function nearest_neighbor(u::AbstractHV, collection::Dict, k::Int; kwargs...)
    sims = [
        (similarity(u, xi; kwargs...), i)
            for (i, xi) in collection
    ]
    return partialsort!(sims, 1:k, rev = true)
end
