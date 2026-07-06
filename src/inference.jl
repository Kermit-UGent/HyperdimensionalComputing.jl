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

similarity(u::BipolarHV, v::BipolarHV) = sim_cos(u, v)
similarity(u::TernaryHV, v::TernaryHV) = sim_cos(u, v)
similarity(u::GradedBipolarHV, v::GradedBipolarHV) = sim_cos(u, v)
similarity(u::RealHV, v::RealHV) = sim_cos(u, v)
similarity(u::BinaryHV, v::BinaryHV) = sim_jacc(u, v)
similarity(u::GradedHV, v::GradedHV) = sim_jacc(u, v)
similarity(u::FHRR, v::FHRR) = real(dot(u.v, v.v)) / length(u)

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
δ = similarity


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
