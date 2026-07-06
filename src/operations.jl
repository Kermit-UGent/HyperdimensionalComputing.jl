#=
operations.jl; This file implements operations that can be done on hypervectors to enable them to encode text-based data.
=#

#=

| Operation            | symbol | remark                                                                                                          |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Bundling/aggregating | `+`    | combines the information of two vectors into a new vector similar to both                                       |
| Binding              | `*`    | mapping, combines the two vectors in something different from both, preserves distance, distributes of bundling |
| Shifting             | `ρ`    | Permutation (in practice cyclic shifting), distributes over addition, conserves distance                        |
=#

"""
    grad2bipol(x::Number)

Maps a graded number in [0, 1] to the [-1, 1] interval.
"""
grad2bipol(x::Real) = 2x - one(x)


"""
bipol2grad(x::Number)

Maps a bipolar number in [-1, 1] to the [0, 1] interval.
"""
bipol2grad(x::Real) = (x + one(x)) / 2

three_pi(x, y) = abs(x - y) == 1 ? zero(x) : x * y / (x * y + (one(x) - x) * (one(y) - y))
fuzzy_xor(x, y) = (one(x) - x) * y + x * (one(y) - y)

three_pi_bipol(x, y) = grad2bipol(three_pi(bipol2grad(x), bipol2grad(y)))
fuzzy_xor_bipol(x, y) = grad2bipol(fuzzy_xor(bipol2grad(x), bipol2grad(y)))  # currently just *

aggfun(::Type{<:AbstractHV}) = +
aggfun(::GradedHV) = three_pi
aggfun(::GradedBipolarHV) = three_pi_bipol

bindfun(::AbstractHV) = *
bindfun(::BinaryHV) = ⊻
bindfun(::GradedHV) = fuzzy_xor
bindfun(::GradedBipolarHV) = fuzzy_xor_bipol

neutralbind(hdv::AbstractHV) = one(eltype(hdv))
neutralbind(hdv::BinaryHV) = false
neutralbind(hdv::GradedHV) = zero(eltype(hdv))
neutralbind(hdv::GradedBipolarHV) = -one(eltype(hdv))

noisy_and(a, b) = a == b ? a : rand(Bool)

function elementreduce!(f, itr, init)
    return foldl(itr; init) do acc, value
        acc .= f.(acc, value)
    end
end

# computes `r[i] = f(x[i], y[i+offset])`
# assumes postive offset (for now)
@inline function offsetcombine!(r, f, x, y, offset = 0)
    @assert length(r) == length(x) == length(y)
    n = length(r)
    if offset == 0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

@inline function offsetcombine(f, x::V, y::V, offset = 0) where {V <: AbstractVecOrMat}
    @assert length(x) == length(y)
    r = similar(x)
    n = length(r)
    if offset == 0
        r .= f.(x, y)
    else
        i′ = n - offset
        for i in 1:n
            i′ = i′ == n ? 1 : i′ + 1
            @inbounds r[i] = f(x[i], y[i′])
        end
    end
    return r
end

# BUNDLE
# ------

# binary and bipolar: use majority
function bundle(hvr::Union{BinaryHV, BipolarHV}, hdvs, r)
    m = length(hdvs)
    for hv in hdvs
        r .+= hv.v
    end
    if iseven(m)  # break ties
        r .+= bitrand(length(r))
    end
    hvr = similar(hvr)
    hvr.v .= r .> m / 2
    return hvr
end

# ternary: just add them, no normalization by default
function bundle(
        ::TernaryHV, hdvs, r;
        normalize = false
    )
    for hv in hdvs
        r .+= hv.v
    end
    normalize && clamp!(r, -1, 1)
    return TernaryHV(r)
end

# realhv: just add + rescale with sqrt m
function bundle(::RealHV, hdvs, r)
    m = 0
    for hv in hdvs
        r .+= hv.v
        m += 1
    end
    r ./= sqrt(m)
    return RealHV(r)
end

function bundle(::GradedHV, hdvs, r)
    for hv in hdvs
        r .= three_pi.(r, hv.v)
    end
    return GradedHV(r)
end

function bundle(::GradedBipolarHV, hdvs, r)
    for hv in hdvs
        r .= three_pi_bipol.(r, hv.v)
    end
    return GradedBipolarHV(r)
end

function bundle(::FHRR, hdvs, r)
    for hv in hdvs
        r .+= hv.v
    end
    r ./= abs.(r)
    return FHRR(r)
end

function bundle(hdvs; kwargs...)
    hv = first(hdvs)
    r = empty_vector(hv)
    return bundle(hv, hdvs, r, kwargs...)
end

Base.:+(hv1::HV, hv2::HV) where {HV <: AbstractHV} = bundle((hv1, hv2))

# BINDING
# -------
Base.bind(hv1::HV, hv2::HV) where {HV <: AbstractHV} = HV(hv1.v .* hv2.v)  # default
Base.bind(hv1::BinaryHV, hv2::BinaryHV) = BinaryHV(hv1.v .⊻ hv2.v)
Base.bind(hv1::BipolarHV, hv2::BipolarHV) = BipolarHV(hv1.v .⊻ hv2.v)
Base.bind(hv1::TernaryHV, hv2::TernaryHV) = TernaryHV(hv1.v .* hv2.v)
Base.bind(hv1::RealHV, hv2::RealHV) = RealHV(hv1.v .* hv2.v)
Base.bind(hv1::GradedHV, hv2::GradedHV) = GradedHV(fuzzy_xor.(hv1.v, hv2.v))
Base.bind(hv1::GradedBipolarHV, hv2::GradedBipolarHV) = GradedBipolarHV(fuzzy_xor_bipol.(hv1.v, hv2.v))
Base.bind(hv1::FHRR, hv2::FHRR) = FHRR(hv1.v .* hv2.v)
Base.:*(hv1::HV, hv2::HV) where {HV <: AbstractHV} = bind(hv1, hv2)
Base.bind(hvs::AbstractVector{HV}) where {HV <: AbstractHV} = prod(hvs)


"""
    unbind(hv1::HV, hv2::HV)

Unbinds `hv2` from `hv1`. For many types of hypervectors, the binding operator is
idempotent, i.e., `u * v * v == u`.

Aliases with `/`.
"""
unbind(hv1::HV, hv2::HV) where {HV <: AbstractHV} = bind(hv1, hv2)
unbind(hv1::HV, hv2::HV) where {HV <: Union{RealHV, FHRR}} = HV(hv1.v ./ hv2.v)
Base.:/(hv1::HV, hv2::HV) where {HV <: AbstractHV} = unbind(hv1, hv2)


# SHIFTING
# --------

# Shifting / Permutation
shift!(hv::AbstractHV, k = 1) = circshift!(hv.v, k)

function shift(hv::AbstractHV, k = 1)
    r = similar(hv)
    r.v .= circshift(hv.v, k)
    return r
end

function shift!(hv::V, k = 1) where {V <: Union{BinaryHV, BipolarHV}}
    v = similar(hv.v)  # empty bitvector
    hv.v .= circshift!(v, hv.v, k)
    return hv
end

function shift(hv::V, k = 1) where {V <: Union{BinaryHV, BipolarHV}}
    v = similar(hv.v)  # empty bitvector
    return V(circshift!(v, hv.v, k))
end

ρ(hv::AbstractHV, k = 1) = shift(hv, k)
ρ!(hv::AbstractHV, k = 1) = shift!(hv, k)


# Comparison
Base.isequal(v::AbstractHV, u::AbstractHV) = v.v == u.v

"""
    Base.isapprox(u::AbstractHV, v::AbstractHV, atol=length(u)/100, ptol=0.01)

Measures when two hypervectors are similar (have more elements in common than expected
by chance).

One can specify either:
- `atol=N/100` number of matches more than due to chance needed for being assumed similar
- `ptol=0.01` threshold for seeing that many matches due to chance
"""
function Base.isapprox(u::T, v::T; atol = length(u) / 100, ptol = 0.01) where {T <: Union{BinaryHV, BipolarHV}}
    @assert length(u) == length(v) "Vectors have to be of equal length"
    N = length(u)
    missmatches = sum(ui != vi for (ui, vi) in zip(u, v))
    matches = N - missmatches
    # probability of seeing fewer mismatches due to chance
    pval = cdf(Binomial(N, 0.5), missmatches)
    return pval < ptol || matches - N / 2 > atol
end

"""
    Base.isapprox(u::AbstractHV, v::AbstractHV, atol=length(u)/100, ptol=0.01)

Measures when two hypervectors are similar (have more elements in common than expected
by chance) using the Hamming distance. Uses a bootstrap to construct a null distribution.

One can specify either:
- `ptol=1e-10` threshold for seeing that many matches due to chance
- `N_bootstrap=200` number of samples for bootstrapping
"""
function Base.isapprox(u::T, v::T; ptol = 1.0e-10, N_bootstrap = 500) where {T <: AbstractHV}
    @assert length(u) == length(v) "Vectors have to be of equal length"
    N = length(u)
    # bootstrap to find the zero distr
    B = [abs(rand(u) - rand(v)) for _ in 1:N_bootstrap]
    Bmean = N * mean(B)
    Bstd = sqrt(N) * std(B)
    # Hamming distance
    d = sum(abs(ui - vi) for (ui, vi) in zip(u, v))
    # probability of seeing fewer mismatches due to chance
    pval = cdf(Normal(Bmean, Bstd), d)
    return pval < ptol
end


# Perturbation
function randbv(n::Int, m::Int; rng::AbstractRNG = Random.GLOBAL_RNG)
    v = falses(n)
    v[1:m] .= true
    return shuffle!(rng, v)
end

function randbv(n::Int, p::Number; rng::AbstractRNG = Random.GLOBAL_RNG)
    @assert 0 ≤ p ≤ 1 "p should be a valid probability"
    return randbv(n, round(Int, p * n); rng = rng)
end

function randbv(n::Int, I; rng::AbstractRNG = Random.GLOBAL_RNG)
    v = falses(n)
    v[I] .= true
    return v
end

function perturbate!(::Type{HVByteVec}, hv::HV, I, dist = eldist(hv); rng::AbstractRNG = Random.GLOBAL_RNG) where {HV <: AbstractHV}
    hv.v[I] .= rand(rng, dist, length(I))
    return hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, M::BitVector, dist = eldist(hv); rng::AbstractRNG = Random.GLOBAL_RNG) where {HV <: AbstractHV}
    hv.v[M] .= rand(rng, dist, sum(M))
    return hv
end

function perturbate!(::Type{HVByteVec}, hv::HV, p::Number, args...; rng::AbstractRNG = Random.GLOBAL_RNG) where {HV <: AbstractHV}
    return perturbate!(hv, randbv(length(hv), p; rng = rng), args...; rng = rng)
end

function perturbate!(::Type{HVBitVec}, hv::AbstractHV, binargs; rng::AbstractRNG = Random.GLOBAL_RNG)
    n = length(hv)
    M = randbv(n, binargs; rng = rng)
    hv.v .⊻= M
    return hv
end

perturbate!(hv, args...; rng::AbstractRNG = Random.GLOBAL_RNG) = perturbate!(vectype(hv), hv, args...; rng = rng)
perturbate(hv::AbstractHV, args...; rng::AbstractRNG = Random.GLOBAL_RNG, kwargs...) = perturbate!(copy(hv), args...; rng = rng, kwargs...)

# OTHER
# -----
Base.:^(hv::FHRR, x::Number) = FHRR(hv.v .^ x)