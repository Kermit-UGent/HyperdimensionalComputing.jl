#=
kernels.jl; elementwise kernels shared by the hypervector operations, run in parallel
with Polyester when there is enough work.

Code inside a `@batch` body must never throw. Under Polyester an exception raised by
a worker task leaves every later `@batch` in the process running on a single thread,
and an exception raised in the calling thread's share of the iterations crashes the
process on the next `@batch`. Every kernel here therefore validates its inputs (lengths,
element types) before entering `@batch`, and the callers route anything that could
fail, such as an element conversion, to a serial path that throws normally.
=#

# Functions passed to these kernels are annotated `f::F ... where {F}`: Julia does not
# specialize on a function argument that is only passed on (e.g. to broadcasting),
# and the resulting dynamic dispatch costs more than small kernels themselves.

# Positions are split into blocks of whole multiples of `BLOCK_QUANTUM` elements, so
# that no two tasks write to the same cache line, and of at most `MAX_BLOCK`
# elements, so that a block of an accumulator stays in L1 cache while several
# inputs stream over it. Below `PARALLEL_MIN_WORK` element operations, or with a
# single Julia thread, a kernel makes one pass over the full range on the calling
# thread instead: waking the worker threads costs more than small work saves, and
# serially, long sequential reads of each input prefetch better than the blocked
# order once the inputs outgrow the cache.
const BLOCK_QUANTUM = 128
const MAX_BLOCK = 1024
const PARALLEL_MIN_WORK = 1 << 13

function blocksize(n::Integer)
    per_thread = cld(n, Threads.nthreads())
    return clamp(BLOCK_QUANTUM * cld(per_thread, BLOCK_QUANTUM), BLOCK_QUANTUM, MAX_BLOCK)
end

# Whether `work` element operations over `n` positions are worth splitting across
# threads. Callers that are not use plain serial broadcasts, which for small inputs
# are also cheaper than the blocked kernels' views and checks.
parallel_worthwhile(n::Integer, work::Integer) =
    Threads.nthreads() > 1 && work >= PARALLEL_MIN_WORK && n > blocksize(n)

# Calls `f!(rng)` for consecutive ranges `rng` that together cover `idx`. `work` is
# the total number of element operations the calls perform.
function foreach_block!(f!::F, idx::AbstractUnitRange, work::Integer) where {F}
    n = length(idx)
    if !parallel_worthwhile(n, work)
        f!(idx)
        return nothing
    end
    block = blocksize(n)
    lo, hi = first(idx), last(idx)
    @batch for b in 1:cld(n, block)
        start = lo + (b - 1) * block
        f!(start:min(start + block - 1, hi))
    end
    return nothing
end
foreach_block!(f!::F, r::AbstractVector, work::Integer) where {F} = (foreach_block!(f!, eachindex(r), work); r)

# Calls `f(dst, src)` on the (at most two) contiguous pieces of `rng` such that
# `circshift(v, s)[dst] == v[src]` for a 1-based `v` of length `n ≥ length(rng)`.
@inline function foreach_shifted(f::F, rng::AbstractUnitRange, s::Integer, n::Integer) where {F}
    src0 = mod(first(rng) - 1 - s, n) + 1
    len1 = min(length(rng), n - src0 + 1)
    f(first(rng):(first(rng) + len1 - 1), src0:(src0 + len1 - 1))
    len1 < length(rng) && f((first(rng) + len1):last(rng), 1:(length(rng) - len1))
    return nothing
end

function check_samelength(r, hvs)
    for hv in hvs
        length(hv) == length(r) ||
            throw(DimensionMismatch("hypervectors must have equal length: got $(length(hv)) and $(length(r))"))
    end
    return nothing
end

# Element types whose arithmetic cannot throw: fixed-width integers wrap around and
# IEEE floats saturate to Inf/NaN. (Rationals, for example, throw on overflow.)
const NonThrowingElement = Union{Bool, Base.BitInteger, Base.IEEEFloat, Complex{<:Union{Base.BitInteger, Base.IEEEFloat}}}

# Whether `f` can run inside `@batch` on elements of types `Tr` (the accumulator) and
# `Tx`: both are `NonThrowingElement`s and `f` maps them to a `Tr`, so storing its
# result into the accumulator cannot fail either.
kernel_safe(f, ::Type{Tr}, ::Type{Tx}) where {Tr, Tx} =
    Tr <: NonThrowingElement && Tx <: NonThrowingElement && typeof(f(zero(Tr), zero(Tx))) === Tr

# Whether `hvs` is a non-empty collection of one concrete hypervector type, the
# precondition for the fused kernels.
function uniform_collection(hvs)
    isempty(hvs) && return false
    T = typeof(first(hvs))
    return all(hv -> typeof(hv) === T, hvs)
end

# `cost` arguments weight the work per element against `PARALLEL_MIN_WORK`, in units
# of a cheap arithmetic operation (~0.2 ns); e.g. `abs` of a complex number is ~16.

# `all(pred, v)` evaluated blockwise in parallel; `pred` must not throw.
function parallel_all(pred::P, v::AbstractVector, cost::Integer = 1) where {P}
    idx = eachindex(v)
    (eltype(v) <: NonThrowingElement && parallel_worthwhile(length(idx), cost * length(idx))) || return all(pred, v)
    block = blocksize(length(idx))
    lo, hi = first(idx), last(idx)
    ok = Vector{Bool}(undef, cld(length(idx), block))
    @batch for b in eachindex(ok)
        start = lo + (b - 1) * block
        ok[b] = all(pred, view(v, start:min(start + block - 1, hi)))
    end
    return all(ok)
end

# `f.(a, b)` computed blockwise in parallel. `f` must be type-stable and must not
# throw for `NonThrowingElement`s; other element types are broadcast serially.
function elementwise(f::F, a::AbstractVector, b::AbstractVector, cost::Integer = 1) where {F}
    axes(a) == axes(b) || throw(DimensionMismatch("hypervectors must have equal length: got $(length(a)) and $(length(b))"))
    (eltype(a) <: NonThrowingElement && eltype(b) <: NonThrowingElement && parallel_worthwhile(length(a), cost * length(a))) ||
        return f.(a, b)
    out = similar(a, typeof(f(zero(eltype(a)), zero(eltype(b)))))
    foreach_block!(out, cost * length(out)) do rng
        view(out, rng) .= f.(view(a, rng), view(b, rng))
    end
    return out
end

# `v .= f.(v)` computed blockwise in parallel, with the same requirements on `f`.
function elementwise!(f::F, v::AbstractVector, cost::Integer = 1) where {F}
    (eltype(v) <: NonThrowingElement && typeof(f(zero(eltype(v)))) === eltype(v) && parallel_worthwhile(length(v), cost * length(v))) ||
        return (v .= f.(v))
    foreach_block!(v, cost * length(v)) do rng
        view(v, rng) .= f.(view(v, rng))
    end
    return v
end
