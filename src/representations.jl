#=
Pretty printing for hypervectors.

The plain path relies on Base's AbstractArray display machinery: we only provide
type-specific `Base.summary` headers and let Base handle element alignment and
`⋮` truncation. When the UnicodePlotting package extension is loaded (by loading
UnicodePlots), `show` delegates to the extension's rich display, unless the IO
context is `:compact`.
=#

# Type-specific headers; Base's array printing does the rest.
function Base.summary(io::IO, hv::BinaryHV)
    ntrue = count(hv.v)
    return print(io, length(hv), "-element ", typeof(hv), " with ", ntrue, " true and ", length(hv) - ntrue, " false")
end

function Base.summary(io::IO, hv::BipolarHV)
    # stored bit true ↦ -1, so count(hv.v) counts the NEGATIVES
    nneg = count(hv.v)
    return print(io, length(hv), "-element ", typeof(hv), " with ", length(hv) - nneg, " positives and ", nneg, " negatives")
end

function Base.summary(io::IO, hv::TernaryHV)
    npos = count(>(0), hv.v)
    nneg = count(<(0), hv.v)
    return print(io, length(hv), "-element ", typeof(hv), " with ", npos, " positives, ", length(hv) - npos - nneg, " zeros, and ", nneg, " negatives")
end

# μ ± σ is informative for real-valued elements; for FHRR (complex points on the
# unit circle) the mean/std carry no information, so FHRR intentionally has no
# summary override and gets Base's plain "N-element FHRR{...}" header.
function Base.summary(io::IO, hv::Union{RealHV, GradedHV, GradedBipolarHV})
    return print(io, length(hv), "-element ", typeof(hv), " with μ ± σ = ", round(mean(hv), digits = 3), " ± ", round(std(hv), digits = 3))
end

function Base.show(io::IO, mime::MIME"text/plain", hv::AbstractHV)
    ext = Base.get_extension(@__MODULE__, :UnicodePlotting)
    if ext === nothing || get(io, :compact, false)::Bool
        # standard Julia array display with the custom `summary` header
        return invoke(show, Tuple{IO, MIME"text/plain", AbstractArray}, io, mime, hv)
    else
        return ext._show_rich(io, mime, hv)
    end
end

# Vectors of hypervectors: one summary line per hypervector.
function Base.show(io::IO, ::MIME"text/plain", hvs::AbstractVector{<:AbstractHV})
    println(io, summary(hvs), ":")
    r = [" " * summary(hv) for hv in hvs]

    rows = displaysize(io)[1]
    if length(r) <= max(rows - 4, 0)
        return print(io, join(r, '\n'))
    end

    chunksize = max(rows ÷ 2 - 3, 0)
    if chunksize == 0
        return print(io, " ⋮")
    end
    return print(io, join([first(r, chunksize); " ⋮"; last(r, chunksize)], '\n'))
end

"""
    unicodeheatmap(hv::AbstractHV)

Render a hypervector as a square unicode heatmap of its leading `⌊√D⌋²` elements
(phases for [`FHRR`](@ref)).

Only available when UnicodePlots is loaded (`using UnicodePlots`); implemented in
the UnicodePlotting package extension.

See also [`unicodehistogram`](@ref).
"""
function unicodeheatmap end

"""
    unicodehistogram(hv::AbstractHV)

Render the distribution of a hypervector's elements as a unicode histogram
(a bar plot of counts for the discrete types [`BinaryHV`](@ref), [`BipolarHV`](@ref)
and [`TernaryHV`](@ref); phases for [`FHRR`](@ref)).

Only available when UnicodePlots is loaded (`using UnicodePlots`); implemented in
the UnicodePlotting package extension.

See also [`unicodeheatmap`](@ref).
"""
function unicodehistogram end
