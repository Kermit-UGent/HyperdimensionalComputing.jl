module UnicodePlotting

#=
Rich terminal display for hypervectors based on UnicodePlots.

This extension deliberately defines NO `Base.show` methods: the single `show`
method for `AbstractHV` lives in src/representations.jl and delegates to
`_show_rich` here when the extension is loaded. Defining the same `show`
signature in both places would be method overwriting, which is forbidden during
precompilation.
=#

using UnicodePlots
using HyperdimensionalComputing
using HyperdimensionalComputing: AbstractHV
import HyperdimensionalComputing: unicodeheatmap, unicodehistogram

# Values used for plotting: the element values, or the phases for FHRR.
plotvalues(hv::AbstractHV) = collect(hv)
plotvalues(hv::FHRR) = angle.(hv.v)

function unicodeheatmap(hv::AbstractHV)
    vals = plotvalues(hv)
    nsq = floor(Int, sqrt(length(vals)))
    return heatmap(reshape(vals[1:(nsq^2)], (nsq, nsq)))
end

unicodehistogram(hv::AbstractHV) = histogram(plotvalues(hv))

function unicodehistogram(hv::Union{BinaryHV, BipolarHV, TernaryHV})
    counts = Dict(string(x) => count(==(x), hv) for x in unique(collect(hv)))
    return barplot(counts)
end

# Called by `Base.show(io, ::MIME"text/plain", ::AbstractHV)` in
# src/representations.jl when this extension is loaded.
function _show_rich(io::IO, ::MIME"text/plain", hv::AbstractHV)
    println(io, summary(hv), ":")
    println(io, unicodehistogram(hv))
    return print(io, unicodeheatmap(hv))
end

end
