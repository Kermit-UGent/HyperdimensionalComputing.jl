module HyperdimensionalComputing

using Random
using Distributions
import LinearAlgebra
import LinearAlgebra: norm, dot

include("types.jl")
export AbstractHV,
    BinaryHV,
    BipolarHV,
    GradedBipolarHV,
    RealHV,
    GradedHV,
    TernaryHV,
    FHRR

include("representations.jl")
export unicodeheatmap,
    unicodehistogram

include("operations.jl")
export bundle,
    bind,
    unbind,
    shift!,
    shift,
    ρ,
    ρ!,
    perturbate,
    perturbate!,
    normalize,
    normalize!,
    norm,
    dot

include("encoding.jl")
export multiset,
    multibind,
    bundlesequence,
    bindsequence,
    hashtable,
    crossproduct,
    ngrams,
    graph,
    level,
    encodelevel,
    decodelevel,
    convertlevel

include("inference.jl")
export similarity,
    δ,
    nearest_neighbor


end
