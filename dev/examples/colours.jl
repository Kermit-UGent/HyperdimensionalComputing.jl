# # Colours: encoding continuous data with random projections
#
# The other tutorials encode *symbols* -- ingredients, characters, k-mers -- where the only
# question is whether two things are the same or different. Real data is often continuous: a
# measurement, an embedding, a pixel. Two colours can be *almost* the same, and we want the
# hypervectors to say so.
#
# The tool for this is [`RandomProjection`](@ref). It multiplies your `d`-dimensional input by a
# fixed random `D × d` matrix and applies a nonlinearity:
#
# ```math
# \mathbf{h} = f(R\,\mathbf{x}), \qquad R \in \mathbb{R}^{D \times d}
# ```
#
# The [Johnson–Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
# guarantees that such a projection approximately preserves relative distances, so *close inputs
# give similar hypervectors*. Colours are the perfect testbed: an RGB triple is just a point in
# $[0,1]^3$, and we already have strong intuitions about which colours are alike.

using HyperdimensionalComputing
using Colors, Random, Statistics, LinearAlgebra
Random.seed!(42);
nothing # hide

# ## Encoding colours
#
# The projection matrix is drawn once, at construction, and stored inside the encoder. That is
# what makes separately encoded colours comparable -- they all pass through the same `R`.

rp = RandomProjection(BipolarHV, 3; seed = 1)

# `Colors.RGB` is a struct, so we unpack it into a plain 3-vector before encoding:

col2vec(c) = [Float64(c.r), Float64(c.g), Float64(c.b)]
encodecolour(c) = encode(rp, col2vec(c))

# Let's take three colours -- two blue-greens and one orange:

teal = RGB(23 / 255, 146 / 255, 153 / 255)

#

sky = RGB(4 / 255, 165 / 255, 229 / 255)

#

orange = RGB(254 / 255, 100 / 255, 11 / 255)

# ## Comparing colours
#
# Encoding them and comparing gives exactly what our eyes say:

similarity(encodecolour(teal), encodecolour(sky))       # both blue-green

#

similarity(encodecolour(teal), encodecolour(orange))    # opposite ends

# Perceptual closeness has become hypervector similarity, and we never wrote a rule about
# colours. Compare this with the token path, where `teal` and `sky` would be two unrelated
# symbols with a similarity near zero.
#
# ## Blending by bundling
#
# Because the encoding is geometric, the HDC primitives now do geometric things. Bundling two
# colour hypervectors gives something *between* them, i.e., a blend:

v_blend = bundle([encodecolour(teal), encodecolour(orange)])
similarity(v_blend, encodecolour(teal)), similarity(v_blend, encodecolour(orange))

# The blend sits close to both parents, which is precisely the defining property of `bundle`.
# But what colour *is* it? To answer that we need to get back out of hyperspace.
#
# ## Decoding: clean-up, not inversion
#
# Here is an important limitation: the nonlinearity in
# ``f(R\mathbf{x})`` throws away magnitudes: `sign` maps a whole half-space to the same bit. A
# random projection therefore has **no analytic inverse**. [`decode`](@ref)
# performs *clean-up*, searching a set of reference hypervectors for the closest
# match. Calling it without references is an error.
#
# So we build a codebook: a few thousand random colours, encoded once.

reference_colours = [RGB(rand(), rand(), rand()) for _ in 1:2000]
reference_hvs = encodecolour.(reference_colours);

# `decode` returns a `(similarity, index, neighbour)` tuple, so we look the colour up by index:

decodecolour(hv) = reference_colours[decode(rp, hv, reference_hvs)[2]]

# Can we encode and decode to recover our original? Encode teal, decode it, and compare:

decodecolour(encodecolour(teal))

# Not bit-identical, it is the nearest of 2000 random colours, and it is lossy by construction.
# However, it is unmistakably the same colour. And now we can finally see the blend from before:

decodecolour(v_blend)

# ## Learning from data: the average colour of a concept
#
# Time for something less toy-like. Suppose we observe *categories* paired with colours (a
# label and an observation) and want to learn what colour each category "is". This is the
# classic HDC associative memory: **bind** each observation to its label, **bundle** everything
# into one hypervector, then **unbind** to query.
#
# We use the named colours that ship with `Colors.jl` as our data:

named(word) = [RGB((c ./ 255)...) for (n, c) in Colors.color_names if occursin(word, n)]
categories = Dict(:fire => named("red"), :water => named("blue"), :plant => named("green"))
length.(values(categories))

# Each observation becomes `label ⊗ colour`, and the whole dataset collapses into a *single*
# hypervector:

memory = bundle(
    [
        bind(encode(BipolarHV, label), encodecolour(c))
            for (label, colours) in categories for c in colours
    ]
)

# That one vector is the entire model. To ask "what colour is `:plant`?" we unbind the label and
# decode what remains:

decodecolour(memory * encode(BipolarHV, :plant))

#

decodecolour(memory * encode(BipolarHV, :fire))

#

decodecolour(memory * encode(BipolarHV, :water))

# Green, red and blue. The bundle averaged each category's colours, and unbinding pulled the
# right average back out from one hypervector holding all three categories at once.
#
# ## Robustness: signal buried in noise
#
# Real data is messier than that. Suppose each observation comes with *one* correct colour and
# *two* random distractors, and we are not told which is which. This is
# [multiple-instance learning](https://en.wikipedia.org/wiki/Multiple_instance_learning), and it
# is a natural fit for HDC: random hypervectors are quasi-orthogonal, so noise tends to cancel
# while the consistent signal reinforces.

noisy = bundle(
    [
        bind(
            encode(BipolarHV, label),
            bundle([encodecolour(c), encodecolour(RGB(rand(), rand(), rand())), encodecolour(RGB(rand(), rand(), rand()))])
        )
            for (label, colours) in categories for c in colours
    ]
)
decodecolour(noisy * encode(BipolarHV, :plant))

#

decodecolour(noisy * encode(BipolarHV, :fire))

# Still green and still red, with two-thirds of every observation being pure noise.
#
# ## Tuning the projection
#
# [`RandomProjection`](@ref) exposes the knobs that matter. The `matrix` keyword picks the
# distribution of `R`: `:gaussian` (the default), `:bipolar` (entries `±1`, cheap), and
# `:sparse_ternary` (mostly zeros, cheapest):

rp_bipolar = RandomProjection(BipolarHV, 3; matrix = :bipolar, seed = 1)
similarity(encode(rp_bipolar, col2vec(teal)), encode(rp_bipolar, col2vec(sky)))

# For the sign-based types, `θ` is the threshold applied after projection, and it controls the
# sparsity of the result. Rather than rebuilding the encoder to change it, [`rethreshold`](@ref)
# returns a new encoder that *shares the same matrix*, so the encodings stay comparable:

rp_shifted = rethreshold(rp, 0.5)
similarity(encode(rp_shifted, col2vec(teal)), encode(rp_shifted, col2vec(sky)))

# !!! tip "Mind your feature scales"
#     Colours are already on a common `[0, 1]` scale. Real feature vectors often are not, and a
#     projection is dominated by whichever feature has the largest spread -- so features measured
#     in grams and kilometres need to be put on a common footing first.
#
#     Standardising is not an automatic win, though. When features already share a unit *and*
#     their spread carries signal, equalising them throws information away; the *Iris dataset*
#     tutorial measures a case where standardising makes the classifier markedly worse.
#
# ## The kernel connection
#
# One combination is worth singling out. With [`FHRR`](@ref) hypervectors and a Gaussian
# matrix, `RandomProjection` is *exactly*
# [random Fourier features](https://en.wikipedia.org/wiki/Random_feature): the similarity between
# two encoded points approximates a Gaussian (RBF) kernel with bandwidth `β`,
#
# ```math
# \delta(\mathbf{h}_x, \mathbf{h}_y) \;\approx\; \exp\!\left(-\tfrac{1}{2}\beta^2\|\mathbf{x}-\mathbf{y}\|^2\right).
# ```
#
# We can check that directly. For random pairs of points, compare the measured hypervector
# similarity against the kernel value:

β = 1.5
rp_fhrr = RandomProjection(FHRR, 3; β = β, seed = 7, D = 20_000)

comparison = map(1:6) do _
    x, y = rand(3), rand(3)
    d = norm(x - y)
    (
        distance = round(d, digits = 3),
        hdc = round(similarity(encode(rp_fhrr, x), encode(rp_fhrr, y)), digits = 3),
        kernel = round(exp(-β^2 * d^2 / 2), digits = 3),
    )
end

# The two columns agree to about three decimals:

comparison

# This connects HDC to kernel methods: it gives an *explicit*, finite-dimensional
# feature map for a kernel that is normally only accessible implicitly through dot products.
#
# ## Sharpening a boundary
#
# Bundling all examples of a class into a prototype is the simplest HDC classifier, and it works
# well when classes are distinct. It struggles when they are not. Yellows and greens are
# neighbours in RGB space, so their prototypes sit close together:

yellows, greens = named("yellow"), named("green")
proto_yellow, proto_green = bundle(encodecolour.(yellows)), bundle(encodecolour.(greens))
similarity(proto_yellow, proto_green)

# Classifying by nearest prototype gets most, but not all, of them right:

prototype_accuracy = mean(
    vcat(
        [similarity(encodecolour(c), proto_yellow) > similarity(encodecolour(c), proto_green) for c in yellows],
        [similarity(encodecolour(c), proto_green) > similarity(encodecolour(c), proto_yellow) for c in greens],
    )
)

# We can do better by *retraining*: instead of averaging the examples once, iteratively nudge a
# weight vector whenever it misclassifies one. This is the classic
# [perceptron](https://en.wikipedia.org/wiki/Perceptron), and on hypervectors it is a few lines:

function perceptron(positives, negatives; α = 1, maxiter = 50)
    w = zeros(length(first(positives)))
    for _ in 1:maxiter
        errors = 0
        for v in positives
            if dot(w, collect(v)) <= 0
                w .+= α .* collect(v)
                errors += 1
            end
        end
        for v in negatives
            if dot(w, collect(v)) >= 0
                w .-= α .* collect(v)
                errors += 1
            end
        end
        errors == 0 && break
    end
    return w
end

w = perceptron(encodecolour.(yellows), encodecolour.(greens))

# The retrained boundary separates the two classes far more cleanly:

perceptron_accuracy = mean(
    vcat(
        [dot(w, collect(encodecolour(c))) > 0 for c in yellows],
        [dot(w, collect(encodecolour(c))) < 0 for c in greens],
    )
)

# !!! note "Retraining is not yet part of the package API"
#     The perceptron above is plain Julia written against the hypervectors, not a package
#     function. A proper `train`/`predict` workflow is planned; until then this is the pattern to
#     copy.
#
# ## Summary
#
# * [`RandomProjection`](@ref) turns continuous vectors into hypervectors while preserving
#   relative distances -- perceptual closeness becomes hypervector similarity.
# * It is **stateful**: the matrix is drawn once and shared, so only vectors encoded through the
#   same encoder are comparable. Use [`rethreshold`](@ref) to change `θ` while keeping `R`.
# * [`decode`](@ref) is **clean-up against a codebook**, not inversion -- the nonlinearity is
#   lossy by design.
# * Once data is in hyperspace, `bind`/`bundle`/`unbind` give you blending, associative memories
#   and noise-robust multi-instance learning for free.
# * With `FHRR` and a Gaussian matrix you get random Fourier features, i.e. an explicit feature
#   map for the Gaussian kernel.
