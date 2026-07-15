# # Predictive modelling with HDC: Iris dataset
#
# This example shows an end-to-end, classical machine-learning workflow built entirely out of
# hyperdimensional computing (HDC) primitives: we encode the numeric [Iris
# dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set) into hypervectors, decode them back
# to check the encoding is faithful, and then train and evaluate a tiny nearest-prototype
# classifier. Along the way we highlight the specific `HyperdimensionalComputing.jl` functions that
# do the heavy lifting.
#
# If you have not seen the core operations (mapping, bundling, binding) yet, the
# *"Introduction to HDC"* tutorial is a gentler starting point.

using HyperdimensionalComputing
using MLDatasets, DataFrames
using Statistics
using AlgebraOfGraphics, CairoMakie

# The dataset ships with `MLDatasets`. We ask for the raw arrays (`as_df = false`) rather than a
# `DataFrame`: `X` is a `4 × 150` matrix of measurements (sepal length/width, petal length/width,
# in centimetres) and `y` holds the species label of each of the 150 flowers.

X, y = Iris(as_df = false)[:]

# ## Encoding
#
# The first thing we need to do in order to work with HDC is to encode our problem into
# hyperdimensional space. For that, we need an *encoder* that converts our objects in real space
# (denoted $\mathbb{R}$) into hypervectors representing the hyperdimensional space (denoted
# $\mathbb{H}$).
#
# In this classical example we build a **key–value (hash-table) encoder**: each flower is a record
# whose *keys* are the four feature names and whose *values* are the measured numbers. We bind each
# value to its key and bundle the four pairs into a single hypervector -- exactly what the
# [`hashtable`](@ref) encoder does.
#
# First, we map each feature *name* to a random hypervector that will act as its key:

SEPALLENGTH = BinaryHV()
SEPALWIDTH = BinaryHV()
PETALLENGTH = BinaryHV()
PETALWIDTH = BinaryHV()
H_features = [SEPALLENGTH, SEPALWIDTH, PETALLENGTH, PETALWIDTH]

# The *values* are continuous numbers, so a purely random mapping would throw away their ordering
# (5.0 cm and 5.1 cm would be as unrelated as 5.0 cm and 100 cm). Instead we use a **level
# encoder**: [`level`](@ref) builds a ladder of hypervectors in which neighbouring levels are
# similar and far-apart levels are dissimilar, so numeric closeness becomes hypervector similarity.
# We lay out one level every 0.1 cm across the observed range:

cm = range(extrema(X)...; step = 0.1)
#
H_cm = level(BinaryHV, length(cm))

# [`encodelevel`](@ref) turns that ladder into a ready-to-use function mapping any number to its
# closest level hypervector:

cm2hv = encodelevel(H_cm, cm)

# We can now encode a single flower: bind each feature value to its key and bundle the pairs with
# [`hashtable`](@ref).

encode(features::AbstractVector{Float64}) = hashtable(cm2hv.(features), H_features)

# !!! info "On encoders"
#     `hashtable` and `level` are only two of the built-in encoders. The package ships several
#     more (`multiset`, `bundlesequence`, `ngrams`, `graph`, ...) for prototyping or training small
#     models. For the full catalogue, see the [API reference](../api.md).
#
# Applying it to every column encodes the whole dataset -- 150 flowers, each now a single
# hypervector:

H_allflowers = map(encode, eachcol(X))

# ## Decoding
#
# Once we have our hypervectors, we can use the same operations to decode them back into the
# original space. Here we exploit the fact that binding is its own inverse: **unbinding** a flower
# with a feature key recovers (an approximation of) the value hypervector for that feature.
#
# Let's pick a random flower from the dataset:

H_flower = H_allflowers[rand(1:size(X, 2))]

# Unbinding it with a key yields the (noisy) level hypervector for that feature -- for example, its
# sepal length:

H_flower * SEPALLENGTH

# That hypervector is not exactly any level, but it is *closest* to the right one. The counterpart
# of `encodelevel` is [`decodelevel`](@ref), which builds a decoder that snaps a hypervector back
# to the numeric level it most resembles:

hv2cm = decodelevel(H_cm, cm)

# Putting it together, a decoder for a whole flower unbinds every feature and reads off its value:

decode(hv) = Ref(hv) .* H_features .|> hv2cm
decode(H_flower)

# Compare that with the flower's true measurements -- the round-trip through $\mathbb{H}$ recovers
# them up to the 0.1 cm resolution of our level ladder:

X[:, rand(1:size(X, 2))]  # a real measurement vector, for reference on scale

# !!! tip "One-shot level codecs with `convertlevel`"
#     Instead of creating the encoder and decoder separately, [`convertlevel`](@ref) returns both
#     with a single call, which is handier for the vast majority of applications:
#     `cm2hv, hv2cm = convertlevel(H_cm, cm)`.
#
# ## Training a small model
#
# We can now build a classifier. The idea is beautifully simple: a class is represented by the
# **bundle** (superposition) of all its training examples, giving a single *prototype* hypervector
# that sits close to every flower of that species. `bundle` is the bundling operation from the
# introduction.

H_setosa = bundle(H_allflowers[vec(y) .== "Iris-setosa"])
H_versicolor = bundle(H_allflowers[vec(y) .== "Iris-versicolor"])
H_virginica = bundle(H_allflowers[vec(y) .== "Iris-virginica"])

# As a sanity check, let's decode each prototype and compare it against the *mean* measurements of
# its class. The prototypes are not just abstract vectors -- decoding them recovers something very
# close to the class averages:

[decode(H_setosa)'; mean(X[:, vec(y) .== "Iris-setosa"], dims = 2)']

#

[decode(H_versicolor)'; mean(X[:, vec(y) .== "Iris-versicolor"], dims = 2)']

#

[decode(H_virginica)'; mean(X[:, vec(y) .== "Iris-virginica"], dims = 2)']

# Pretty close! Let's evaluate the model properly. First, we split the data into a training and a
# test set:

split = 0.8
test = rand(length(y)) .> split
train = .! test

# We regenerate the prototype hypervectors exactly as before, but using only the training flowers:

H_setosa = bundle(H_allflowers[(vec(y) .== "Iris-setosa") .&& train])
H_versicolor = bundle(H_allflowers[(vec(y) .== "Iris-versicolor") .&& train])
H_virginica = bundle(H_allflowers[(vec(y) .== "Iris-virginica") .&& train])
H_prototypes = [H_setosa, H_versicolor, H_virginica]

# To classify a flower we simply find the most similar prototype with [`nearest_neighbor`](@ref),
# which returns a `(similarity, index, hypervector)` tuple -- the index tells us which class won:

id2class = unique(y)
correct = map(findall(test)) do i
    H_test = H_allflowers[i]
    ytrue = y[i]
    ypred = nearest_neighbor(H_test, H_prototypes)
    ytrue == id2class[ypred[2]]
end |> sum
accuracy = correct / sum(test)

# Great -- we trained a classifier out of nothing but bundling and similarity, and it is already
# very accurate!
#
# ### Data diet in HDC
#
# One of the interesting properties of HDC is that we can train usable models from very little
# data -- sometimes even a single example per class. Let's probe this by repeating the experiment
# across training sizes ranging from 1 point per class (~2% of the data) up to 49 points per class
# (~98%).
#
# We define a helper that draws `n` training flowers per class, builds the prototypes, and returns
# the test accuracy. Note the in-body comments are written with `##` so `Literate.jl` keeps them
# inside the code block:

function traintest(n)
    ## Draw n training flowers per class and use the rest as test set
    train = zeros(Bool, 150)
    train[[rand(1:50, n); rand(51:100, n); rand(101:150, n)]] .= true
    test = .! train

    ## Construct one prototype per class from the training flowers
    H_setosa = bundle(H_allflowers[(vec(y) .== "Iris-setosa") .&& train])
    H_versicolor = bundle(H_allflowers[(vec(y) .== "Iris-versicolor") .&& train])
    H_virginica = bundle(H_allflowers[(vec(y) .== "Iris-virginica") .&& train])
    H_prototypes = [H_setosa, H_versicolor, H_virginica]

    ## Predict every test flower and return the accuracy
    id2class = unique(y)
    correct = map(findall(test)) do i
        H_flower = H_allflowers[i]
        ytrue = y[i]
        ypred = nearest_neighbor(H_flower, H_prototypes)
        ytrue == id2class[ypred[2]]
    end |> sum
    return correct / sum(test)
end

# Now we run this train/test workflow over the range of training sizes, repeating each 100 times to
# get a performance distribution:

results = Dict("points" => Int[], "accuracy" => Float64[])
for points in 1:49
    for _ in 1:100
        push!(results["points"], points)
        push!(results["accuracy"], traintest(points))
    end
end

draw(
    data(results)
        * mapping(:points => "Training points per class", :accuracy => "Accuracy ↑")
        * visual(BoxPlot, color = :gainsboro, width = 1)
    , axis = (; aspect = 1.5, limits = (0, 50, 0.5, 1.05), xticks = 0:5:50, yticks = 0.5:0.1:1.0)
)

# As the plot shows, HDC is capable of few-shot learning: even a handful of examples per class
# gets us close to the accuracy reached with the full training set.
