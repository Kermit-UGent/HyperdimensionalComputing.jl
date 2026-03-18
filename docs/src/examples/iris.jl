# # Iris dataset


using HyperdimensionalComputing
using MLDatasets, DataFrames
using AlgebraOfGraphics, CairoMakie

# Load Iris dataset
X, y = Iris(as_df = false)[:]

# ## Encoding
#
# The first thing we need to do in order to work with HDC is to encode our problem into
# hyperdimensional space. For that, we need to propose an encoder that converts our objects in
# real space (denoted as R) into hypervectors representing the hyperdimensional space (denoted
# as H).
#
# In this simple, classical example, we will employ the key-value encoder, which
#
# Encode feature names as hypervectors
SEPALLENGTH = BinaryHV()
SEPALWIDTH = BinaryHV()
PETALLENGTH = BinaryHV()
PETALWIDTH = BinaryHV()
H_features = [SEPALLENGTH, SEPALWIDTH, PETALLENGTH, PETALWIDTH]

# Encode values names as level hypervectors
cm = range(extrema(X)...; step = 0.1)
H_cm = level(BinaryHV, length(cm))
cm2hv = encodelevel(H_cm, cm)

# Encode entries as hypervectors using the hash-table encoder
encode(features::AbstractVector{Float64}) = hashtable(cm2hv.(features), H_features)

# !!! info "On encoders"
#     The package contains a series of built-in encoders that can be useful for prototyping or
#     training small models. For more information on the available encoders in HDC.jl, please
#     refer to the [API reference]().
H_allflowers = map(encode, eachcol(X))

# ## Decoding
#
# Once we have our hypervectors, we can use the same operations to decode this back into the
# original space. For this, we exploit the properties of HDC and make use of approximatations to
# obtain the original elements.
#
# Let's start selecting a random entry from the dataset

H_flower = H_allflowers[rand(1:size(X, 2))]

# From here, we can obtain the approximate hypervectors for the features that constitute it
# via unbinding:

H_flower * SEPALWIDTH
H_flower * SEPALLENGTH
H_flower * PETALWIDTH
H_flower * PETALLENGTH

# Let's implement a decoder based on unbinding and closest match:

hv2cm = decodelevel(H_cm, cm)
decode(hv) = Ref(hv) .* H_features .|> hv2cm
decode(H_flower)

# !!! info "On level encoders"
#     Instead of creating a level encoder and decoder separately, we also provide the `convertlevel`
#     function to create both with a single call, which makes more sense for the vast majority
#     of applications.


# ## Training a small model
#
# First, let's check

H_setosa = bundle(H_allflowers[vec(y) .== "Iris-setosa"])
H_versicolor = bundle(H_allflowers[vec(y) .== "Iris-versicolor"])
H_virginica = bundle(H_allflowers[vec(y) .== "Iris-virginica"])

# Let's decode each of this representations and compare them against the mean values of each
# class:

[decode(H_setosa)'; mean(X[:, vec(y) .== "Iris-setosa"], dims = 2)']

#

[decode(H_versicolor)'; mean(X[:, vec(y) .== "Iris-versicolor"], dims = 2)']

#

[decode(H_virginica)'; mean(X[:, vec(y) .== "Iris-virginica"], dims = 2)']

# Pretty close! Let's evaluate our model properly. First, we will create a training/test split
# and regenerate the prototype hypervectors for each type of flower:

split = 0.8
test = rand(length(y)) .> split
train = .! test

# We will generate the prototype hypervectors just as before, but we will only use the training
# set:

H_setosa = bundle(H_allflowers[(vec(y) .== "Iris-setosa") .&& train])
H_versicolor = bundle(H_allflowers[(vec(y) .== "Iris-versicolor") .&& train])
H_virginica = bundle(H_allflowers[(vec(y) .== "Iris-virginica") .&& train])
H_prototypes = [H_setosa, H_versicolor, H_virginica]

# Let's predict the classes using nearest neighbor:
id2class = unique(y)
correct = map(findall(test)) do i
    H_test = H_allflowers[i]
    ytrue = y[i]
    ypred = nearest_neighbor(H_test, H_prototypes)
    ytrue == id2class[ypred[2]]
end |> sum
accuracy = correct / sum(test)

# Great! we have trained a model using HDC and it's pretty good!
#
# ### Data diet in HDC
#
# One of the intersting properties of HDC is that we can train models with small amounts or even
# a single training data point. Let's test this property by repeating the previous experiment
# but this time using different training splits ranging from 5% (around 3 data points per class)
# to 95% (around 48 data points per class).
#
# First, let's define a small function that trains our model and evaluates it based on a given
# number of training points:

function traintest(n)
    # Generate splits
    # n = trunc(Int, split * 50)
    train = zeros(Bool, 150)
    train[[rand(1:50, n); rand(51:100, n); rand(101:150, n)]] .= true
    test = .! train

    # Construct prototypes
    H_setosa = bundle(H_allflowers[(vec(y) .== "Iris-setosa") .&& train])
    H_versicolor = bundle(H_allflowers[(vec(y) .== "Iris-versicolor") .&& train])
    H_virginica = bundle(H_allflowers[(vec(y) .== "Iris-virginica") .&& train])
    H_prototypes = [H_setosa, H_versicolor, H_virginica]

    # Predict and evaluate
    id2class = unique(y)
    correct = map(findall(test)) do i
        H_flower = H_allflowers[i]
        ytrue = y[i]
        ypred = nearest_neighbor(H_flower, H_prototypes)
        ytrue == id2class[ypred[2]]
    end |> sum
    return correct / sum(test)
end

# Now, let's run this training - testing workflow over multiple splits, repeating this 100 times
# to get a performance distribution per split:

results = Dict("points" => [], "accuracy" => [])
for split in 1:49
    for n in 1:100
        acc = traintest(split)
        push!(results["points"], split)
        push!(results["accuracy"], acc)
    end
end

draw(
    data(results)
        * mapping(:points => "Training points", :accuracy => "Accuracy ↑")
        * visual(BoxPlot, color = :gainsboro, width = 1)
    , axis = (; aspect = 1.5, limits = (0, 50, 0.5, 1.05), xticks = 0:5:50, yticks = 0.5:0.1:1.0)
)

# As you see here, HDC is capable of few-shot learning, achieving performance level similar to
# model trained with most of the data.
