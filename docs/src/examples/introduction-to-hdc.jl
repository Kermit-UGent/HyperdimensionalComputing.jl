# # Introduction
#
# Hyperdimensional Computing (HDC) is a brain-inspired computational paradigm that represents
# and manipulates information using high-dimensional vectors called **hypervectors**. These
# vectors typically have thousands of dimensions (often 1,000-10,000), making them
# "hyperdimensional."
#
# The key insight is that high-dimensional spaces have unique mathematical properties that allow
# for robust, fault-tolerant computation.
#
# Let's start by loading the package in question, as follows:

using HyperdimensionalComputing

# # Creating hypervectors
#
# First, we will create a random bipolar hypervector. This is done as follows:

BipolarHV()

# As you may see, by default the hypervector created has 10.000 dimensions. This is the default
# value in `HyperdimensionalComputing.jl`, but one can can create a hypervector of any given
# dimensionality by providing the size of this as an argument:

BipolarHV(; D = 8)

# Alternatively, one can create a hypervector directly from a `Vector{T}` where `{T}` is an
# appropiate data type, e.g. integers for BipolarHV:

BipolarHV(rand((-1, 1), 8))

# or you can directly pass any Julia structure to use it as a seed for the hypervector
# generation:

BipolarHV(:foo)

# Let's create 3 bipolar hypervector to use for the tutorial:

h₁ = BipolarHV(; D = 8)
h₂ = BipolarHV(; D = 8)
h₃ = BipolarHV(; D = 8);


# The package has different hypervector types, such as `BipolarHV`, `TernaryHV`, `RealHV`,
# `GradedBipolarHV`, and `GradedHV`. All of this hypervectors have a common abstract type
# `AbstractHV` which can be used to build additional functions or encoding strategies (more on
# both later).
#
# !!! info "On (abstract) types"
#     All hypervectors implemented on `HyperdimensionalComputing.jl` can be found by checking the
#     docstrings for the `AbstractHV` (by typing `?AbstractHV` on the Julia REPL).
#
#     For more information on a specific hypervector type, the docstrings contain information on
#     the implementation, operations, similarity measurement and other technical
#     characteristics.
#
# # Fundamental operations with hypervectors
#
# HDC uses three primary operations that preserve the hyperdimensional properties and allow for
# the representation more complex structures:
#
# ## Bundling
#
# Bundling (also known as superposition) combines multiple hypervectors to create a new hypervector
# that is similar to it's constituents.
#
# $$u = [h_1 + h_2 + h_3]$$
#
# where $[...]$ denotes a potential normalization operations. In the case of bipolar
# hypervectors, this normalization operation is the `sign` function, which is defined as
# follows:
#
# $$\text{sign}(i) = \begin{cases}
#   +1 & \text{if } i > 0 \\
#   -1 & \text{if } i < 0 \\
#    0 & \text{otherwise }
# \end{cases}$$
#
# In HyperdimensionalComputing.jl, you can bundle hypervectors as follows:

bundle([h₁, h₂, h₃])

# alternatively, you can use the `+` operator (which if overloaded for all `AbstractHV`):

h₁ + h₂ + h₃

# This operation generates a hypervector that is similar to all it's constituent hypervectors,
# such that
#
# $$h₁ \sim u, h₂ \sim u, h₃ \sim u$$
#
# where $\sim$ means that the hypervectors are similar, i.e. they share more components than
# expected by chance.
#
# ## Binding
#
# Binding combines multiple hypervectors to create a new hypervector that is dissimilar to it's
# constituents, such that:
#
# $$v = [h₁ \times h₂ \times h₃]$$
#
# where $[...]$ represents a normalization procedure.
#
# In HyperdimensionalComputing.jl, you can bind hypervectors as follows:

bind([h₁, h₂, h₃])

# alternatively, you can use the `*` operator (which if overloaded for all `AbstractHV`):

h₁ * h₂ * h₃

# This operation generates a hypervector that is similar to all it's constituent hypervectors,
# such that
#
# $$h₁ \nsim v, h₂ \nsim v, h₃ \nsim v$$
#
# where $\nsim$ means that the hypervectors are dissimilar, i.e. they are quasi-orthogonal.
#
# ## Permutation
#
# Permutation (also known as shifting) is a special case of binding that creates a variant of a
# single hypervector via, generally speaking, a circular vector shifting with one or more
# positions.
#
# $$m = \rho(h₁)$$

h₄ = TernaryHV(collect(0:9))
h₄.v

#

ρ(h₄).v

#
# The new hypervector will be, in principle, dissimilar to it's original version, such that:
#
# $$h_1 \nsim \rho(h_1) \nsim \rho\rho(h_1) \nsim \rho\rho(h_1) ...$$
#
# where $\nsim$ means that the hypervectors are dissimilar, i.e. they are quasi-orthogonal.
#
# In `HyperdimensionalComputing.jl`, one can shift hypervector as follows:

ρ(h₁, 1)

#

h₁ != ρ(h₁, 1) != ρ(h₁, 2) != ρ(h₁, 3)

# ## Similarity
#
# Althought technically not an operation, in order to retrieve information from hypervectors,
# we need to compare them using similarity/distance functions. `HyperdimensionalComputing.jl`
# provides a handy `similarity` function that accepts:
#
# 2 hypervectors:

similarity(h₁, h₂)

# A vector of hypervectors:

similarity(h₁, h₁)

# or a hypervector and a vector of hypervectors:

similarity.(Ref(h₁), [h₁, h₂, h₃])

# `δ` is a synonim of `similarity`, and can also be used to create a function for similarity
# comparison, e.g.

f = δ(h₁)

#

f.([h₁, h₂, h₃])


# ## Encoding things as hypervectors
#
# The true power of HDC emerges when we combine the fundamental operations to encode complex data
# structures as hypervectors. By creatively applying bundling, binding, and shifting, we can
# represent virtually any type of information - from sequences and hierarchies to graphs and
# associative memories. The operations act as building blocks that can be composed in countless
# ways, limited only by our imagination and the specific requirements of our application. Let's
# explore some fundamental encoding strategies that demonstrate this flexibility.
#
# ### Key-value pairs
#
# Animal hypervectors:

H_dog = TernaryHV(:dog)
H_cat = TernaryHV(:cat)
H_cow = TernaryHV(:cow)
H_animals = [H_dog, H_cat, H_cow]

# Sound hypervectors:

H_bark = TernaryHV(:bark)
H_meow = TernaryHV(:meow)
H_moo = TernaryHV(:moo)
H_sounds = [H_bark, H_meow, H_moo]


# Associative memory:

memory = (H_dog * H_bark) + (H_cat * H_meow) + (H_cow * H_moo);

# !!! note
#	  Alternatively you can use the `hashtable` encoder to achieve the same:

memory == hashtable(H_animals, H_sounds)

# Querying memory to search for dog's sound:

nearest_neighbor(H_dog * memory, H_sounds)

# Querying memory to search which animals go "moo":

nearest_neighbor(H_moo * memory, H_animals)

# This is a very simple example, but you could think of having a more complex thing going on or
# having more animals that, for example, share sounds.
#
# ### Sequences
#
# **N-grams** represent sequences by encoding the order of elements. This is particularly useful for text processing where word order matters.
#
# Encode the phrases using the builtin `ngrams` encoder, with uses a sliding window of 3
# characters.
#
# Let's encode some phrases and then search for a specific word in them. First, the sentences
# list:

phrases = [
    "the quick brown fox jumps over the lazy dog",
    "the slick grown box bumps under the hazy fog",
    "the thick known cox dumps inter the crazy cog",
    "the brick shown pox lumps enter the glazy jog",
    "the stick blown sox pumps winter the blazy log",
];

# Now, lets encode sentences using the characters as seed for our basis hypervectors and use n-gram
# encoding to represent the sentences as hypervectors:

encode(p::String) = map(c -> BinaryHV(c), collect(p)) |> ngrams

#

H_phrases = map(encode, phrases)

# Now that we have the sentence hypervectors, let's search for "crazy" in phrases:

query = map(c -> BinaryHV(c), collect("crazy")) |> ngrams

#

nearest_neighbor(query, H_phrases)

# Great! We correctly found that "crazy" is in phrase 3.
