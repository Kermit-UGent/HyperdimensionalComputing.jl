```@meta
CurrentModule = HyperdimensionalComputing
```

# API Reference

This page contains the complete API reference for HyperdimensionalComputing.jl.

## Types

```@docs
AbstractHV
BinaryHV
BipolarHV
TernaryHV
GradedHV
GradedBipolarHV
RealHV
FHRR
```

## Operations

```@docs
bundle
bind
unbind
shift!
shift
ρ
ρ!
normalize
normalize!
perturbate
perturbate!
```

## Inference

```@docs
similarity
δ
nearest_neighbor
```

## Encoders

The package comes with a selected set of encoders for general computing and prototyping
purposes:

```@docs
multiset
multibind
bundlesequence
bindsequence
hashtable
crossproduct
ngrams
graph
level
encodelevel
decodelevel
convertlevel
```

Additionally, we provide an `encode` function and the `AbstractEncoding` type for implementing
more advances encoding strategies:

```@docs
encode
AbstractEncoding
BagOfSymbols
Sequence
NGram
KMer
```

## Package extensions

`HyperdimensionalComputing.jl` has a couple of package extensions to interact with commonly used
Julia packages:

### [UnicodePlots.jl](https://juliaplots.org/UnicodePlots.jl/stable/)

```@docs
unicodeheatmap
unicodehistogram
```
