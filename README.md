# HyperdimensionalComputing.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://KERMIT-UGent.github.io/HyperdimensionalComputing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://KERMIT-UGent.github.io/HyperdimensionalComputing.jl/dev)
[![Build Status](https://github.com/MichielStock/HyperdimensionalComputing.jl/workflows/CI/badge.svg)](https://github.com/MichielStock/HyperdimensionalComputing.jl/actions)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

<img src="/docs/src/assets/logo.png" align="right" style="padding-left:10px;" width="400"/>

This package implements special types of vectors and associated methods for hyperdimensional
computing/vector-symbolic architectures.

Hyperdimensional computing (HDC) is a paradigm to represent patterns by means of a
high-dimensional vectors (typically 10,000 dimensions). Specific operations can be used to
create new vectors by combining the information or encoding some kind of position. HDC is an
alternative machine learning method that is extremely computationally efficient. It is inspired
by the distributed, holographic representation of patterns in the brain. Typically, the
high-dimensionality is more important than the nature of the operations. This package provides
various types of vectors (binary, graded, bipolar...) with sensible operations for
*aggregating*, *binding* and *permutation*. Basic functionality for fitting a k-NN like
classifier is also supported.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Installation

The package can be installed using Pkg.jl as follows:

```julia
using Pkg; Pkg.add(url = "https://github.com/Kermit-UGent/HyperdimensionalComputing.jl")
```

or in the package mode (by pressing `]`):

```julia-repl
]add https://github.com/Kermit-UGent/HyperdimensionalComputing.jl#main
```

## Usage

### Creating hypervectors

Several vector symbolic architectures are implemented (see `?AbstractHV` for all subtypes).
They all share the same constructor convention:

```julia
using HyperdimensionalComputing

x = BinaryHV()                                    # fresh random hypervector, 10,000 dimensions
y = BipolarHV(; D = 64)                           # dimensionality is set with the keyword D
z = TernaryHV([1, 1, -1, 0, 0, 0, 1, 1, -1, 0])   # wrap an existing vector
```

The positional argument is reserved for *the object you want to encode*: `HV(this)` returns
the deterministic hypervector representing `this`, seeded by `hash(this)`. Any token — a
symbol, string, character, number — gets its own reproducible, quasi-orthogonal hypervector:

```julia
julia> cat = BipolarHV(:cat)
10000-element BipolarHV with 5078 positives and 4922 negatives:
 -1
 -1
 -1
  ⋮
 -1
  1

julia> cat == BipolarHV(:cat)  # the same object always yields the same hypervector
true

julia> similarity(cat, BipolarHV(:dog))  # different objects are quasi-orthogonal
0.001
```

> [!IMPORTANT]
> The positional argument is never a dimension: `BinaryHV(6)` is the 10,000-dimensional
> hypervector *encoding the number 6*. Use `BinaryHV(; D = 6)` to set the dimensionality.

### Operations

Hypervectors can be combined to represent more complex structures. The basic operations are
`bundle` (creating a vector that is similar to the provided vectors), `bind` (creating a vector
that is dissimilar to the vectors) and `shift` (cyclically shifting the vector, used to encode
position). For `bundle` and `bind`, we overload `+` and `*` as binary operators, while `ρ`
(`\rho`) is an alias for `shift`. Each VSA uses its own implementation of these operations.

```julia
julia> x, y, z = GradedHV(; D = 5), GradedHV(; D = 5), GradedHV(; D = 5);

julia> bundle([x, y, z])
5-element GradedHV{Float64} with μ ± σ = 0.786 ± 0.435:
 0.9980386053693185
 0.9994897128289538
 0.9696790867890732
 0.008871444428233321
 0.9548707362741092

julia> x + y + z == bundle([x, y, z])
true

julia> bind([x, y, z])
5-element GradedHV{Float64} with μ ± σ = 0.536 ± 0.232:
 0.5340650961987313
 0.3071283370775813
 0.5324987246729835
 0.9135871730556507
 0.3929268002269075

julia> x * y * z == bind([x, y, z])
true

julia> shift(x, 2)
5-element GradedHV{Float64} with μ ± σ = 0.899 ± 0.157:
 0.9857814092925962
 0.9345994482275566
 0.9844262541167156
 0.9709891727120051
 0.6206103652316713

julia> ρ(x, 2) == shift(x, 2)
true
```

In-place variants `shift!`, `ρ!`, `perturbate!` and `normalize!` are also available.

Additionally, we provide common encoder strategies for different data structures:

- `multiset`
- `multibind`
- `bundlesequence`
- `bindsequence`
- `hashtable`
- `crossproduct`
- `ngrams`
- `graph`
- `level`

Finally, the `similarity` function can be used to compare two hypervectors, by default using
the best similarity metric for the hypervector type:

```
julia> a = GradedBipolarHV(:a);

julia> b = GradedBipolarHV(:b);

julia> c = a + b;  # bundling preserves similarity to the inputs

julia> similarity(a, b)
0.007006016597693629

julia> similarity(a, c)
0.6740992305784635

julia> similarity(b, c)
0.6824065675304283
```

For more information, refer to the documentation.

## Support

Please [open an issue](https://github.com/KERMIT-UGent/HyperdimensionalComputing.jl/issues/new) for
support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add
commits, and [open a pull request](https://github.com/KERMIT-UGent/HyperdimensionalComputing.jl/compare/).
