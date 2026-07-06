# HiperdimensionalComputing.jl

Hyperdimensional computing in Julia

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://michielstock.github.io/HyperdimensionalComputing.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://michielstock.github.io/HyperdimensionalComputing.jl/dev)
[![Build Status](https://github.com/MichielStock/HyperdimensionalComputing.jl/workflows/CI/badge.svg)](https://github.com/MichielStock/HyperdimensionalComputing.jl/actions)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

This package implements special types of vectors and associated methods for hyperdimensional
computing.

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
using Pkg; Pkg.add(url = "https://github.com/MichielStock/HyperdimensionalComputing.jl")
```

or in the package mode (by pressing `]`):

```julia-repl
]add https://github.com/MichielStock/HyperdimensionalComputing.jl#main
```

## Usage

Several vector symbolic architectures are implemented (see `?AbstractHV` for all subtypes),
having different interfaces for hypervector creation:

```julia
using HyperdimensionalComputing

julia> x = BinaryHV() # default length is 10,000
10000-element BinaryHV:
 1
 0
 0
 0
 0
 1
 1
 0
 ⋮
 1
 0
 1
 0
 1
 0
 0
 0

julia> y = BipolarHV(6) # different size
6-element BipolarHV:
  1
 -1
  1
  1
 -1
  1

julia> z = TernaryHV([1, 1, -1, 0, 0, 0, 1, 1, -1, 0]) # From a vector
10-element TernaryHV:
  1
  1
 -1
  0
  0
  0
  1
  1
 -1
  0
```

Hypervectors can be combined to represent more complex structures.  The basic operations are
`bundle` (creating a vector that is similar to the provided vectors), `bind` (creating a vector
that is dissimilar to the vectors) and `shift` (shifting the vector to create a new vector). For
`aggregate` and `bind`, we overload `+` and `*` as binary operators, while `ρ` (`\rho`) is an
alias for `shift`. For each VSA

```julia
julia> x, y, z = GradedHV(10), GradedHV(10), GradedHV(10);

julia> bundle([x,y,z])
10-element GradedHV{Float64}:
 0.24160752324192117
 0.0004620105852614061
 0.21122703146393468
 0.8806160209097325
 0.748086047467331
 0.29234791431258145
 0.2804922831134219
 0.18556141274368268
 0.08208462507331278
 0.9015873952761569

julia> x + y + z
10-element GradedHV{Float64}:
 0.24160752324192117
 0.0004620105852614061
 0.21122703146393468
 0.8806160209097325
 0.748086047467331
 0.29234791431258145
 0.2804922831134219
 0.18556141274368268
 0.08208462507331278
 0.9015873952761569

julia> bind([x, y, z])
10-element GradedHV{Float64}:
 0.5061620120454368
 0.26070792597812487
 0.513025014409134
 0.4717013369896599
 0.49961155414024105
 0.46309236900588013
 0.5006006610486007
 0.533210817253628
 0.529186380757248
 0.46622954535393824

julia> x * y * z
10-element GradedHV{Float64}:
 0.5061620120454368
 0.26070792597812487
 0.513025014409134
 0.4717013369896599
 0.49961155414024105
 0.46309236900588013
 0.5006006610486007
 0.533210817253628
 0.529186380757248
 0.46622954535393824

julia> shift(x, 2)
10-element GradedHV{Float64}:
 0.6572388961520694
 0.38592896770130286
 0.5732316268033676
 0.1280407117618328
 0.13125571831454053
 0.40139428175287556
 0.5286213065298765
 0.1038774285737532
 0.43575817327464744
 0.32479919832749704

julia> ρ(x, 2)
10-element GradedHV{Float64}:
 0.6572388961520694
 0.38592896770130286
 0.5732316268033676
 0.1280407117618328
 0.13125571831454053
 0.40139428175287556
 0.5286213065298765
 0.1038774285737532
 0.43575817327464744
 0.32479919832749704

julia> shift!(x, 2)
10-element Vector{Float64}:
 0.6572388961520694
 0.38592896770130286
 0.5732316268033676
 0.1280407117618328
 0.13125571831454053
 0.40139428175287556
 0.5286213065298765
 0.1038774285737532
 0.43575817327464744
 0.32479919832749704

julia> x
10-element GradedHV{Float64}:
 0.6572388961520694
 0.38592896770130286
 0.5732316268033676
 0.1280407117618328
 0.13125571831454053
 0.40139428175287556
 0.5286213065298765
 0.1038774285737532
 0.43575817327464744
 0.32479919832749704
```

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

Finally, `similarity` function can be used to compute two hypervectors, by default using the
best similarity metric for the hypervector type:

```
julia> a = GradedBipolarHV(10);

julia> b = GradedBipolarHV(10);

julia> c = a + b;

julia> similarity(a, b)
0.7857595020921353

julia> similarity(a, c)
0.9241024788902716

julia> similarity(b, c)
0.9295606930584789
```

For more information, refer to the documentation.

## Support

Please [open an issue](https://github.com/MichielStock/HyperdimensionalComputing.jl/issues/new) for
support.

## Contributing

Please contribute using [Github Flow](https://guides.github.com/introduction/flow/). Create a branch, add
commits, and [open a pull request](https://github.com/MichielStock/HyperdimensionalComputing.jl/compare/).
