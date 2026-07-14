```@meta
EditURL = "whats-the-dollar-of-mexico.jl"
```

# Replicating "What’s the Dollar of Mexico?" in `HyperdimensionalComputing.jl`

Kanerva showcases in this publication how we can create prototype concepts and mappings between
them using Hyperdimensional Computing (HDC), a computing paradigm based on very big random vectors
populated with binary values.

This very big random vectors, which we calls "words" or "codes", represent concepts, but when
analyzing this as mathematical constructs they are nothing else than random signals. When
combined using bundling and binding operations, one can construct concepts and reason based on
the similarity of this words, always assuming that this retrieval is approximated.

Since the properties of the operations that enable composability of this "codes" are
mathematically grounded, the characteristics of the operations permeate into this modelling
mindset and enable for the creation of (i) prototype representations and (ii) mapping function,
both based on the same underlying mathematical structure. Thanks to this, we can either model
mapping one concept into another space or directly map a mapping function from one space into
another, basically mimicking how the brain works:

> The literal image created by the words is intended to be transferred to and interpreted in a
> new context, exemplifying the mind's reliance on prototypes: the literal meaning provides the
> prototype. When the mind expands its scope in this way it creates an incredibly rich web of
> associations and meaning...

To showcase this, we will reimplement the example proposed by Kanerva on this paper.

````@example whats-the-dollar-of-mexico
using HyperdimensionalComputing
````

In HDC, concepts are represented as high-dimensional random vectors (hypervectors).
Here, we create hypervectors for the abstract concepts of COUNTRY, CAPITAL, and MONEY.

````@example whats-the-dollar-of-mexico
COUNTRY = BinaryHV(:country)
CAPITAL = BinaryHV(:capital)
MONEY = BinaryHV(:money)

[COUNTRY, CAPITAL, MONEY]
````

Next, we create hypervectors for specific instances: countries, their capitals, and currencies.

````@example whats-the-dollar-of-mexico
USA = BinaryHV(:usa)
MEX = BinaryHV(:mexico)

WDC = BinaryHV(:wdc)      # Washington DC
MXC = BinaryHV(:mxc)      # Mexico City

DOL = BinaryHV(:dollar)
PES = BinaryHV(:peso)

[USA, MEX, WDC, MXC, DOL, PES]
````

We now build holistic representations for the United States and Mexico by binding each concept
(country, capital, money) to its specific instance and then adding them together.

````@example whats-the-dollar-of-mexico
USTATES = (COUNTRY * USA) + (CAPITAL * WDC) + (MONEY * DOL)
MEXICO = (COUNTRY * MEX) + (CAPITAL * MXC) + (MONEY * PES)

[USTATES, MEXICO]
````

These composite hypervectors encode all the information about each country in a single vector.
The `Base.isapprox` function or `≈` operator checks if two hypervectors are approximately equal
(i.e., similar).

````@example whats-the-dollar-of-mexico
USTATES ≈ MEXICO
````

By binding (represented by the `*` operator) the holistic representations of USA and Mexico, we
create a new hypervector that encodes the relationships between their respective elements: USA
with Mexico, Washington DC with Mexico City, and dollar with peso, plus some noise due to the
high-dimensional operations.

````@example whats-the-dollar-of-mexico
F_UM = USTATES * MEXICO
````

````@example whats-the-dollar-of-mexico
F_UM ≈ (USA * MEX) + (WDC * MXC) + (DOL * PES)
````

To answer Kanerva's question "What in Mexico corresponds to United States' dollar?", we unbind
(represented again by the `*` operator) the dollar hypervector with the USA-Mexico relationship
vector. The result should be similar to the peso hypervector.

````@example whats-the-dollar-of-mexico
DOL * F_UM ≈ PES
````

Let's add another country, Sweden, with its capital and currency.

````@example whats-the-dollar-of-mexico
SWE = BinaryHV(:sweden)
STO = BinaryHV(:stockholm)
SEK = BinaryHV(:krona)

SWEDEN = (COUNTRY * SWE) + (CAPITAL * STO) + (MONEY * SEK)
````

We can now explore more complex relationships by binding and combining these holistic representations.
For example, `F_SU`encodes the relationship between Sweden and USA, and F_SM between Sweden and Mexico.

````@example whats-the-dollar-of-mexico
F_UM = USTATES * MEXICO
F_SU = SWEDEN * USTATES
F_SM = SWEDEN * MEXICO
````

Combining these relationship vectors allows us to infer new relationships, such as how Sweden relates
to Mexico via the USA.

````@example whats-the-dollar-of-mexico
F_SU * F_UM ≈ F_SM
````

We can also directly query for corresponding elements between countries:

For example, what is the currency of Mexico, given the currency of the USA?

````@example whats-the-dollar-of-mexico
USTATES * DOL ≈ MEXICO * PES
````

Or, what is the currency of Mexico, given the USA and its currency?

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ PES
````

Similarly, we can query for Sweden's currency using the same approach.

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ SEK
````

Or, recover the original currency of the USA.

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ DOL
````

This tutorial demonstrates how hyperdimensional computing enables analogical reasoning and flexible
querying by representing and manipulating concepts as high-dimensional vectors.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

