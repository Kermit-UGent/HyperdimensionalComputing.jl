```@meta
EditURL = "whats-the-dollar-of-mexico.jl"
```

# Replicating "What’s the Dollar of Mexico?" in `HyperdimensionalComputing.jl`

````@example whats-the-dollar-of-mexico
using HyperdimensionalComputing
using LinearAlgebra #hide
````

Concept hypervectors

````@example whats-the-dollar-of-mexico
COUNTRY = BipolarHV()
CAPITAL = BipolarHV()
MONEY = BipolarHV()
````

Holistic representation of countries

````@example whats-the-dollar-of-mexico
USA = BipolarHV()
MEX = BipolarHV()

WDC = BipolarHV()
MXC = BipolarHV()

DOL = BipolarHV()
PES = BipolarHV()

USTATES = (COUNTRY * USA) + (CAPITAL * WDC) + (MONEY * DOL)
MEXICO = (COUNTRY * MEX) + (CAPITAL * MXC) + (MONEY * PES)
````

````@example whats-the-dollar-of-mexico
USTATES ≈ MEXICO
````

If we now pair USTATES with MEXICO, we get a bundle that pairs USA with Mexico, Washington DC
with Mexico City, and dollar with peso, plus noise:

````@example whats-the-dollar-of-mexico
F_UM = USTATES * MEXICO
````

What in Mexico corresponds to United States' dollar:

````@example whats-the-dollar-of-mexico
DOL * F_UM ≈ PES
````

````@example whats-the-dollar-of-mexico
SWE = BipolarHV()
STO = BipolarHV()
KRO = BipolarHV()

SWEDEN = (COUNTRY * SWE) + (CAPITAL * STO) + (MONEY * KRO)
````

````@example whats-the-dollar-of-mexico
F_UM = USTATES * MEXICO
F_SU = SWEDEN * USTATES
F_SM = SWEDEN * MEXICO

F_SU * F_UM ≈ F_SM
````

````@example whats-the-dollar-of-mexico
USTATES * DOL ≈ MEXICO * PES
````

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ PES
````

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ KRO
````

````@example whats-the-dollar-of-mexico
USTATES * DOL * MEXICO ≈ DOL
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

