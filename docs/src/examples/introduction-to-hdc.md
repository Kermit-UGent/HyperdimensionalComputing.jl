```@meta
EditURL = "introduction-to-hdc.jl"
```

# Introduction to Hyperdimensional Computing

Hyperdimensional Computing (HDC) is a brain-inspired computational paradigm that represents
and manipulates information using high-dimensional vectors called **hypervectors**. These
vectors typically have thousands of dimensions (often 1.000-10.000), making them
"hyperdimensional." The key insight is that high-dimensional spaces have unusual mathematical
properties that allow for robust, fault-tolerant computation based on a defined set of
operations that enable representing any object or structure as a hypervector.

Our running example is deliberately simple: we will "cook" a set of plates using hypervectors,
showcase how to compare them, and finally how to do some inference using algebraic operations
over our "food" hypervectors.

# Setting up our experiment

First of all, we need to define the nature of our hypervectors. For this, we pick with
*vector-symbolic architecture* (VSA) we will work in. This is, essentially, the flavour of
hypervectors we will work with. For tutorial sake, we will use `BinaryHV`, the *binary spatter
code* and arguably the most widely used VSA.

````@example introduction-to-hdc
using HyperdimensionalComputing

H = BinaryHV
````

Let's create a hypervector to see what one actually looks like:

````@example introduction-to-hdc
h = H()
````

It is just a long vector of *bits*. For `BinaryHV` each component is a `0` or a `1`:

````@example introduction-to-hdc
eltype(h)
````

...and there are a lot of them -- this is where the "hyper" comes from:

````@example introduction-to-hdc
length(h)
````

!!! info "On hypervector types"
    We aliased `H = BinaryHV`, but the package offers several flavours -- `BipolarHV`,
    `TernaryHV`, `RealHV`, `GradedHV`, `GradedBipolarHV`, `FHRR` -- all sharing the abstract
    type `AbstractHV`. Type `?AbstractHV` in the REPL to see them, or `?BinaryHV` for the one
    we use here. Because everything below is written in terms of `H`, you can rerun the whole
    tutorial in another VSA by changing that single alias. By default a hypervector has 10.000
    dimensions; pass `D` to change it, e.g. `H(; D = 8)`.

Now, let's go over each operation we need to "cook" with hypervectors:

## Mapping ($\varphi$): _every ingredient is a hypervector_

The first rule of HDC is that *everything is a hypervector*. The mapping $\varphi$ takes any
object -- a word, a number, or here an ingredient -- and assigns it a hypervector. The simplest
mapping just draws a fresh random hypervector for each item. Let's stock our pantry, giving every
ingredient its own random hypervector named (for fun) with its emoji:

````@example introduction-to-hdc
🥓 = H()  # bacon
🥩 = H()  # beef
🍞 = H()  # bread
🍔 = H()  # bun
🧀 = H()  # cheese
🍗 = H()  # chicken
🥬 = H()  # lettuce
🥚 = H()  # mayo
🧅 = H()  # onion
🌶️ = H()  # salsa
🍅 = H()  # tomato
🫓 = H()  # tortilla
🦃 = H(); # turkey
nothing #hide
````

!!! tip "Seeding hypervectors"
    Each ingredient above is an *independent random draw*, so the exact numbers throughout this
    tutorial will differ every time you run it. When you instead need **reproducible** vectors --
    or want the same object to always map to the same hypervector (e.g. so the token `"🥩"` maps
    to one fixed vector everywhere in a pipeline) -- you can *seed* a hypervector from any Julia
    object by passing it to the constructor: `H("🥩")`, `H(:beef)`, and `H(42)` each return a
    vector fully determined by their seed.

Because each ingredient is an independent random draw, *different* ingredients are essentially
unrelated ("quasi-orthogonal"). We can check this by comparing 🥩 against a few ingredients at
once. `similarity(🥩)` returns a *function* that measures similarity to 🥩, which we broadcast
over a list:

````@example introduction-to-hdc
similarity(🥩).([🥩, 🧀, 🧅])
````

`BinaryHV` uses the **Jaccard** similarity, which runs from `0` to `1`. A hypervector is
perfectly similar to itself (`1.0`), while two *unrelated* vectors share about a third of their
set bits and so sit near a **baseline of ≈ 0.33** -- that is the "chance" level. Read the rest
of the tutorial with that in mind: *similar* means "clearly above ~0.33", and *dissimilar* means
"right around ~0.33." This reliable separation between related and random vectors is the bedrock
the rest of HDC is built on.

# The kitchen operations

HDC has **three** primary operations. Each takes hypervectors and returns another hypervector of
the same size, so results can be fed back in indefinitely -- this composability is what lets us
build a whole plate out of a handful of ingredients.

| Operation           | Purpose                    | Math                | In `HyperdimensionalComputing.jl` |
|:--------------------|:---------------------------|:--------------------|:----------------------------------|
| Bundling ($\oplus$) | *mix* into something alike  | $[\,h_1 + h_2 + \dots\,]$ | `bundle`, or the `+` operator |
| Binding ($\otimes$) | *associate* into something new | $h_1 \otimes h_2$ (XOR for `BinaryHV`) | `bind` / `unbind`, or the `*` operator |
| Permutation ($\rho$)| *order* by shuffling         | $\rho(h)$           | `ρ` (a.k.a. `shift`) |

where $[\,\cdot\,]$ denotes a normalization step that keeps the result a valid hypervector.
Let's meet them one at a time.

## Bundling ($\oplus$): mixing

**Bundling** (a.k.a. superposition) combines hypervectors into a new one that is *similar to all
of its ingredients* -- think of tossing everything into one bowl. Let's mix a taco filling:

````@example introduction-to-hdc
filling = bundle([🥩, 🧅, 🧀])
````

You can also use the overloaded `+` operator:

````@example introduction-to-hdc
filling == 🥩 + 🧅 + 🧀
````

The mix "remembers" what went into it: it is clearly similar to each of its ingredients, but not
to something we never added (bread is a stranger to this bowl):

````@example introduction-to-hdc
similarity(filling).([🥩, 🧅, 🧀, 🍞])
````

## Binding ($\otimes$): associating

**Binding** combines hypervectors into a new one that is *dissimilar to its inputs*. It is the
tool for **associating** things -- for saying "this ingredient plays *that* role." Let's define
a `ROLE` hypervector and bind cheese to it:

````@example introduction-to-hdc
ROLE = H(:role)
topping = ROLE * 🧀
````

The resulting `topping` sits back at the ~0.33 baseline against both the role and the ingredient
-- binding *hides* its operands, so `topping` looks unrelated to either:

````@example introduction-to-hdc
similarity(topping).([🧀, ROLE])
````

Crucially, binding is *reversible*. For `BinaryHV` the bind is a bitwise **XOR**, which is its
own inverse, so binding again with the role recovers the ingredient *exactly* (similarity `1.0`).
This "unbinding" is what will later let us *query* a recipe:

````@example introduction-to-hdc
similarity(ROLE * topping, 🧀)
````

## Permutation ($\rho$): ordering

**Permutation** ($\rho$) takes a single hypervector and cyclically shifts it into a new one that
is dissimilar to the original. It is how HDC encodes *order* -- because in the kitchen, order
matters (sear *then* simmer is not the same as simmer *then* sear):

````@example introduction-to-hdc
similarity(🥩, ρ(🥩))
````

Applying it repeatedly keeps producing fresh, quasi-orthogonal vectors, giving each position its
own signature:

````@example introduction-to-hdc
similarity(🥩).([🥩, ρ(🥩, 1), ρ(🥩, 2), ρ(🥩, 3)])
````

We can use this to make *order matter*. Encode a two-step recipe by permuting the second step
once (position 0, then position 1), and compare it to the same steps performed in the opposite
order:

````@example introduction-to-hdc
sear = H(:sear)
simmer = H(:simmer)

similarity(sear + ρ(simmer), simmer + ρ(sear))
````

Same two actions, different order -- and the hypervectors come out unrelated.

# Cooking a plate: encoding recipes

We now combine the operations to "cook." There is no single right way to turn a list of
ingredients into a plate hypervector -- the choice of *encoder* determines what the resulting
vector remembers. The package ships several; here we compare three, from least to most
structured, using a taco's ingredients:

````@example introduction-to-hdc
ingredients = [🫓, 🥩, 🧅, 🌶️, 🧀]
````

**`multiset` -- an unordered bag** ($\oplus_i V_i$). It simply bundles the ingredients. It is
the simplest encoder, but it forgets *everything* except which ingredients are present: shuffle
them and you get the exact same vector.

````@example introduction-to-hdc
similarity(multiset(ingredients), multiset(reverse(ingredients)))
````

**`bundlesequence` -- an ordered stack** ($\oplus_i \rho^{\,i-1}(V_i)$). It permutes each
ingredient by its position before bundling, so *order is remembered*. Now reversing the stack
gives an unrelated vector -- useful for layered dishes or recipe steps, where sequence matters:

````@example introduction-to-hdc
similarity(bundlesequence(ingredients), bundlesequence(reverse(ingredients)))
````

**`hashtable` -- a keyed record** ($\oplus_i K_i \otimes V_i$). It binds each *value* to a *key*
and bundles the pairs. This is the most structured of the three: order is irrelevant, but each
ingredient is filed under the role it plays, so we can later *query it back*. Let's define our
roles:

````@example introduction-to-hdc
BASE = H(:base)      # the carb: tortilla, bun, bread...
PROTEIN = H(:protein)   # beef, chicken, turkey...
VEGGIE = H(:veggie)    # onion, lettuce...
SAUCE = H(:sauce)     # salsa, ketchup, mayo...
EXTRA = H(:extra)     # cheese, bacon...
roles = [BASE, PROTEIN, VEGGIE, SAUCE, EXTRA]
````

Our 🌮 **taco** -- a tortilla base, beef, onion, salsa, and a bit of cheese -- and our
🍔 **hamburger** -- a bun, beef, lettuce, tomato, and cheese:

````@example introduction-to-hdc
taco = hashtable(roles, [🫓, 🥩, 🧅, 🌶️, 🧀])
burger = hashtable(roles, [🍔, 🥩, 🥬, 🍅, 🧀])
````

Each plate is now a *single* hypervector encoding its whole (structured) recipe. The three
encoders trade off resolving power against simplicity: `multiset` answers only *"what is in
it?"*, `bundlesequence` also captures *"in what order?"*, and `hashtable` captures *"what plays
which role?"* -- the one we need to reason about recipes.

!!! tip "More encoders"
    `multiset`, `bundlesequence`, and `hashtable` are just three of the built-in encoders. The
    package also provides `multibind`, `bindsequence`, `ngrams`, `graph`, `crossproduct`, level
    encoders, and more -- see the [API reference](../api.md) for the full catalogue.

# Comparison: are two plates alike?

With every plate living in the same space, we can reason about them in two complementary ways:
by **measuring similarity** and by doing **algebra** on the hypervectors.

## Measuring similarity

Let's add a third plate, a 🥪 **club sandwich**. This one is interesting: its protein could be
chicken *or* turkey. We express that ambiguity directly by **superposing** (bundling) the two
poultry options into a single hypervector that is similar to both:

````@example introduction-to-hdc
poultry = 🍗 + 🦃
````

The sandwich is then bread, that poultry, lettuce, mayo, and bacon:

````@example introduction-to-hdc
sandwich = hashtable(roles, [🍞, poultry, 🥬, 🥚, 🥓])
````

How close are the taco and the hamburger? The shared beef and cheese make them noticeably
alike, while the sandwich shares only its lettuce with the burger and nothing with the taco:

````@example introduction-to-hdc
similarity(taco).([burger, sandwich])
````

We can look at all three plates at once with a similarity matrix (rows/columns are taco, burger,
sandwich):

````@example introduction-to-hdc
plates = [taco, burger, sandwich]
similarity(plates)
````

The pattern matches culinary intuition: **taco and burger are the most alike** (shared beef +
cheese), **burger and sandwich are mildly alike** (shared lettuce), and **taco and sandwich are
strangers**. Similar recipes give similar vectors.

## Algebra: querying and mapping between plates

Because binding is reversible, a plate is not a black box -- it is a little database we can
query. Unbinding a plate with a *role* recovers the ingredient that filled it. We compare the
result against the pantry with `nearest_neighbor`:

````@example introduction-to-hdc
pantry = [🫓, 🍔, 🍞, 🥩, 🍗, 🦃, 🧅, 🥬, 🌶️, 🍅, 🥚, 🧀, 🥓]
names = [
    "tortilla", "bun", "bread", "beef", "chicken", "turkey", "onion",
    "lettuce", "salsa", "tomato", "mayo", "cheese", "bacon",
]

nearest_neighbor(taco * PROTEIN, pantry)
````

The result is a `(similarity, index, hypervector)` tuple pointing at the winning ingredient --
here, the taco's protein is beef. Remember the sandwich's *ambiguous* protein? Querying it
recovers **both** poultry options and rejects beef, exactly as the superposition intended:

````@example introduction-to-hdc
similarity(sandwich * PROTEIN).([🍗, 🦃, 🥩])
````

Sweeping every role reconstructs the full menu straight from the plate hypervectors alone:

````@example introduction-to-hdc
recover(plate, role) = names[nearest_neighbor(plate * role, pantry)[2]]
[recover(plate, role) for plate in plates, role in roles]
````

Each row is a plate, each column a role. Unbinding also works the other way around: give a plate
an *ingredient* and it tells you the *role* that ingredient plays.

````@example introduction-to-hdc
rolenames = ["BASE", "PROTEIN", "VEGGIE", "SAUCE", "EXTRA"]
rolenames[argmax(similarity(taco * 🧅).(roles))]
````

This two-way lookup lets us **map concepts from one dish to another**. Suppose we like the onion
in our taco and ask: *"what plays the same part in the burger?"* We do it in two clean steps --
first find onion's role in the taco, then read that role out of the burger:

````@example introduction-to-hdc
onion_role = roles[argmax(similarity(taco * 🧅).(roles))]   # 🧅 is the taco's VEGGIE...
recover(burger, onion_role)                                    # ...and the burger's VEGGIE is?
````

The system answers `lettuce`: *onion is to the taco what lettuce is to the burger*. We have
inferred an analogy the recipes never stated explicitly -- the kind of associative reasoning
that makes hyperdimensional representations so powerful.

# Wrap-up

In one sitting we cooked three plates and met the whole HDC toolkit:

- **Mapping** turned emojis into hypervectors.
- **Bundling** mixed ingredients into a filling similar to its parts (and let a protein be
  "chicken *or* turkey").
- **Binding** associated ingredients with roles -- and let us un-associate them again.
- **Permutation** let order matter.
- Different **encoders** (`multiset`, `bundlesequence`, `hashtable`) remember different things.
- **Similarity** told us which plates are alike, and **algebra** let us query recipes and map
  concepts from one dish to another.

The takeaways generalize far beyond the kitchen: all data lives in the *same* high-dimensional
space, the representation is robust to noise thanks to the blessing of dimensionality, and
hypervectors plus a handful of encoders can represent richly structured data. From here, take a
look at the *"What's the Dollar of Mexico?"* example for more analogical reasoning, or the *Iris
dataset* example for a full classification workflow.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

