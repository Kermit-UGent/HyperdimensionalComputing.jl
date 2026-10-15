"""
    multiset(vs::AbstractVector{<:T})::T where {T <: AbstractHV}

Multiset of input hypervectors, bundles all the input hypervectors together.

# Arguments
- `vs::AbstractVector{<:AbstractHV}`: Hypervectors

# Example
```julia-repl
julia> vs = BinaryHV.('a':'j'; D = 10)  # a hypervector for each character
10-element Vector{BinaryHV}:
 10-element BinaryHV with 5 true and 5 false
 10-element BinaryHV with 6 true and 4 false
 10-element BinaryHV with 3 true and 7 false
 10-element BinaryHV with 6 true and 4 false
 10-element BinaryHV with 3 true and 7 false
 10-element BinaryHV with 6 true and 4 false
 10-element BinaryHV with 4 true and 6 false
 10-element BinaryHV with 3 true and 7 false
 10-element BinaryHV with 5 true and 5 false
 10-element BinaryHV with 3 true and 7 false

julia> multiset(vs)
10-element BinaryHV with 3 true and 7 false:
 1
 0
 0
 0
 0
 0
 1
 1
 0
 0
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\bigoplus_{i=1}^{m} V_i
```

where \$V\$ is the hypervector collection, \$m\$ is the size of the hypervector collection,
\$i\$ is the position of the entry in the collection, and \$\\oplus\$ is the bundling operation.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.multiset.html)

# See also

- [`multibind`](@ref): Multibind encoding, binding-variant of this encoder
"""
function multiset(vs::AbstractVector{<:T})::T where {T <: AbstractHV}
    return bundle(vs)
end

"""
    multibind(vs::AbstractVector{<:AbstractHV})

Binding of multiple hypervectors, binds all the input hypervectors together.

# Arguments
- `vs::AbstractVector{<:AbstractHV}`: Hypervectors

# Examples
```julia-repl
julia> vs = BinaryHV.('a':'j'; D = 10);  # a hypervector for each character

julia> multibind(vs)
10-element BinaryHV with 4 true and 6 false:
 1
 0
 0
 0
 0
 0
 1
 0
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\bigotimes_{i=1}^{m} V_i
```

where \$V\$ is the hypervector collection, \$m\$ is the size of the hypervector collection,
\$i\$ is the position of the entry in the collection, and \$\\otimes\$ is the binding operation.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.multibind.html)

# See also

- [`multiset`](@ref): Multiset encoding, bundling-variant of this encoder
"""
function multibind(vs::AbstractVector{<:AbstractHV})
    return bind(vs)
end


"""
    bundlesequence(vs::AbstractVector{<:AbstractHV})

Bundling-based sequence. The first value is not permuted, the last value is permuted n-1 times.

# Arguments
- `vs::AbstractVector{<:AbstractHV}`: Hypervector sequence

# Examples
```julia-repl
julia> vs = BinaryHV.('a':'j'; D = 10);  # a hypervector for each character

julia> bundlesequence(vs)
10-element BinaryHV with 4 true and 6 false:
 0
 0
 0
 1
 1
 0
 1
 1
 0
 0
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\bigoplus_{i=1}^{m} \\rho(V_i, i-1)
```

where \$V\$ is the hypervector collection, \$m\$ is the size of the hypervector collection,
\$i\$ is the position of the entry in the collection, and \$\\oplus\$ and \$\\rho\$ are the
bundling and shift operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.bundle_sequence.html)

# See also

- [`bindsequence`](@ref): Binding-sequence encoding, binding-variant of this encoder
"""
# Fused: folds the bundling rule over the shifted inputs block by block, without
# materializing the shifts. Collections the fused kernel cannot take (mixed types,
# element types whose arithmetic can throw) shift and bundle explicitly.
function bundlesequence(vs::AbstractVector{<:AbstractHV})
    @assert length(vs) > 1 "Can't bundle sequence of a single hypervector"
    hv1 = first(vs)
    r = empty_vector(hv1)
    f = aggfun(hv1)
    if !(uniform_collection(vs) && kernel_safe(f, eltype(r), eltype(hv1.v)))
        return bundle([shift(hv, i - 1) for (i, hv) in enumerate(vs)])
    end
    check_samelength(r, vs)
    n = length(r)
    foreach_block!(r, length(vs) * n) do rng
        for (i, hv) in enumerate(vs)
            foreach_shifted(rng, i - 1, n) do dst, src
                rv = view(r, dst)
                rv .= f.(rv, view(hv.v, src))
            end
        end
    end
    return bundleresult(hv1, r, length(vs))
end

"""
    bindsequence(vs::AbstractVector{<:AbstractHV})

Binding-based sequence. The first value is not permuted, the last value is permuted n-1 times.

# Arguments
- `vs::AbstractVector{<:AbstractHV}`: Hypervector sequence

# Examples
```julia-repl
julia> vs = BinaryHV.('a':'j'; D = 10);  # a hypervector for each character

julia> bindsequence(vs)
10-element BinaryHV with 6 true and 4 false:
 1
 0
 0
 1
 1
 0
 0
 1
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\bigotimes_{i=1}^{m} \\rho(V_i, i-1)
```

where \$V\$ is the hypervector collection, \$m\$ is the size of the hypervector collection,
\$i\$ is the position of the entry in the collection, and \$\\otimes\$ and \$\\rho\$ are the
binding and shift operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.bind_sequence.html)

# See also

- [`bundlesequence`](@ref): Bundle-sequence encoding, bundling-variant of this encoder
"""
# Fused like `bundlesequence`.
function bindsequence(vs::AbstractVector{<:AbstractHV})
    @assert length(vs) > 1 "Can't bind sequence of a single hypervector"
    hv1 = first(vs)
    f = bindfun(hv1)
    T = eltype(hv1.v)
    # Packed bits bind faster word-wise, on materialized shifts, than unpacked.
    if hv1.v isa BitVector || !(uniform_collection(vs) && kernel_safe(f, T, T))
        return bind([shift(hv, i - 1) for (i, hv) in enumerate(vs)])
    end
    check_samelength(hv1, vs)
    r = Vector{T}(hv1.v)
    n = length(r)
    foreach_block!(r, length(vs) * n) do rng
        for (i, hv) in enumerate(vs)
            i == 1 && continue
            foreach_shifted(rng, i - 1, n) do dst, src
                rv = view(r, dst)
                rv .= f.(rv, view(hv.v, src))
            end
        end
    end
    return rewrap(hv1, r)
end

"""
    hashtable(keys::T, values::T) where {T <: AbstractVector{<:AbstractHV}}

Hash table from keys-values hypervector pairs. Keys and values must be the same length in order
to encode as hypervector.

# Arguments
- `keys::AbstractVector{<:AbstractHV}`: Keys hypervectors
- `values::AbstractVector{<:AbstractHV}`: Values hypervectors

# Example
```julia-repl
julia> ks = BinaryHV.([:name, :age, :city]; D = 10);  # key hypervectors

julia> vs = BinaryHV.(["Alice", "42", "Ghent"]; D = 10);  # value hypervectors

julia> hashtable(ks, vs)
10-element BinaryHV with 3 true and 7 false:
 0
 0
 0
 1
 1
 0
 0
 0
 1
 0
```

# Extended help

This encoding is based on the following mathematical notation:

```math
\\bigoplus_{i=1}^{m} K_i \\otimes V_i
```

where \$K\$ and \$V\$ are the key and value hypervector collections, \$m\$ is the size of the
hypervector collection, \$i\$ is the position of the entry in the collection, and \$\\otimes\$
and \$\\oplus\$ are the binding and bundling operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.hash_table.html)
"""
function hashtable(keys::T, values::T) where {T <: AbstractVector{<:AbstractHV}}
    @assert length(keys) == length(values) "Number of keys and values aren't equal"
    isempty(keys) && return bundle(map(prod, zip(keys, values)))
    hv1 = first(keys)
    r = empty_vector(hv1)
    af, bf = aggfun(hv1), bindfun(hv1)
    S = eltype(hv1.v)
    fusable = !(hv1.v isa BitVector) && uniform_collection(keys) && uniform_collection(values) &&
        typeof(first(values)) === typeof(hv1) && kernel_safe(bf, S, S) && kernel_safe(af, eltype(r), S)
    fusable || return bundle(map(prod, zip(keys, values)))
    # Fused: bundles each key-value product block by block, without materializing
    # the bound pairs.
    check_samelength(r, keys)
    check_samelength(r, values)
    foreach_block!(r, 2 * length(keys) * length(r)) do rng
        rv = view(r, rng)
        for (k, v) in zip(keys, values)
            rv .= af.(rv, bf.(view(k.v, rng), view(v.v, rng)))
        end
    end
    return bundleresult(hv1, r, length(keys))
end

"""
    crossproduct(U::T, V::T) where {T <: AbstractVector{<:AbstractHV}}

Cross product between two sets of hypervectors.


# Arguments
- `U::AbstractVector{<:AbstractHV}`: Hypervectors
- `V::AbstractVector{<:AbstractHV}`: Hypervectors

# Examples
```julia-repl
julia> us = BinaryHV.('a':'e'; D = 10);

julia> vs = BinaryHV.('v':'z'; D = 10);

julia> crossproduct(us, vs)
10-element BinaryHV with 5 true and 5 false:
 0
 1
 0
 1
 0
 0
 1
 1
 1
 0
```

# Extended help

This encoding strategy first creates a multiset from both input hypervector sets,
which are then bound together to generate all cross products, i.e.

```math
U_1 \\times V_1 + U_1 \\times V_2 + ... + U_1 \\times V_m + ... + U_n \\times V_m
```

This encoding is based on the following formula:

```math
\\bigoplus_{i=1}^{m} U_i \\ \\otimes \\ \\bigoplus_{i=1}^{n} V_i
```

where \$U\$ and \$V\$ are collections of hypervectors, \$m\$ and \$n\$ are the sizes of the U and V collections,
\$i\$ is the position in the hypervector collection, and \$\\oplus\$ and \$\\otimes\$ are the bundling
and binding operations.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.cross_product.html)
"""
function crossproduct(U::T, V::T) where {T <: AbstractVector{<:AbstractHV}}
    # TODO: This should be bundled without normalizing
    return bind(multiset(U), multiset(V))
end

"""
    ngrams(vs::AbstractVector{<:AbstractHV}, n::Int = 3)

Creates a hypervector with the _n_-gram statistics of the input.

# Arguments
- `vs::AbstractVector{<:AbstractHV}`: Hypervector collection
- `n::Int = 3`: _n_-gram size

# Examples
```julia-repl
julia> vs = BinaryHV.('a':'j'; D = 10);  # a hypervector for each character

julia> ngrams(vs)
10-element BinaryHV with 5 true and 5 false:
 1
 0
 0
 1
 0
 1
 0
 0
 1
 1
```

# Extended help

This encoding is defined by the following mathematical notation:

```math
\\bigoplus_{i=1}^{m-n}\\bigotimes_{j=1}^{n-1}\\rho^{n-j-1}(V_{i+j})
```

where \$V\$ is the collection of hypervectors, \$m\$ is the number of hypervectors in the
collection \$V\$, \$n\$ is the window size, \$i\$ is the position in the sequence, \$j\$ is the
position in the _n_-gram, and \$\\oplus\$, \$\\otimes\$ and \$\\rho\$ are the bundling, binding
and shift operations.

!!! note
    - For `n = 1` use `multiset()` instead
    - For `n = m` use `bindsequence()` instead

# See also

- [`multiset`](@ref): Multiset encoding, equivalent to `ngram(vs, 1)`
- [`bindsequence`](@ref): Bind-sequence encoding, equivalent to `ngram(vs, length(vs))`

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.ngrams.html)
"""
function ngrams(vs::AbstractVector{<:AbstractHV}, n::Int = 3)
    l = length(vs)
    p = l - n + 1
    @assert 1 <= n <= length(vs) "`n` must be 1 ≤ n ≤ $l"
    hv1 = first(vs)
    r = empty_vector(hv1)
    af, bf = aggfun(hv1), bindfun(hv1)
    T = eltype(hv1.v)
    if n == 1 || hv1.v isa BitVector || !(uniform_collection(vs) && kernel_safe(bf, T, T) && kernel_safe(af, eltype(r), T))
        return map(
            s -> bindsequence(s),
            (vs[f:(f + (n - 1))] for f in 1:p)
        ) |> multiset
    end
    # Fused: each block binds every window's shifted members into a block-sized
    # buffer and folds it into the bundle, so no per-window hypervectors are allocated.
    check_samelength(r, vs)
    D = length(r)
    i0 = firstindex(vs)
    foreach_block!(r, p * n * D) do rng
        buf = Vector{T}(undef, length(rng))
        for f in 0:(p - 1)
            buf .= view(vs[i0 + f].v, rng)
            for j in 1:(n - 1)
                hv = vs[i0 + f + j]
                foreach_shifted(rng, j, D) do dst, src
                    bv = view(buf, dst .- (first(rng) - 1))
                    bv .= bf.(bv, view(hv.v, src))
                end
            end
            rv = view(r, rng)
            rv .= af.(rv, buf)
        end
    end
    return bundleresult(hv1, r, p)
end

"""
    graph(source::T, target::T, directed::Bool = false)

Graph for `source`-`target` pairs. Can be directed or undirected.

# Arguments
- `source::AbstractVector{<:AbstractHV}`: Set of source node hypervectors
- `target::AbstractVector{<:AbstractHV}`: Set of target node hypervectors
- `directed::Bool = false`: Whether the graph is directed or not

# Example

```julia-repl
julia> V = [BinaryHV(i; D = 10) for i in 1:7];

julia> E = [1 2; 1 3; 1 4; 2 3; 2 4; 3 4; 4 5; 5 6; 6 7]; # Lollipop graph

julia> graph(V[E[:, 1]], V[E[:, 2]])
10-element BinaryHV with 7 true and 3 false:
 1
 1
 0
 1
 1
 1
 0
 0
 1
 1
```

# Extended help

This encoding is based on the following mathematical notation:

*Undirected graphs*
```math
\\bigoplus_{i=1}^{E} V_i \\otimes V_j
```

*Directed graphs*
```math
\\bigoplus_{i=1}^{E} V_i \\otimes \\rho(V_j)
```

where \$V\$ is the node hypervector, \$i\$ and \$j\$ refer to the source and target nodes,
\$E\$ is the set of edges between nodes in the graph, and \$\\otimes\$,
\$\\oplus\$ and \$\\rho\$ are the binding, bundling and shift operations.

# See also

- [`hashtable`](@ref): Hash table encoding, underlying encoding strategy of this encoder.

# References

- [Torchhd documentation](https://torchhd.readthedocs.io/en/stable/generated/torchhd.graph.html)

"""
function graph(source::T, target::T; directed::Bool = false) where {T <: AbstractVector{<:AbstractHV}}
    @assert length(source) == length(target) "`source` and `target` must be the same length"
    return hashtable(source, shift.(target, convert(Int, directed)))
end
