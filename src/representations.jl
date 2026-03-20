#=
Representing the hypervectors using pretty printing
and a custom plotting recipe

See ext/UnicodePlotting.jl for extensions based on UnicodePlotting
=#

function Base.show(io::IO, ::MIME"text/plain", hvs::AbstractVector{<:AbstractHV})
    println(io, "$(length(hvs))-element $(typeof(hvs)):")
    r = map(hvs) do hv
        if typeof(hv) == BinaryHV
            counts = Dict(e => count(==(e), hv) for e in unique(hv))
            " $(length(hv))-element $(typeof(hv)) with $(counts[true]) true and $(counts[false]) false"
        elseif typeof(hv) == BipolarHV
            counts = Dict(e => count(==(e), hv) for e in unique(hv))
            " $(length(hv))-element $(typeof(hv)) with $(counts[1]) positives and $(counts[-1]) negatives"
        elseif typeof(hv) == TernaryHV
            counts = Dict(1 => count(>=(1), hv), -1 => count(<=(-1), hv), 0 => count(==(0), hv))
            " $(length(hv))-element $(typeof(hv)) with $(counts[1]) positives, $(counts[0]) zeros, and $(counts[-1]) negatives"
        else
            " $(length(hv))-element $(typeof(hv)) with μ ± σ = $(round(mean(hv), digits = 3)) ± $(round(std(hv), digits = 3))"
        end
    end

    if length(r) <= displaysize(io)[1] - 4
        return print(io, join(r, '\n'))
    end

    chunksize = displaysize(io)[1] ÷ 2 - 3
    return print(io, join([first(r, chunksize); " ⋮"; last(r, chunksize)], '\n'))
end

function Base.show(io::IO, ::MIME"text/plain", hv::AbstractHV)
    # NOTE: Based off https://github.com/JuliaLang/julia/blob/cf40898d56a5b32c6a2e97f61355440df36a7357/base/arrayshow.jl#L363
    # Fast return for empty hypervectors
    if isempty(hv) && (get(io, :compact, false)::Bool || hv isa AbstractHV)
        return println(io, typeof(hv))
    end

    # 1) show summary before setting :compact
    if typeof(hv) == BinaryHV
        counts = Dict(e => count(==(e), hv) for e in unique(hv))
        print(io, "$(length(hv))-element $(typeof(hv)) with $(counts[true]) true and $(counts[false]) false")
    elseif typeof(hv) == BipolarHV
        counts = Dict(e => count(==(e), hv) for e in unique(hv))
        print(io, "$(length(hv))-element $(typeof(hv)) with $(counts[1]) positives and $(counts[-1]) negatives")
    elseif typeof(hv) == TernaryHV
        counts = Dict(1 => count(>=(1), hv), -1 => count(<=(-1), hv), 0 => count(==(0), hv))
        print(io, "$(length(hv))-element $(typeof(hv)) with $(counts[1]) positives, $(counts[0]) zeros, and $(counts[-1]) negatives")
    else
        print(io, "$(length(hv))-element $(typeof(hv)) with μ ± σ = $(round(mean(hv), digits = 3)) ± $(round(std(hv), digits = 3))")
    end

    print(io, ":")

    # 2) compute new IOContext
    if !haskey(io, :compact) && length(axes(hv, 2)) > 1
        io = IOContext(io, :compact => true)
    end
    if get(io, :limit, false)::Bool && eltype(hv) === Method
        io = IOContext(io, :limit => false)
    end

    if get(io, :limit, false)::Bool && displaysize(io)[1] - 4 <= 0
        print(io, " …")
    else
        println(io)
    end

    # 3) update typeinfo
    io = IOContext(io, :typeinfo => eltype(hv))

    # 4) show actual content
    recur_io = IOContext(io, :SHOWN_SET => hv)
    return Base.print_array(recur_io, hv)
end
