# The hypervector operations use fused and (with more than one Julia thread)
# Polyester-parallel kernels. These tests compare them against references built from
# the serial primitives: passing a generator to `bundle` takes its serial path, and
# folding the pairwise `bind` materializes every intermediate. Kernels that apply
# the same elementwise rules in the same order must agree exactly. `runtests.jl`
# also runs this file in separate processes with one and with four threads, so both
# code paths are exercised whatever thread count the main test process has.

using HyperdimensionalComputing
using Random
using Test

const H = HyperdimensionalComputing

serial_bundle(hvs; kwargs...) = bundle((hv for hv in hvs); kwargs...)
ref_bindseq(hs) = foldl(bind, [shift(h, i - 1) for (i, h) in enumerate(hs)])
ref_bundleseq(hs) = serial_bundle(shift(h, i - 1) for (i, h) in enumerate(hs))
ref_ngrams(hs, n) = serial_bundle(ref_bindseq(hs[f:(f + n - 1)]) for f in 1:(length(hs) - n + 1))
ref_hashtable(ks, vs) = serial_bundle(bind(k, v) for (k, v) in zip(ks, vs))

function ref_similarity(u, v)
    x, y = collect(u), collect(v)
    if similaritymetric(u) === :jaccard
        x, y = Float64.(x), Float64.(y)
        return sum(x .* y) / sum(x .+ y .- x .* y)
    elseif u isa FHRR
        return real(sum(conj.(x) .* y)) / length(x)
    else
        x, y = Float64.(x), Float64.(y)
        return sum(x .* y) / sqrt(sum(abs2, x) * sum(abs2, y))
    end
end

const HV_TYPES = (BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR)

@testset "kernels (threads = $(Threads.nthreads()))" begin
    @testset "$HV" for HV in HV_TYPES
        rng = Xoshiro(3)
        # sizes on both sides of the block size and the parallel-work threshold,
        # including a dimension that is not a multiple of the block size, and both
        # even and odd counts (majority bundling breaks ties for even counts)
        sets = (
            [HV(; D = 10_000, rng) for _ in 1:40],
            [HV(; D = 10_007, rng) for _ in 1:9],
            [HV(; D = 7, rng) for _ in 1:6],
            [HV(; D = 10_000, rng) for _ in 1:2],
        )
        for hs in sets
            @test bundle(hs) == serial_bundle(hs)
            @test bundle(Tuple(hs)) == serial_bundle(hs)
            @test bind(hs) == foldl(bind, hs)
            @test bind(Tuple(hs)) == foldl(bind, hs)
            @test bind(hs[1], hs[2]).v == H.bindfun(hs[1]).(hs[1].v, hs[2].v)
            @test bundlesequence(hs) == ref_bundleseq(hs)
            @test bindsequence(hs) == ref_bindseq(hs)
            for n in 2:min(4, length(hs))
                @test ngrams(hs, n) == ref_ngrams(hs, n)
            end
            @test hashtable(hs, reverse(hs)) == ref_hashtable(hs, reverse(hs))
            @test graph(hs, reverse(hs); directed = true) == ref_hashtable(hs, shift.(reverse(hs), 1))
            @test shift(hs[1], 3).v == circshift(hs[1].v, 3)
            @test shift(hs[1], -2).v == circshift(hs[1].v, -2)

            @test similarity(hs[1], hs[2]) ≈ ref_similarity(hs[1], hs[2])
            S = similarity(hs)
            @test S == [similarity(a, b) for a in hs, b in hs]
            @test S == S'

            u = bundle(hs[1:2])
            @test nearest_neighbor(u, hs) === maximum((similarity(u, x), i, x) for (i, x) in enumerate(hs))
            @test nearest_neighbor(u, hs, 2) == partialsort!([(similarity(u, x), i) for (i, x) in enumerate(hs)], 1:2, rev = true)
            @test nearest_neighbor(u, hs; method = :hamming)[1] == maximum(similarity(u, h; method = :hamming) for h in hs)
        end

        long, short = sets[1][1], sets[3][1]
        @test_throws DimensionMismatch bundle([long, short])
        @test_throws DimensionMismatch bind([long, short])
        @test_throws DimensionMismatch bind(long, short)
        @test_throws DimensionMismatch bundlesequence([long, short])
        @test_throws DimensionMismatch ngrams([long, short, long], 2)
        @test_throws DimensionMismatch hashtable([long, long], [long, short])
        @test_throws DimensionMismatch similarity(long, short)
        @test_throws DimensionMismatch similarity([long, short])
        @test_throws AssertionError ngrams(sets[1], 1)
    end

    @testset "bundle keyword arguments" begin
        bs = [BinaryHV(; D = 10_000, seed = i) for i in 1:10]
        @test bundle(bs; rng = Xoshiro(1)) == serial_bundle(bs; rng = Xoshiro(1))
        ts = [TernaryHV(; D = 10_000, seed = i) for i in 1:10]
        @test bundle(ts; normalize = true).v == clamp.(serial_bundle(ts).v, -1, 1)
    end

    @testset "TernaryHV element types" begin
        small = [TernaryHV{Int8}(; D = 7, seed = i) for i in 1:6]
        @test eltype(bundle(small).v) == Int8
        @test eltype(ngrams(small, 3).v) == Int8
        @test eltype(shift(small[1], 1).v) == Int8
        big, other = bundle([TernaryHV(; D = 10_000, seed = i) for i in 1:40]), TernaryHV(; D = 10_000, seed = 99)
        @test similarity(big, other) ≈ ref_similarity(big, other)
        @test isnan(similarity(TernaryHV(zeros(Int, 5)), TernaryHV([1, 0, -1, 1, 0])))
    end

    @testset "validating constructors" begin
        @test_throws "unit modulus" FHRR(fill(2.0 + 0im, 10_000))
        @test_throws "[0, 1]" GradedHV([fill(0.5, 9_999); 1.5])
        @test_throws "[-1, 1]" GradedBipolarHV([fill(0.5, 9_999); -1.5])
    end

    # Inputs whose element operations could throw take the serial path, which reports
    # the error instead of failing inside a parallel task.
    @testset "serial fallback" begin
        mixed = [TernaryHV{Int8}(; D = 10_000, seed = 1), TernaryHV{Int}(fill(200, 10_000))]  # 200 overflows Int8
        @test_throws InexactError bundle(mixed)
        reals = [RealHV(Rational{Int}[1 // typemax(Int), 1 // 2]) for _ in 1:2]
        @test_throws OverflowError bind(reals[1], reals[2])
        # the parallel kernels still run after the errors
        hs = [RealHV(; D = 10_000, seed = i) for i in 1:10]
        @test bundle(hs) == serial_bundle(hs)
    end
end
