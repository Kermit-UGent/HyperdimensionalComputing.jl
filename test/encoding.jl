# Extension-point demo for the encode-strategy tests: one struct, one method.
struct ReversedSeq <: AbstractEncoding end
function HyperdimensionalComputing.encode(HV::Type{<:AbstractHV}, x, ::ReversedSeq; kwargs...)
    return encode(HV, reverse(collect(x)), Sequence(); kwargs...)
end

@testset "encoding" begin
    @testset "encode strategies" begin
        seq = "ACGTAC"
        D = 64

        # KMer: each window substring is one atomic token, bundled
        km = encode(BinaryHV, seq, KMer(3); D = D)
        km_ref = multiset([encode(BinaryHV, seq[i:(i + 2)]; D = D) for i in 1:4])
        @test km == km_ref

        # NGram: symbols encoded, windows composed by shift-binding (= ngrams)
        ng = encode(BinaryHV, seq, NGram(3); D = D)
        ng_ref = ngrams([encode(BinaryHV, c; D = D) for c in seq], 3)
        @test ng == ng_ref

        # KMer and NGram are genuinely different operations
        @test km != ng

        # the token path hashes the whole string and differs from any strategy
        @test encode(BinaryHV, seq) == encode(BinaryHV, seq)
        @test encode(BinaryHV, seq; D = D) != km

        # Sequence and BagOfSymbols match their combinator references
        vs = [encode(BipolarHV, c; D = D) for c in seq]
        @test encode(BipolarHV, seq, Sequence(); D = D) == bundlesequence(vs)
        @test encode(BipolarHV, seq, BagOfSymbols(); D = D) == multiset(vs)

        # non-string sequences work too (windows are tuples of symbols)
        v = [:a, :b, :a, :c]
        @test encode(BinaryHV, v, KMer(2); D = D) ==
            multiset([encode(BinaryHV, (v[i], v[i + 1]); D = D) for i in 1:3])

        # argument validation
        @test_throws ArgumentError KMer(0)
        @test_throws ArgumentError NGram(0)
        @test_throws ArgumentError encode(BinaryHV, "AC", KMer(3); D = D)

        # extension point: a struct plus one encode method is all it takes
        @test encode(BinaryHV, "AB", ReversedSeq(); D = D) ==
            encode(BinaryHV, "BA", Sequence(); D = D)
    end

    hvs = BinaryHV.(
        [
            Bool.([1, 0, 0, 0, 0]),
            Bool.([1, 1, 0, 0, 0]),
            Bool.([1, 1, 1, 0, 0]),
            Bool.([1, 1, 1, 1, 0]),
            Bool.([1, 1, 1, 1, 1]),
        ]
    )

    @testset "multiset" begin
        @test multiset(hvs).v == Bool.([1, 1, 1, 0, 0])
    end

    @testset "multibind" begin
        @test multibind(hvs).v == Bool.([1, 0, 1, 0, 1])
    end

    @testset "bundlesequence" begin
        @test bundlesequence(hvs).v == ones(Bool, 5)
        @test_throws AssertionError bundlesequence([first(hvs)])
    end

    @testset "bindsequence" begin
        @test bindsequence(hvs).v == ones(Bool, 5)
        @test_throws AssertionError bindsequence([first(hvs)])
    end

    @testset "hashtable" begin
        @test hashtable(hvs, hvs) == zeros(Bool, 5)
        @test_throws AssertionError hashtable(hvs, hvs[1:2])
    end

    @testset "crossproduct" begin
        @test crossproduct(hvs, hvs) == zeros(Bool, 5)
    end

    @testset "ngrams" begin
        @test ngrams(hvs).v == Bool.([0, 1, 0, 0, 1])
        @test ngrams(hvs) == bundle([hvs[1] * ρ(hvs[2]) * ρ(hvs[3], 2), hvs[2] * ρ(hvs[3]) * ρ(hvs[4], 2), hvs[3] * ρ(hvs[4]) * ρ(hvs[5], 2)])
        @test_throws AssertionError ngrams(hvs, 0)
        @test_throws AssertionError ngrams(hvs, length(hvs) + 1)
    end

    @testset "graph" begin
        s = [1, 3, 4, 2, 5]
        t = [3, 4, 2, 1, 4]
        @test graph(hvs[s], hvs[t]) == Bool.([0, 0, 0, 0, 0])
        @test graph(hvs[s], hvs[t]; directed = true) == Bool.([1, 0, 0, 1, 0])
        @test_throws AssertionError graph(hvs[s], hvs[[1, 2, 3]])
    end

    @testset "LevelEncoder" begin
        @testset "ladder for all types: $HV" for HV in
            (BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR)
            lvl = LevelEncoder(HV, (0, 1), 20; D = 1_000, seed = 17)
            @test lvl isa LevelEncoder{<:HV}
            @test lvl.base === nothing   # the ladder mechanism, even for FHRR
            @test length(lvl.levels) == length(lvl.values) == 20
            @test encode(lvl, 0.5) isa HV
            # round-trips within one quantization step
            step = 1 / 19
            for x in (0.0, 0.31, 0.77, 1.0)
                @test abs(decode(lvl, encode(lvl, x)) - x) ≤ step
            end
            # adjacent levels are more similar than distant ones
            @test similarity(lvl.levels[1], lvl.levels[2]) >
                similarity(lvl.levels[1], lvl.levels[end])
        end

        @testset "one shared level set (regression TODO §1.4/§1.4b)" begin
            # every encode/decode call draws from the level set built at
            # construction, so separate calls are comparable by construction —
            # the incomparability bug of the old encodelevel/decodelevel
            # function family cannot occur
            lvl = LevelEncoder(BipolarHV, (0, 2π), 50; D = 2_000, seed = 3)
            @test encode(lvl, 1.0) == encode(lvl, 1.0)
            a, b, c = encode(lvl, 1.0), encode(lvl, 1.1), encode(lvl, 5.0)
            @test similarity(a, b) > similarity(a, c)
            # encode and decode share ONE ladder: exact round-trip on the grid
            # (the old instance path built two ladders; error was up to 1.0)
            for v in lvl.values[[1, 10, 25, 50]]
                @test decode(lvl, encode(lvl, v)) == v
            end
            # seeded construction is fully deterministic
            @test LevelEncoder(BipolarHV, (0, 1), 5; D = 100, seed = 9).levels ==
                LevelEncoder(BipolarHV, (0, 1), 5; D = 100, seed = 9).levels
            # the old function family is gone, not just patched
            for f in (:level, :encodelevel, :decodelevel, :convertlevel)
                @test !isdefined(HyperdimensionalComputing, f)
            end
        end

        @testset "bandwidth controls similarity decay" begin
            ends(lvl) = similarity(lvl.levels[1], lvl.levels[end])
            narrow = LevelEncoder(BipolarHV, (0, 1), 20; D = 2_000, bandwidth = 0.02, seed = 7)
            wide = LevelEncoder(BipolarHV, (0, 1), 20; D = 2_000, bandwidth = 0.3, seed = 7)
            @test ends(narrow) > ends(wide)
            # FPE: β is the analogous knob
            slow = LevelEncoder(FHRR, 0:0.1:1; β = 0.05, D = 2_000, seed = 7)
            fast = LevelEncoder(FHRR, 0:0.1:1; β = 1.0, D = 2_000, seed = 7)
            @test similarity(encode(slow, 0.0), encode(slow, 1.0)) >
                similarity(encode(fast, 0.0), encode(fast, 1.0))
        end

        @testset "fractional power encoding (FHRR)" begin
            fpe = LevelEncoder(FHRR, 0:0.1:10; D = 1_000, seed = 11)
            @test fpe.base isa FHRR
            hv = encode(fpe, 3.14)   # continuous: no quantization
            @test hv isa FHRR
            @test all(abs.(hv.v) .≈ 1)
            # nearest-neighbour decode lands on the grid, within one step
            @test abs(decode(fpe, hv) - 3.14) ≤ 0.1
            # analytic decode is continuous and exact on a clean vector
            @test decode(fpe, hv; method = :analytic) ≈ 3.14
            # similarity decays with distance
            @test similarity(encode(fpe, 2.0), encode(fpe, 2.2)) >
                similarity(encode(fpe, 2.0), encode(fpe, 8.0))
        end

        @testset "precomputed levels constructor" begin
            src = LevelEncoder(BinaryHV, (0, 1), 10; D = 500, seed = 5)
            lvl = LevelEncoder(src.levels, src.values)
            @test decode(lvl, encode(lvl, 0.42)) == decode(src, encode(src, 0.42))
            @test_throws ArgumentError LevelEncoder(src.levels, 1:3)   # length mismatch
            # analytic decoding is FPE-only
            @test_throws ArgumentError decode(lvl, encode(lvl, 0.5); method = :analytic)
            @test_throws ArgumentError decode(lvl, encode(lvl, 0.5); method = :nonsense)
        end

        @testset "bounds and argument checks" begin
            lvl = LevelEncoder(BinaryHV, (0, 1), 10; D = 200, seed = 1)
            @test encode(lvl, 3.0) == lvl.levels[end]   # snaps to nearest by default
            @test_throws DomainError encode(lvl, 3.0; testbound = true)
            @test encode(lvl, 0.5; testbound = true) isa BinaryHV
            @test_throws ArgumentError LevelEncoder(BinaryHV, (0, 1), 1)
            @test_throws ArgumentError LevelEncoder(BinaryHV, (0, 1), 10; bandwidth = 0)
            @test_throws ArgumentError LevelEncoder(BinaryHV, (0, 1), 10; bandwidth = 1.5)
        end
    end
end
