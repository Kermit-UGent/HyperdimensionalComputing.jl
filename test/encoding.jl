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

    @testset "RandomProjection" begin
        x = [0.9, -0.2, 0.4]
        far = [-1.5, 2.0, -0.3]

        indomain = Dict(
            BinaryHV => hv -> all(b -> b == 0 || b == 1, collect(hv)),
            BipolarHV => hv -> all(b -> b == -1 || b == 1, collect(hv)),
            TernaryHV => hv -> all(b -> b in (-1, 0, 1), collect(hv)),
            RealHV => hv -> all(isfinite, collect(hv)),
            GradedHV => hv -> all(b -> 0 ≤ b ≤ 1, collect(hv)),
            GradedBipolarHV => hv -> all(b -> -1 ≤ b ≤ 1, collect(hv)),
            FHRR => hv -> all(z -> abs(z) ≈ 1, hv.v),
        )

        @testset "comparability and locality: $HV" for HV in keys(indomain)
            rp = RandomProjection(HV, 3; D = 2_000, seed = 21)
            # separate encode calls share R, so they are comparable and
            # deterministic ...
            @test encode(rp, x) == encode(rp, x)
            # ... and nearby feature vectors are more similar than distant ones
            @test similarity(encode(rp, x), encode(rp, x .+ 0.02)) >
                similarity(encode(rp, x), encode(rp, far))
        end

        @testset "matrix distributions stay in domain: $HV" for HV in keys(indomain)
            for matrix in (:gaussian, :bipolar, :sparse_ternary)
                rp = RandomProjection(HV, 3; D = 500, matrix, seed = 22)
                @test indomain[HV](encode(rp, x))
            end
        end

        @testset "supplied matrix" begin
            rp = RandomProjection(BipolarHV, 3; D = 200, seed = 6)
            # reusing the drawn matrix reproduces identical encodings
            rp2 = RandomProjection(BipolarHV, rp.R)
            @test encode(rp2, x) == encode(rp, x)
            # d and D are inferred from the supplied matrix
            @test size(rp2.R) == (200, 3)
            # the ternary supplied-matrix path (no target_sparsity keyword)
            rpt = RandomProjection(TernaryHV, rp.R; θ = 0.1)
            @test rpt.θ == 0.1
            @test encode(rpt, x) isa TernaryHV
            # a zero projection row must not throw: z = 0 has a defined image
            rpz = RandomProjection(BipolarHV, [0.0 0.0 0.0; 1.0 -2.0 0.5])
            @test collect(encode(rpz, x))[1] == -1
            @test collect(encode(RandomProjection(TernaryHV, [0.0 0.0 0.0]), x))[1] == 0
        end

        @testset "sparse_ternary matrix is sparse and ternary" begin
            R = HyperdimensionalComputing.projection_matrix(:sparse_ternary, 1_000, 9; rng = Xoshiro(5))
            @test all(r -> r in (-1.0, 0.0, 1.0), R)
            # nonzero density 1/√d = 1/3
            @test count(!iszero, R) / length(R) ≈ 1 / 3 atol = 0.03
            @test_throws ArgumentError RandomProjection(BinaryHV, 3; matrix = :nope)
        end

        @testset "ternary data-driven threshold" begin
            X = randn(Xoshiro(1), 4, 100)
            rp = RandomProjection(TernaryHV, X; target_sparsity = 0.7, D = 1_000, seed = 2)
            encoded = encode(rp, X)   # per-column, all comparable
            @test length(encoded) == 100
            @test encoded[7] == encode(rp, X[:, 7])
            zerofrac = sum(hv -> count(iszero, collect(hv)), encoded) / (1_000 * 100)
            @test zerofrac ≈ 0.7 atol = 0.01
            @test !ismutable(rp)   # construction from data, not fitting
            @test_throws ArgumentError RandomProjection(TernaryHV, X; target_sparsity = 1.2)
        end

        @testset "ternary constructor: positional collision" begin
            # `RandomProjection(TernaryHV, M)` reads M as a supplied projection
            # matrix; `RandomProjection(TernaryHV, M; target_sparsity)` reads M
            # as training data. Same positional shape, opposite meanings of the
            # matrix — these tests pin the documented reading of each path and
            # that a misread cannot silently encode the intended features.
            R = randn(Xoshiro(11), 50, 4)
            x4 = [0.3, -1.0, 0.8, 0.1]

            # supplied-matrix path: D and d are read off size(R), and encode
            # projects through R itself (R is not treated as data)
            rp = RandomProjection(TernaryHV, R; θ = 0.2)
            @test size(rp.R) == (50, 4)
            z = R * x4
            @test collect(encode(rp, x4)) == Int.(sign.(z) .* (abs.(z) .> 0.2))

            # data-driven path: M is d × n data, so the drawn R has D rows and
            # size(M, 1) columns, and the sparsity target is hit
            X = randn(Xoshiro(12), 4, 60)
            rpd = RandomProjection(TernaryHV, X; target_sparsity = 0.3, D = 500, seed = 13)
            @test size(rpd.R) == (500, 4)
            zerofrac = sum(hv -> count(iszero, collect(hv)), encode(rpd, X)) / (500 * 60)
            @test zerofrac ≈ 0.3 atol = 0.01

            # the dangerous misread: data passed WITHOUT target_sparsity is
            # taken as a projection matrix (here D = 4, d = 60). That reading
            # is coherent on its own terms, but the intended 4-feature vectors
            # CANNOT be encoded through it — the mistake surfaces as an
            # immediate error at first use, never as silently wrong
            # hypervectors of the intended features
            oops = RandomProjection(TernaryHV, X)
            @test size(oops.R) == (4, 60)
            @test_throws DimensionMismatch encode(oops, x4)
            @test encode(oops, randn(Xoshiro(14), 60)) isa TernaryHV
            # NOTE: a *square* matrix is genuinely ambiguous — both readings
            # yield a working encoder — and cannot be caught by tests without
            # changing the constructor. Tracked as a design item in TODO.md.
        end

        @testset "scalar and vector θ, rethreshold" begin
            rp = RandomProjection(TernaryHV, 3; D = 400, θ = 0.5, seed = 8)
            rpv = rethreshold(rp, fill(0.5, 400))
            @test rpv.R === rp.R   # rethreshold shares the projection matrix
            @test encode(rpv, x) == encode(rp, x)   # uniform vector θ ≡ scalar θ
            # a genuinely per-component θ changes only its components
            rp0 = rethreshold(rp, 0.0)
            @test encode(rethreshold(rp, zeros(400)), x) == encode(rp0, x)
            @test_throws ArgumentError rethreshold(rp, ones(7))   # must have length D
            @test_throws ArgumentError RandomProjection(BinaryHV, 3; β = -1)
        end

        @testset "FHRR is random Fourier features (shared phase helper)" begin
            rff = RandomProjection(FHRR, 3; D = 500, β = 0.5, seed = 4)
            # the nonlinearity IS the shared helper, not a copy of it
            @test encode(rff, x) == HyperdimensionalComputing.phase_encode(rff.R * x, 0.5)
            # ... which LevelEncoder's fractional power path also routes through
            fpe = LevelEncoder(FHRR, 0:0.1:1; D = 500, seed = 4)
            @test encode(fpe, 0.3) ==
                HyperdimensionalComputing.phase_encode(angle.(fpe.base.v), fpe.bandwidth * 0.3)
            # similarity approximates the Gaussian kernel exp(-β²‖x-y‖²/2)
            kern = RandomProjection(FHRR, 3; β = 0.5, seed = 1)
            y = x .+ [0.2, 0.0, 0.0]
            @test similarity(encode(kern, x), encode(kern, y)) ≈ exp(-(0.5 * 0.2)^2 / 2) atol = 0.02
        end

        @testset "decode is clean-up, never inversion" begin
            X = randn(Xoshiro(3), 3, 20)
            rp = RandomProjection(BipolarHV, 3; D = 1_000, seed = 9)
            refs = encode(rp, X)
            τ, i, nn = decode(rp, refs[13], refs)
            @test i == 13 && nn === refs[13] && τ ≈ 1
            # noisy queries still clean up to the right reference
            @test decode(rp, encode(rp, X[:, 13] .+ 0.01), refs)[2] == 13
            # no codebook, no decode: a random projection is lossy
            @test_throws ArgumentError decode(rp, refs[1])
            @test_throws ArgumentError decode(rp, refs[1], refs; method = :analytic)
            @test_throws ArgumentError decode(rp, refs[1], refs; method = :nonsense)
        end

        @testset "argument checks" begin
            rp = RandomProjection(BinaryHV, 3; D = 100, seed = 10)
            @test_throws DimensionMismatch encode(rp, [1.0, 2.0])
            @test_throws ArgumentError RandomProjection(BinaryHV, 0)
        end
    end
end
