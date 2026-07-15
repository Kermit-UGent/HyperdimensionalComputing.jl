const n = 100
const s = :test
const hash_s = hash(s)

using Distributions

@testset "types" begin
    # `encode(HV, x)` is the canonical token path; `HV(x)` is shorthand for
    # non-Number tokens, and Number arguments throw (irreducibly ambiguous
    # with the dimensionality).
    @testset "encode and constructor sugar $HV" for HV in [
            BinaryHV, BipolarHV, TernaryHV, RealHV,
            GradedHV, GradedBipolarHV, FHRR,
        ]
        @test encode(HV, 42) == encode(HV, 42)      # deterministic per object
        @test encode(HV, "cat") == encode(HV, "cat")
        @test length(encode(HV, 42)) == 10_000
        @test length(encode(HV, :cat; D = n)) == n
        @test HV(:cat) == encode(HV, :cat)          # the sugar is genuinely sugar
        @test HV("cat") == encode(HV, "cat")
        @test HV(:cat) != HV(:dog)
        @test length(HV(:cat; D = n)) == n
        @test length(HV(; D = n)) == n

        # Numbers throw, and the message teaches both alternatives
        @test_throws ArgumentError HV(5)
        @test_throws "D = 5" HV(5)
        @test_throws "encode" HV(5)
        @test_throws ArgumentError HV(2.5)
        @test_throws ArgumentError HV(true)   # Bool is a Number
    end

    # Distinct objects give quasi-orthogonal hypervectors at the default D.
    @testset "quasi-orthogonality of tokens" begin
        for HV in [BipolarHV, RealHV, FHRR]  # cosine-type similarity: ≈ 0
            @test abs(similarity(HV(:cat), HV(:dog))) < 0.05
        end
        # Jaccard similarity of random binary vectors has baseline ≈ 1/3
        @test isapprox(similarity(BinaryHV(:cat), BinaryHV(:dog)), 1 / 3; atol = 0.05)
    end

    # Data constructors: one meaning, validated per element domain.
    @testset "data constructors and validation" begin
        # the regression that motivated the encode interface: this used to
        # silently token-encode into a 10,000-element hypervector
        @test length(BinaryHV([1, 0])) == 2
        @test collect(BinaryHV([1, 0])) == [true, false]
        @test BinaryHV([1.0, 0.0]) == BinaryHV(Bool[1, 0])
        @test_throws ArgumentError BinaryHV([1, 2])
        @test_throws ArgumentError BinaryHV([0.5])

        @test collect(BipolarHV([-1, 1])) == [-1, 1]
        @test_throws ArgumentError BipolarHV([1, 0, -1])
        @test_throws "TernaryHV" BipolarHV([1, 0, -1])   # zero points to TernaryHV
        @test_throws ArgumentError BipolarHV([2.5, -1])  # no silent sign-taking

        @test collect(TernaryHV([-1, 0, 1])) == [-1, 0, 1]
        @test TernaryHV([-1.0, 0.0, 1.0]) == TernaryHV([-1, 0, 1])
        @test_throws ArgumentError TernaryHV([2, 0])

        @test RealHV([0.5, -3.7]) isa RealHV   # any real is fine

        @test collect(GradedHV([0.0, 0.5, 1.0])) == [0.0, 0.5, 1.0]
        @test_throws ArgumentError GradedHV([1.12])
        @test_throws ArgumentError GradedHV([-0.2])

        @test collect(GradedBipolarHV([-1.0, 0.5])) == [-1.0, 0.5]
        @test_throws ArgumentError GradedBipolarHV([1.5])

        @test FHRR([im, -1.0 + 0.0im]) isa FHRR
        @test_throws ArgumentError FHRR([2.0 + 0.0im])   # not unit modulus

        # tuples of reals read as data, like vectors (decided deliberately)
        @test BipolarHV((1, -1)) == BipolarHV([1, -1])
        @test length(BinaryHV((1, 0))) == 2

        # arrays that are not valid element data throw instead of silently
        # token-encoding; encode is the explicit escape hatch
        @test_throws ArgumentError BinaryHV(["a", "b"])
        @test encode(BinaryHV, ["a", "b"]) isa BinaryHV
    end

    # Construction and indexing agree for every type.
    @testset "data round-trip $HV" for HV in [
            BinaryHV, BipolarHV, TernaryHV, RealHV,
            GradedHV, GradedBipolarHV, FHRR,
        ]
        x = HV(; D = 20, seed = 11)
        @test HV(collect(x)) == x
    end

    # regression (TODO §1.8): hypervectors of different types used to compare
    # `isequal` whenever their stored bits matched
    @testset "equality and hashing" begin
        x = BinaryHV(; D = 50, seed = 1)
        @test x == BinaryHV(x.v)
        @test isequal(x, BinaryHV(x.v))
        @test hash(x) == hash(BinaryHV(x.v))

        # same stored bits, different type: not equal
        y = BipolarHV(; D = 50, seed = 1)   # same seed ⇒ identical stored bits
        @test x.v == y.v                    # precondition for the regression
        @test !isequal(x, y)
        @test x != y
        @test !isequal(y, x)

        # cross-type equality is strictly false, even when element values
        # coincide numerically (true == 1): since the polarity flip, an all-true
        # BinaryHV and an all-+1 BipolarHV have OPPOSITE stored bits
        @test !isequal(BinaryHV(Bool[1, 0]), BipolarHV([1, -1]))
        @test !isequal(BinaryHV(Bool[1, 1]), BipolarHV([1, 1]))
        @test BinaryHV(Bool[1, 1]) != BipolarHV([1, 1])

        # same family, different element type: compares by value
        @test TernaryHV{Int8}(Int8[1, -1]) == TernaryHV{Int64}([1, -1])
        @test isequal(TernaryHV{Int8}(Int8[1, -1]), TernaryHV{Int64}([1, -1]))

        # hash/equality contract for every type — including BipolarHV, where
        # storage and elements disagree; and against plain vectors, where
        # elementwise isequal can be true and hashes must then match
        for HV in [BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR]
            h = HV(; D = 20, seed = 5)
            h2 = HV(; D = 20, seed = 5)
            @test isequal(h, h2) && hash(h) == hash(h2)
            @test isequal(h, collect(h)) && hash(h) == hash(collect(h))
        end
        p = BipolarHV([1, -1])
        @test hash(p) == hash([1, -1])   # elements, not stored bits
        @test hash(p) != hash(p.v)
    end

    @testset "BipolarHV" begin
        hdv = BipolarHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Int
        @test hdv[2] isa Int
        @test all(-1 .≤ hdv .≤ 1)
        @test hdv == BipolarHV(hdv.v)
        @test similar(hdv) isa BipolarHV
        @test sum(hdv) == sum(vi for vi in hdv)
        @test BipolarHV(s) == BipolarHV(; seed = hash_s)
        # strict ±1 data; Bool vectors are raw stored bits (true ↦ -1)
        @test collect(BipolarHV([-1, 1])) == [-1, 1]
        @test BipolarHV([-1, 1]) == BipolarHV([true, false])
        # zero has no bipolar state
        @test_throws ArgumentError BipolarHV([1, 0, -1])
        @test_throws "no zero state" BipolarHV([1.0, 0.0])
    end

    # These tests are deliberately NOT polarity-blind: XOR self-inversion,
    # quasi-orthogonality and cosine similarity are all invariant under flipping
    # the bit↦element mapping, which is how the polarity bug survived a green
    # suite. Each assertion here pins the mapping itself.
    @testset "BipolarHV polarity" begin
        x = BipolarHV(; D = 100, seed = 7)

        # bind is self-inverse AND the identity is the all-+1 vector
        @test all(collect(x * x) .== 1)

        # construction and indexing agree: values round-trip through the sign constructor
        @test BipolarHV(collect(x)) == x

        # the summary header counts actual element values
        s = summary(x)
        @test occursin("$(count(==(1), collect(x))) positives", s)
        @test occursin("$(count(==(-1), collect(x))) negatives", s)
    end

    # Sharp edges of the `false ↦ +1 / true ↦ -1` storage convention. These
    # lock CURRENT, deliberate behaviour — the convention is load-bearing
    # (XOR on stored bits IS the ±1 product) and each trap below has caused a
    # real bug (the original polarity flip; RandomProjection's bipolar
    # nonlinearity). Do not "fix" these; they are documented-by-test so the
    # obvious-but-wrong idiom trips a red test instead of shipping silent
    # corruption. (Bind identity, construction round-trip and the zero-throws
    # are locked in the two BipolarHV testsets above.)
    @testset "BipolarHV sharp edges" begin
        # `sign.(z)` is NOT a valid real-to-bipolar map: sign(0) = 0, and zero
        # has no bipolar state — this is the trap RandomProjection's bipolar
        # nonlinearity had to avoid
        @test_throws ArgumentError BipolarHV(sign.([0.5, -1.2, 0.0, 2.0]))

        # Bool vectors are RAW STORED BITS, not logical positivity:
        # true ↦ -1, false ↦ +1 — the opposite of what "true is positive"
        # intuition expects, deliberately, so bind-as-XOR is the exact product
        @test collect(BipolarHV([true, false, true])) == [-1, 1, -1]

        # hence the correct thresholding of reals is an explicit ±1 ifelse;
        # the "obvious" BipolarHV(z .> 0) hits the raw-bits path and silently
        # yields the OPPOSITE polarity — both are pinned so the difference is
        # on record
        z = [0.5, -1.2, 2.0]   # no zeros: only polarity is at issue here
        @test collect(BipolarHV(ifelse.(z .> 0, 1, -1))) == [1, -1, 1]
        @test collect(BipolarHV(z .> 0)) == [-1, 1, -1]   # inverted! never reach for this
    end

    @testset "BinaryHV" begin
        hdv = BinaryHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Bool
        @test hdv[2] isa Bool
        @test hdv == BinaryHV(hdv.v)
        @test similar(hdv) isa BinaryHV
        @test sum(hdv) ≈ sum(hdv.v)
        @test BinaryHV(s) == BinaryHV(; seed = hash_s)
    end

    @testset "TernaryHV" begin
        hdv = TernaryHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Int
        @test hdv[2] isa Int
        @test hdv == TernaryHV(hdv.v)
        @test similar(hdv) isa TernaryHV
        @test sum(hdv) ≈ sum(hdv.v)
        @test TernaryHV(s) == TernaryHV(; seed = hash_s)
        for T in [Int8, Int16, Int32, Int64, Int]
            @test eltype(TernaryHV{T}(; D = n)) <: T
            @test TernaryHV{T}() + TernaryHV{T}() isa TernaryHV{T}
            @test TernaryHV{T}() * TernaryHV{T}() isa TernaryHV{T}
            @test shift(TernaryHV{T}()) isa TernaryHV{T}
            @test normalize(TernaryHV{T}()) isa TernaryHV{T}
            @test copy(TernaryHV{T}()) isa TernaryHV{T}
            @test similar(TernaryHV{T}()) isa TernaryHV{T}
        end
    end

    @testset "GradedBipolarHV" begin
        hdv = GradedBipolarHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(-1 .≤ hdv.v .≤ 1)
        @test hdv == GradedBipolarHV(hdv.v)
        @test similar(hdv) isa GradedBipolarHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(GradedBipolarHV(Float32, n)) <: Float32
        @test GradedBipolarHV(s) == GradedBipolarHV(; seed = hash_s)

        @test GradedBipolarHV(; D = n, distr = 2Beta(10, 2) - 1) isa GradedBipolarHV

        # out-of-range data is rejected, not clamped
        @test_throws ArgumentError GradedBipolarHV([0.1, 1.12, -0.2, -3.0])

    end

    @testset "GradedHV" begin
        hdv = GradedHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real
        @test all(0 .≤ hdv.v .≤ 1)
        @test hdv == GradedHV(hdv.v)
        @test similar(hdv) isa GradedHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(GradedHV(Float32, n)) <: Float32
        @test GradedHV(s) == GradedHV(; seed = hash_s)

        @test GradedHV(; D = n, distr = Beta(10, 2)) isa GradedHV

        # out-of-range data is rejected, not clamped
        @test_throws ArgumentError GradedHV([0.1, 1.12, -0.2, -3.0])
    end

    @testset "RealHV" begin
        hdv = RealHV(; D = n)

        @test length(hdv) == n
        @test eltype(hdv) <: Real
        @test hdv[2] isa Real

        @test hdv == RealHV(hdv.v)
        @test similar(hdv) isa RealHV
        @test sum(hdv) ≈ sum(hdv.v)
        #@test eltype(RealHV(Float32, n)) <: Float32
        @test norm(hdv) ≈ norm(hdv.v)
        normalize!(hdv)
        @test RealHV(s) == RealHV(; seed = hash_s)
    end

    @testset "FHRR" begin
        hdv = FHRR(; D = n)
        @test length(hdv) == n
        @test eltype(hdv) <: Complex
        @test hdv[2] isa Complex
        @test sum(hdv) ≈ sum(hdv.v)
        @test norm(hdv) ≈ norm(hdv.v)
        @test FHRR(s) == FHRR(; seed = hash_s)
    end
end
