const n = 100
const s = :test
const hash_s = hash(s)

using Distributions
using Logging: Logging

@testset "types" begin
    # The constructor convention shared by all hypervector types:
    # `HV(this)` encodes an object deterministically via `hash(this)` — the
    # positional argument is never a dimension; dimensionality is set with `D`.
    @testset "constructor convention $HV" for HV in [
            BinaryHV, BipolarHV, TernaryHV, RealHV,
            GradedHV, GradedBipolarHV, FHRR,
        ]
        @test length(HV(42)) == 10_000  # 42 is a token, not a dimension
        @test HV(42) == HV(42)          # deterministic per object
        @test HV(:cat) == HV(:cat)
        @test HV(:cat) != HV(:dog)
        @test length(HV(:cat; D = n)) == n
        @test length(HV(; D = n)) == n

        # encoding an Integer warns that it is not a dimension; other tokens don't
        @test_logs (:warn, r"never the dimensionality") match_mode = :any HV(42)
        @test_logs min_level = Logging.Warn HV(:cat)
    end

    # Distinct objects give quasi-orthogonal hypervectors at the default D.
    @testset "quasi-orthogonality of tokens" begin
        for HV in [BipolarHV, RealHV, FHRR]  # cosine-type similarity: ≈ 0
            @test abs(similarity(HV(:cat), HV(:dog))) < 0.05
        end
        # Jaccard similarity of random binary vectors has baseline ≈ 1/3
        @test isapprox(similarity(BinaryHV(:cat), BinaryHV(:dog)), 1 / 3; atol = 0.05)
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
        # sign-based construction; Bool vectors are raw stored bits (true ↦ -1)
        @test collect(BipolarHV([-1, 1])) == [-1, 1]
        @test collect(BipolarHV([-2.5, 0.1])) == [-1, 1]
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

        hv2 = GradedBipolarHV([0.1, 1.12, -0.2, -3.0])
        normalize!(hv2)
        @test all(-1 .≤ hv2 .≤ 1)

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

        hv2 = GradedHV([0.1, 1.12, -0.2, -3.0])
        normalize!(hv2)
        @test all(0 .≤ hv2 .≤ 1)
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
