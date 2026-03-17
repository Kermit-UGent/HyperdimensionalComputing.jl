const n = 100
const s = :test
const hash_s = hash(s)

using Distributions, LinearAlgebra

@testset "types" begin
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
        @test BipolarHV([-1, 0, 1]) == BipolarHV([false, false, true])
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
