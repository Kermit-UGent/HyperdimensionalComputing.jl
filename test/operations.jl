using Random

Random.seed!(42)
@testset "operations" begin
    hv_types = [
        BinaryHV, BipolarHV, RealHV, TernaryHV,
        GradedHV, GradedBipolarHV,
    ]

    for HV in hv_types

        N = 500

        @testset "operations $HV" begin

            hv1 = HV(; D = N)
            hv2 = HV(; D = N)

            v1 = collect(hv1)
            v2 = collect(hv2)

            @testset "bundle $HV" begin
                @test bundle((hv1, hv2)) isa HV
                @test bundle([hv1, hv2]) isa HV
                @test bundle((HV(i; D = N) for i in 1:5)) isa HV
                @test hv1 + hv2 isa HV
                @test +((HV(i; D = N) for i in 1:5)...) isa HV
                @test +((HV(i; D = N) for i in 1:5)...) == bundle((HV(i; D = N) for i in 1:5))

                if HV <: Union{BinaryHV, BipolarHV}
                    @test (hv1 + hv2).v == (hv1 + hv2).v
                    @test bundle([hv1, hv2]).v == bundle([hv1, hv2]).v
                    hv3 = HV(; D = N)
                    @test bundle([hv1, hv2, hv3]).v == bundle([hv1, hv2, hv3]).v
                    @test bundle([hv1, hv2]; rng = Xoshiro(1)).v ==
                        bundle([hv1, hv2]; rng = Xoshiro(1)).v
                end
            end

            @testset "bind $HV" begin
                @test bind(hv1, hv2) isa HV
                @test bind([hv1, hv2]) isa HV
                @test bind([HV(i; D = N) for i in 1:5]) isa HV
                @test *([HV(i; D = N) for i in 1:5]...) isa HV
                @test bind([HV(i; D = N) for i in 1:5]) == *([HV(i; D = N) for i in 1:5]...) isa HV
                @test hv1 * hv2 isa HV
            end

            @testset "shift $HV" begin
                @test shift(hv1, 3) ≈ circshift(v1, 3)
                @test shift!(hv2, -8) ≈ circshift(v2, -8)
                @test ρ(hv1, 2) ≈ circshift(v1, 2)
            end

            @testset "perturbate $HV" begin
                @test perturbate(hv1, 10) isa HV
                @test perturbate(hv2, 0.2) isa HV
                hvp = perturbate(hv1, [4, 8])
                @test hvp.v[[1, 2, 3]] ≈ hv1.v[[1, 2, 3]]

                m = bitrand(length(hv1))
                hvp = perturbate(hv2, m)
                @test hv2.v[m] != hvp.v[m]
                @test hv2.v[.!m] ≈ hvp.v[.!m]

                @test perturbate(hv1, 0.1, rng = Xoshiro(1)) == perturbate(hv1, 0.1, rng = Xoshiro(1))
                @test perturbate(hv1, 0.1, rng = Xoshiro(1)) != perturbate(hv1, 0.1, rng = Xoshiro(2))
            end

            # currently not yet a good way of evaluating these
            HV <: Union{TernaryHV, GradedHV, GradedBipolarHV, RealHV} && continue

            @testset "similarity $HV" begin
                N = 10_000
                hv1 = HV(; D = N)
                hv2 = HV(; D = N)
                hv3 = HV(; D = N) + hv1  # similar to 1 but not to 2
                normalize!(hv3)

                @test !(hv1 ≈ hv2)

                @test !(hv1 == hv2)
                @test (hv3 ≈ hv1)
                @test !(hv3 ≈ hv2)
            end
        end
    end

    @testset "unbind" begin
        # XOR- and multiplication-based types: binding is self-inverse, roundtrip exact
        for HV in [BinaryHV, BipolarHV, TernaryHV]
            x, y = HV(:x), HV(:y)
            @test unbind(bind(x, y), y) == x
            @test (x * y) / y == x
        end

        # graded types: fuzzy unbinding is approximate — the recovered vector is
        # closer to the original than to an unrelated one
        for HV in [GradedHV, GradedBipolarHV]
            x, y, z = HV(:x), HV(:y), HV(:z)
            recovered = (x * y) / y
            @test recovered isa HV
            @test similarity(recovered, x) > similarity(recovered, z)
        end

        # FHRR: exact inverse via elementwise complex division
        x, y = FHRR(:x), FHRR(:y)
        @test collect((x * y) / y) ≈ collect(x)

        # RealHV: real-valued MAP binding is not exactly invertible — explicit error
        r1, r2 = RealHV(:x), RealHV(:y)
        @test_throws ArgumentError unbind(r1, r2)
        @test_throws ArgumentError r1 / r2
        @test_throws "not exactly invertible" unbind(r1, r2)
    end

    @testset "FHRR" begin
        hv1 = FHRR(; D = n)
        hv2 = FHRR(; D = n)

        @test bundle([hv1, hv2]) isa FHRR
        @test hv1 + hv2 isa FHRR
        @test bind([hv1, hv2]) isa FHRR
        @test norm(bind([hv1, hv2])) ≈ sqrt(n)

        @test shift(hv1, 2) isa FHRR

        @test similarity(hv1, hv2) < 0.5
        @test similarity(hv2, hv2) ≈ 1

        @test norm(hv1^3) ≈ sqrt(n)
    end

end
