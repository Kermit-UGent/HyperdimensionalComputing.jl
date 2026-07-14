using Random
using Distributions

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
                @test bundle((encode(HV, i; D = N) for i in 1:5)) isa HV
                @test hv1 + hv2 isa HV
                @test +((encode(HV, i; D = N) for i in 1:5)...) isa HV
                @test +((encode(HV, i; D = N) for i in 1:5)...) == bundle((encode(HV, i; D = N) for i in 1:5))

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
                @test bind([encode(HV, i; D = N) for i in 1:5]) isa HV
                @test *([encode(HV, i; D = N) for i in 1:5]...) isa HV
                @test bind([encode(HV, i; D = N) for i in 1:5]) == *([encode(HV, i; D = N) for i in 1:5]...) isa HV
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

    # regression (TODO §1.5): bundle and bind used to silently replace a custom
    # element distribution with the default, which changes normalize! numerics
    @testset "bundle/bind preserve distr" begin
        for (HV, d) in [
                (RealHV, Normal(0, 5)),
                (GradedHV, Beta(10, 2)),
                (GradedBipolarHV, 2Beta(10, 2) - 1),
            ]
            x = HV(; D = 100, distr = d, seed = 1)
            y = HV(; D = 100, distr = d, seed = 2)
            @test bundle([x, y]).distr === x.distr
            @test (x + y).distr === x.distr
            @test bind(x, y).distr === x.distr
            @test (x * y).distr === x.distr
        end

        # the observable consequence: normalize! rescales to the ORIGINAL spread —
        # the metadata assertion alone would not catch this
        x = RealHV(; D = 10_000, distr = Normal(0, 5), seed = 1)
        y = RealHV(; D = 10_000, distr = Normal(0, 5), seed = 2)
        @test isapprox(std(collect(normalize!(x + y))), 5; rtol = 0.1)
        @test isapprox(std(collect(normalize!(x * y))), 5; rtol = 0.1)
    end

    # regression (TODO §1.5c): perturbate used to draw replacement elements from
    # the TYPE-default distribution, ignoring the instance's `distr` — right
    # metadata, wrong-distribution elements
    @testset "perturbate resamples from hv.distr" begin
        I = 1:5_000
        x = RealHV(; D = 10_000, distr = Normal(0, 5), seed = 1)
        xp = perturbate(x, I; rng = Xoshiro(1))
        @test isapprox(std(xp.v[I]), 5; rtol = 0.1)

        g = GradedHV(; D = 10_000, distr = Beta(10, 2), seed = 1)
        gp = perturbate(g, I; rng = Xoshiro(1))
        @test isapprox(mean(gp.v[I]), mean(Beta(10, 2)); rtol = 0.05)

        gb = GradedBipolarHV(; D = 10_000, distr = 2Beta(10, 2) - 1, seed = 1)
        gbp = perturbate(gb, I; rng = Xoshiro(1))
        @test isapprox(mean(gbp.v[I]), mean(2Beta(10, 2) - 1); rtol = 0.1)
    end

    # regression (TODO §1.6): shift! and the clamp!-based normalize! methods
    # used to return the raw wrapped vector instead of the hypervector
    @testset "in-place operations return the hypervector" begin
        for HV in [BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR]
            hv = HV(; D = 20)
            @test shift!(hv, 3) === hv
            @test ρ!(hv) === hv
            @test normalize!(hv) === hv
            # every perturbate! argument form: count, fraction, mask, indices
            @test perturbate!(hv, 2) === hv
            @test perturbate!(hv, 0.1) === hv
            @test perturbate!(hv, bitrand(length(hv))) === hv
            @test perturbate!(hv, [1, 3]) === hv
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

        # regression (TODO §1.3): perturbate used to throw a MethodError for FHRR;
        # phases must be resampled so elements stay on the unit circle
        hvp = perturbate(hv1, 10)
        @test hvp isa FHRR
        @test all(abs.(hvp.v) .≈ 1)
        @test count(hvp.v .!= hv1.v) == 10   # exactly n positions resampled
        hvf = perturbate(hv1, 0.2)
        @test hvf isa FHRR
        @test all(abs.(hvf.v) .≈ 1)
        # untouched positions carry similarity 1, resampled ones ≈ 0 in
        # expectation, so similarity degrades to roughly 1 - p
        hvbig = FHRR(; D = 10_000)
        @test similarity(hvbig, perturbate(hvbig, 0.2)) ≈ 0.8 atol = 0.05
        @test similarity(hvbig, perturbate(hvbig, 0.5)) ≈ 0.5 atol = 0.05
        m = bitrand(length(hv1))
        hvm = perturbate(hv1, m; rng = Xoshiro(1))
        @test all(hvm.v[.!m] .== hv1.v[.!m])
        @test all(abs.(hvm.v[m]) .≈ 1)
        # `level` for FHRR uses its own ^-based method, not perturbation
        @test level(FHRR(; D = 100), 5) isa Vector{<:FHRR}
    end

end
