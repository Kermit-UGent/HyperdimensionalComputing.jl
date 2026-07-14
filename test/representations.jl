using Random: Xoshiro

@testset "representations (plain display)" begin
    # These tests are only valid while the UnicodePlots extension is NOT loaded;
    # the rich-display tests run in a separate process (see ext_display.jl).
    @test Base.get_extension(HyperdimensionalComputing, :UnicodePlotting) === nothing

    plain(hv) = repr(MIME"text/plain"(), hv)

    @testset "headers" begin
        @test occursin(r"^100-element BinaryHV with \d+ true and \d+ false:", plain(BinaryHV(; D = 100)))
        @test occursin(r"^100-element BipolarHV with \d+ positives and \d+ negatives:", plain(BipolarHV(; D = 100)))
        @test occursin(r"^100-element TernaryHV\{Int64\} with \d+ positives, \d+ zeros, and \d+ negatives:", plain(TernaryHV(; D = 100)))
        for HV in [RealHV, GradedHV, GradedBipolarHV, FHRR]
            @test occursin(Regex("^100-element $HV.* with μ ± σ = "), plain(HV(; D = 100)))
        end
    end

    @testset "elements and truncation" begin
        hv = BipolarHV(; D = 19, seed = 1, rng = Xoshiro)

        # untruncated: header plus one line per element, printed by Base
        s = plain(hv)
        @test length(split(s, '\n')) == 20
        @test occursin(r"\n\s+-?1", s)

        # Base's array machinery handles ⋮ truncation in limited displays
        buf = IOBuffer()
        show(IOContext(buf, :limit => true, :displaysize => (10, 80)), MIME"text/plain"(), hv)
        slim = String(take!(buf))
        @test occursin("19-element BipolarHV with", slim)
        @test occursin("⋮", slim)
    end

    @testset "vectors of hypervectors" begin
        vs = [BinaryHV(; D = 10) for _ in 1:3]
        sv = repr(MIME"text/plain"(), vs)
        @test occursin("3-element Vector{BinaryHV}:", sv)
        @test count("-element BinaryHV with", sv) == 3
    end

    @testset "getindex" begin
        # BipolarHV has a custom getindex mapping stored bits to ±1
        hv = BipolarHV([true, false, true, true, false])
        @test hv[1] === 1
        @test hv[2] === -1
        @test hv[end] === -1
        @test hv[2:4] == [-1, 1, 1]
        @test hv[2:4] isa Vector{Int}
        @test hv[[1, 5]] == [1, -1]
        @test hv[[true, false, true, false, false]] == [1, 1]

        # uniform across all types: non-scalar indexing returns element values
        # in a plain vector, never a new hypervector
        for HV in [BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR]
            h = HV(; D = 20)
            @test h[3] == collect(h)[3]
            @test h[2:6] == collect(h)[2:6]
            @test h[[1, 10, 20]] == collect(h)[[1, 10, 20]]
            @test h[2:6] isa AbstractVector
            @test !(h[2:6] isa AbstractHV)
        end
    end
end
