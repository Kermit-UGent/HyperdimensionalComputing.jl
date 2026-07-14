# Rich-display tests for the UnicodePlots package extension.
#
# This file runs in a SEPARATE Julia process (see runtests.jl): loading
# UnicodePlots cannot be undone within a session, and the plain-display tests
# in representations.jl require the extension to be absent.

using Test
using HyperdimensionalComputing
using UnicodePlots

@testset "UnicodePlots extension" begin
    # `nothing` here would mean the extension failed to load (e.g. the
    # precompilation failure from defining `show` in both package and extension)
    ext = Base.get_extension(HyperdimensionalComputing, :UnicodePlotting)
    @test ext !== nothing

    hv_types = [BinaryHV, BipolarHV, TernaryHV, RealHV, GradedHV, GradedBipolarHV, FHRR]

    @testset "rich show $HV" for HV in hv_types
        hv = HV(; D = 100)
        s = repr(MIME"text/plain"(), hv)
        @test occursin("-element", s)   # summary header
        @test occursin("┌", s)          # plot borders from UnicodePlots

        # :compact contexts fall back to the plain display
        c = repr(MIME"text/plain"(), hv; context = :compact => true)
        @test !occursin("┌", c)
    end

    @testset "direct plotting API $HV" for HV in hv_types
        hv = HV(; D = 100)
        @test unicodeheatmap(hv) isa UnicodePlots.Plot
        @test unicodehistogram(hv) isa UnicodePlots.Plot
    end

    # regression: displaying a BipolarHV with the extension loaded used to throw
    # a TypeError (range indexing on the BitVector storage)
    @test sprint(show, MIME"text/plain"(), BipolarHV(; D = 100)) isa String
end
