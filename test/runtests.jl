using HyperdimensionalComputing
using Test

include("types.jl")
include("operations.jl")
include("encoding.jl")
include("inference.jl")
include("representations.jl")

# The rich-display tests load UnicodePlots, which cannot be unloaded again,
# while the plain-display tests above require the extension to be absent —
# so the extension tests run in a fresh Julia process.
@testset "UnicodePlots extension (separate process)" begin
    script = joinpath(@__DIR__, "ext_display.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script`
    @test success(run(ignorestatus(cmd)))
end
