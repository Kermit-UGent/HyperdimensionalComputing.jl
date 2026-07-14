# Generate the HyperdimensionalComputing.jl logo.
#
# The logo is the Julia trefoil rendered in a "hyperdimensional" style: each of
# the three lobes is a dense hypervector, drawn as a disc of grid-packed dots
# whose sizes are randomized to give a halftone-like texture.
#
# Run with:  julia --project=docs docs/logo.jl
# (or plain `julia docs/logo.jl` if CairoMakie is in your global environment)

using CairoMakie
using Random

# Julia brand colors
const JULIA_GREEN = colorant"#389826"
const JULIA_RED = colorant"#CB3C33"
const JULIA_PURPLE = colorant"#9558B2"

"""
    disc_dots(cx, cy, R; spacing = 1.0)

Return the centers of a square lattice of points (spacing `spacing`) that fall
inside the disc of radius `R` centered at `(cx, cy)`.
"""
function disc_dots(cx, cy, R; spacing = 1.0)
    pts = Point2f[]
    n = ceil(Int, R / spacing)
    for i in (-n):n, j in (-n):n
        x, y = i * spacing, j * spacing
        if x^2 + y^2 <= R^2
            push!(pts, Point2f(cx + x, cy + y))
        end
    end
    return pts
end

"""
    dot_sizes(rng, n; spacing = 1.0)

Randomized marker diameters (in data units) for `n` dots. Most dots are near a
full lattice cell; a minority shrink toward zero to create the halftone look.
"""
function dot_sizes(rng, n; spacing = 1.0)
    s = spacing .* (0.3 .+ 0.7 .* rand(rng, n))   # base range ~[0.30, 1.0]
    tiny = rand(rng, n) .< 0.12                       # ~12% become tiny specks
    s[tiny] .*= 0.25
    return s
end

function make_logo(; seed = 42, spacing = 1.0, R = 6.5, ρ = 9.0)
    rng = MersenneTwister(seed)

    # Lobe centers: equilateral triangle (top / bottom-left / bottom-right).
    lobes = [
        (JULIA_GREEN, Point2f(0.0, ρ)),
        (JULIA_RED, Point2f(-ρ * cosd(30), -ρ * sind(30))),
        (JULIA_PURPLE, Point2f(ρ * cosd(30), -ρ * sind(30))),
    ]

    fig = Figure(; size = (1200, 1200), backgroundcolor = :transparent)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), backgroundcolor = :transparent)
    hidespines!(ax)
    hidedecorations!(ax)

    for (color, c) in lobes
        pts = disc_dots(c[1], c[2], R; spacing = spacing)
        sz = dot_sizes(rng, length(pts); spacing = spacing)
        scatter!(ax, pts; markersize = sz, markerspace = :data, color = color, strokecolor=:transparent)
    end

    limits!(ax, -ρ - R - 1, ρ + R + 1, -ρ * sind(30) - R - 1, ρ + R + 1)
    return fig
end

fig = make_logo()

assets = joinpath(@__DIR__, "src", "assets")
mkpath(assets)
save(joinpath(assets, "logo.svg"), fig)
save(joinpath(assets, "logo.png"), fig; px_per_unit = 2)
@info "Saved logo to $assets"
