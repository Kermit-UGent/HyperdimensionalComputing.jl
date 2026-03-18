using HyperdimensionalComputing
using Documenter
using Pkg, Literate, Glob

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

# Compile Literate.jl examples to markdown
TUTORIALS = joinpath(@__DIR__, "src", "examples")
SOURCE_FILES = Glob.glob("*.jl", TUTORIALS)
foreach(fn -> Literate.markdown(fn, TUTORIALS), SOURCE_FILES)

# Setup Documenter.jl
DocMeta.setdocmeta!(
    HyperdimensionalComputing,
    :DocTestSetup,
    :(using HyperdimensionalComputing); recursive = true
)

# Get repository information dynamically for fork support
repo_url = get(ENV, "GITHUB_REPOSITORY", "michielstock/HyperdimensionalComputing.jl")
repo_name = split(repo_url, "/")[end]
repo_owner = split(repo_url, "/")[1]

makedocs(;
    modules = [HyperdimensionalComputing],
    authors = "Carlos Vigil-Vásquez, Dimi Boeckaerts, Michiel Stock, Steff Taelman",
    repo = "https://github.com/$repo_url/blob/{commit}{path}#{line}",
    sitename = "HyperdimensionalComputing.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://$repo_owner.github.io/$repo_name",
        assets = String[],
        edit_link = "main",
    ),
    pages = [
        "HyperdimensionalComputing.jl" => "index.md",
        "Examples" => [
            "Introduction to HDC" => "examples/introduction-to-hdc.md",
            "What's the Dollar of Mexico?" => "examples/whats-the-dollar-of-mexico.md",
            "Iris dataset" => "examples/iris.md",
        ],
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/$repo_url",
    devbranch = begin
        current_branch = get(ENV, "GITHUB_REF", "refs/heads/main")
        dev_branch = if occursin("develop", current_branch)
            "develop"
        else
            "main"
        end
    end,
)
