using LatticeDiracOperators
using Documenter

DocMeta.setdocmeta!(LatticeDiracOperators, :DocTestSetup, :(using LatticeDiracOperators); recursive=true)

makedocs(;
    modules=[LatticeDiracOperators],
    authors="cometscome <cometscome@gmail.com> and contributors",
    repo="https://github.com/cometscome/LatticeDiracOperators.jl/blob/{commit}{path}#{line}",
    sitename="LatticeDiracOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cometscome.github.io/LatticeDiracOperators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/cometscome/LatticeDiracOperators.jl",
    devbranch="main",
)
