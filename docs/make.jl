using LatticeDiracOperators
using Documenter

DocMeta.setdocmeta!(LatticeDiracOperators, :DocTestSetup, :(using LatticeDiracOperators); recursive=true)

makedocs(;
    modules=[LatticeDiracOperators],
    authors="Akio Tomiya, Yuki Nagai <cometscome@gmail.com> and contributors",
    repo="https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/{commit}{path}#{line}",
    sitename="LatticeDiracOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/akio-tomiya/LatticeDiracOperators.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "How to use" => "howtouse.md",
    ],
)

deploydocs(;
    repo="github.com/akio-tomiya/LatticeDiracOperators.jl",
    devbranch="master",
)
