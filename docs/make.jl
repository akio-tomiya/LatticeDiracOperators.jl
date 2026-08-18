using LatticeDiracOperators
using Documenter

DocMeta.setdocmeta!(LatticeDiracOperators, :DocTestSetup, :(using LatticeDiracOperators); recursive=true)

makedocs(;
    modules=[LatticeDiracOperators],
    checkdocs=:none,
    authors="Akio Tomiya, Yuki Nagai <cometscome@gmail.com> and contributors",
    repo="https://github.com/akio-tomiya/LatticeDiracOperators.jl/blob/{commit}{path}#{line}",
    sitename="LatticeDiracOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://akio-tomiya.github.io/LatticeDiracOperators.jl/v1/",
        repolink="https://github.com/akio-tomiya/LatticeDiracOperators.jl",
        edit_link="master",
        size_threshold_warn=150 * 2^10,
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "Quick start" => "quickstart.md",
        ],
        "Fermion formulations" => [
            "Wilson and Wilson--clover" => "wilson.md",
            "Staggered and HISQ" => "staggered_hisq.md",
            "Domain-wall fermions" => "domainwall.md",
            "User-defined operators" => "generalfermion.md",
        ],
        "Guides" => [
            "Actions, forces, and solvers" => "actions_forces.md",
            "MPI, GPU, and multi-GPU" => "mpi_gpu.md",
        ],
        "Reference" => [
            "High-level API parameters" => "highlevelapi.md",
            "Public v1 API index" => "publicapi.md",
            "Citing LDO" => "references.md",
        ],
        "Compatibility" => [
            "v1 compatibility boundary" => "v1_api.md",
            "Legacy API and examples" => "howtouse.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/akio-tomiya/LatticeDiracOperators.jl",
    devbranch="master",
    versions=[
        "stable" => "v^",
        "v#",
        "v#.#",
        "v#.#.#",
        "dev" => "dev",
    ],
)
