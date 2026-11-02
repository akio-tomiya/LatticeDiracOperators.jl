# Tests

Run from the repository root. The ordinary package test includes all added
regressions; no workspace-specific runner, scheduler, or sibling directory is
required.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

For just the solver and domain-wall changes, use the same entry point as the
core and MPI CI runners:

```sh
julia --startup-file=no --project=. test/runtests_domainwall.jl
julia --startup-file=no --threads=4 --project=. test/runtests_domainwall.jl
```

This entry point contains six test files:

| File | Numerical checks |
| --- | --- |
| `cg_cleanup.jl` | Scratch release after convergence, allocation failures, and solver exceptions |
| `domainwall_mpialattice.jl` | Standard five-dimensional field and operator backend |
| `domainwall_coefficients.jl` | Renewed vs rebuilt operators, physical/PV caches, adjoints, actions and forces |
| `domainwall_physical_coefficients.jl` | Coefficient-dependent physical source, mres gradients vs finite differences, one learning update |
| `domainwall_eo.jl` | EO/dense/non-EO agreement, halo exchange, force finite differences |
| `domainwall_grid_reference.jl` | Independent Grid PP and J5qP reference |

Individual files can also be run with `julia --project=. test/<file>.jl`.
The numerical checks are the same for each selected JACC backend.

For MPI and GPU environment setup, use the existing [EO guide](eo/README.md).
Replace `test/domainwall_eo.jl` in its one-process or MPI command with
`test/runtests_domainwall.jl` to run the complete focused suite. For example,
with its MPI environment prepared and JACC configured for threads:

```sh
LDO_TEST_MPI=true LDO_TEST_EXPECT_RANKS=2 \
julia --startup-file=no --project=test/eo -e 'using MPI; run(`$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --startup-file=no --project=test/eo test/runtests_domainwall.jl`)'
```

For CUDA, after installing CUDA and selecting the JACC CUDA backend in that
environment, run with explicit GPU and scalar-indexing checks:

```sh
LDO_TEST_REQUIRE_GPU=true julia --startup-file=no --project=test/eo -e '
using CUDA
@assert CUDA.functional()
CUDA.allowscalar(false)
include("test/runtests_domainwall.jl")
'
```

When reviewing local dependency changes, register their checkout paths with
`Pkg.develop` in the chosen environment before running the tests. Do not commit
generated manifests, local preferences, or calculation logs in the PR.
