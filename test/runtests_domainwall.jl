# Focused solver/domain-wall suite, also used by the package and CI runners.
# Select CPU/CUDA through JACC preferences; LDO_TEST_MPI enables MPI tests.
using Test
import JACC
JACC.@init_backend

@testset "Domain-wall operators and solvers" begin
    include("cg_cleanup.jl")
    include("domainwall_mpialattice.jl")
    include("domainwall_coefficients.jl")
    include("domainwall_physical_coefficients.jl")
    include("domainwall_eo.jl")
    include("domainwall_grid_reference.jl")
end
