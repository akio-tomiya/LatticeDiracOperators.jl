using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra
import LatticeMatrices

@testset "LatticeDiracOperators.jl" begin

    include("public_api.jl")

    @testset "Solver diagnostics" begin
        include("solver_diagnostics.jl")
    end

    @testset "Wilson boundary conditions" begin
        include("wilson_boundary_conditions.jl")
    end

    include("readme_examples.jl")

    include("z4_noise.jl")
    include("pseudofermion_md_action.jl")
    include("mpi_optional.jl")

    @testset "nHYP staggered HMC" begin
        include("nhyp_staggered_hmc.jl")
    end

    if isdefined(LatticeMatrices, :D5DW_GeneralizedDomainwallOperator5D)
        include("latticematrices_backend.jl")
        include("wilson_clover_wrapper.jl")
        include("staggered_mpialattice.jl")
        withenv("LDO_TEST_EXPECT_NO_ENZYME" => "true") do
            include("hisq_mpialattice.jl")
        end
        include("domainwall_mpialattice.jl")
        include("domainwall_grid_reference.jl")
    end

end
