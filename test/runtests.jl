using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra
import LatticeMatrices

@testset "LatticeDiracOperators.jl" begin

    include("public_api.jl")
    include("rhmc_tables.jl")
    include("rhmc_backends.jl")
    include("sakurai_sugiura_generalfermion.jl")
    include("gamma5_2d.jl")
    include("gamma5.jl")
    include("bicg_cleanup.jl")
    include("wilson_action_cleanup.jl")
    include("wilson_mpilattice_boundaries.jl")

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
        include("lattice_staggered_fermi_action.jl")
        include("hisq_hmc_example.jl")
        include("wilson_clover_wrapper.jl")
        include("wilson_cg_action.jl")
        include("clover_migration.jl")
        include("staggered_mpialattice.jl")
        withenv("LDO_TEST_EXPECT_NO_ENZYME" => "true") do
            include("hisq_mpialattice.jl")
        end
        withenv("LDO_EO_BENCH_REPEATS" => get(ENV, "LDO_EO_BENCH_REPEATS", "1")) do
            include("runtests_domainwall.jl")
        end
    end

end
