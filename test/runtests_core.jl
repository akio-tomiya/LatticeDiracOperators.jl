using LatticeDiracOperators
using Gaugefields
using LatticeMatrices
using Test

@testset "LatticeDiracOperators core without Enzyme" begin
    @test Base.find_package("MPI") === nothing
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsEnzymeExt) === nothing
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsMPIExt) === nothing
    @test Base.get_extension(
        LatticeMatrices, :LatticeMatricesEnzymeExt) === nothing
    @test Base.get_extension(Gaugefields, :GaugefieldsMPIExt) === nothing
    @test Base.get_extension(LatticeMatrices, :LatticeMatricesMPIExt) === nothing

    project_directory = dirname(Base.active_project())
    test_files = [
        "public_api.jl",
        "rhmc_tables.jl",
        "rhmc_backends.jl",
        "sakurai_sugiura_generalfermion.jl",
        "gamma5_2d.jl",
        "gamma5.jl",
        "bicg_cleanup.jl",
        "wilson_action_cleanup.jl",
        "readme_examples.jl",
        "solver_diagnostics.jl",
        "wilson_boundary_conditions.jl",
        "wilson_mpilattice_boundaries.jl",
        "z4_noise.jl",
        "pseudofermion_md_action.jl",
        "mpi_optional.jl",
        "latticematrices_backend.jl",
        "wilson_clover_wrapper.jl",
        "clover_migration.jl",
        "staggered_mpialattice.jl",
        "hisq_mpialattice.jl",
        "runtests_domainwall.jl",
    ]
    for test_file in test_files
        @info "Running isolated core test" test_file
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) $(joinpath(@__DIR__, test_file))`
        process = run(ignorestatus(addenv(
            command, "LDO_TEST_EXPECT_NO_ENZYME" => "true",
            "LDO_EO_BENCH_REPEATS" => get(ENV, "LDO_EO_BENCH_REPEATS", "1"))))
        @test process.exitcode == 0
    end
end
