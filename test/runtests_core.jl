using LatticeDiracOperators
using LatticeMatrices
using Test

@testset "LatticeDiracOperators core without Enzyme" begin
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsEnzymeExt) === nothing
    @test Base.get_extension(
        LatticeMatrices, :LatticeMatricesEnzymeExt) === nothing

    project_directory = dirname(Base.active_project())
    test_files = [
        "public_api.jl",
        "readme_examples.jl",
        "solver_diagnostics.jl",
        "wilson_boundary_conditions.jl",
        "z4_noise.jl",
        "pseudofermion_md_action.jl",
        "latticematrices_backend.jl",
        "wilson_clover_wrapper.jl",
        "staggered_mpialattice.jl",
        "hisq_mpialattice.jl",
        "domainwall_mpialattice.jl",
        "domainwall_grid_reference.jl",
    ]
    for test_file in test_files
        @info "Running isolated core test" test_file
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) $(joinpath(@__DIR__, test_file))`
        process = run(ignorestatus(addenv(
            command, "LDO_TEST_EXPECT_NO_ENZYME" => "true")))
        @test process.exitcode == 0
    end
end
