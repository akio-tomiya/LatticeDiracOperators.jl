using MPI
using Test

@testset "LatticeDiracOperators MPI integration" begin
    project_directory = dirname(Base.active_project())
    one_rank_files = ["mpi_lifecycle.jl", "mpi_legacy_lifecycle.jl"]
    for test_file in one_rank_files
        test_path = joinpath(@__DIR__, test_file)
        julia_command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) $test_path`
        command = `$(MPI.mpiexec()) -n 1 $julia_command`
        process = run(ignorestatus(command))
        @test process.exitcode == 0
    end

    test_files = [
        joinpath("MPIJACC", "halo_epoch.jl"),
        joinpath("MPIJACC", "ad.jl"),
        joinpath("MPIJACC", "pseudofermion_md_action.jl"),
        "wilson_clover_wrapper.jl",
        "wilson_lm_callback_ad.jl",
        "staggered_mpialattice.jl",
        "hisq_mpialattice.jl",
        "domainwall_mpialattice.jl",
        "domainwall_grid_reference.jl",
    ]
    for test_file in test_files
        @info "Running isolated two-rank test" test_file
        julia_command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) $(joinpath(@__DIR__, test_file))`
        command = `$(MPI.mpiexec()) -n 2 $julia_command`
        process = run(ignorestatus(addenv(
            command,
            "LDO_TEST_EXPECT_NO_ENZYME" => "false",
            "LDO_TEST_MPI" => "true",
        )))
        @test process.exitcode == 0
    end
end
