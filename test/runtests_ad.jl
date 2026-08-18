using Enzyme
using LatticeDiracOperators
using Test

@testset "LatticeDiracOperators Enzyme integration" begin
    @test Base.get_extension(
        LatticeDiracOperators, :LatticeDiracOperatorsEnzymeExt) !== nothing
    project_directory = dirname(Base.active_project())
    test_files = [
        "wilson_lm_callback_ad.jl",
        "hisq_hmc_example.jl",
        joinpath("MPIJACC", "ad.jl"),
    ]
    for test_file in test_files
        @info "Running isolated AD test" test_file
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) $(joinpath(@__DIR__, test_file))`
        process = run(ignorestatus(command))
        @test process.exitcode == 0
    end
end
