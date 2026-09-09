using Test

include(joinpath(@__DIR__, "..", "examples", "HISQ_HMC_4x4.jl"))

@testset "4^4 HISQ HMC through StaggeredFermiAction" begin
    result = HISQHMC4x4Example.run_hisq_hmc(;
        trajectories=1,
        mdsteps=1,
        trajectory_length=1e-4,
        mass=0.4,
        eps_CG=1e-8,
        maxsteps=2_000,
        seed=1234,
        verbose=false,
    )
    @test result.accepted isa Bool
    @test isfinite(result.delta_h)
    @test isfinite(result.plaquette)
    @test 0 <= result.acceptance_rate <= 1
end
