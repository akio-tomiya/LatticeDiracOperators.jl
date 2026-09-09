module NHYPStaggeredHMCExample

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using Random

function gauge_action(U, beta)
    action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(action, beta / 2, plaquettes)
    return action
end

"""
    run_nhyp_staggered_hmc(; kwargs...)

Run staggered-fermion HMC with nHYP links in the fermion action. The default
`Nf=4` avoids a rational approximation; set `Nf=1` or `Nf=2` to use LDO's
RHMC implementation. The gauge action is evaluated on the thin links.
"""
function run_nhyp_staggered_hmc(;
    lattice=(2, 2, 2, 2),
    trajectories=1,
    mdsteps=2,
    trajectory_length=0.005,
    beta=5.7,
    mass=0.5,
    Nf=4,
    seed=1234,
)
    length(lattice) == 4 || throw(ArgumentError("lattice must be four-dimensional"))
    trajectories >= 1 || throw(ArgumentError("trajectories must be positive"))
    mdsteps >= 1 || throw(ArgumentError("mdsteps must be positive"))
    trajectory_length > 0 || throw(ArgumentError(
        "trajectory_length must be positive",
    ))
    Nf in (1, 2, 4, 8) || throw(ArgumentError(
        "Nf must be one of 1, 2, 4, or 8",
    ))
    U = gauge_configuration(
        lattice;
        colors=3,
        halo=1,
        start=:hot,
        seed=UInt64(seed),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    gauge = gauge_action(U, beta)
    pseudofermion = Initialize_pseudofermion_fields(U[1], "staggered")
    gaussian = similar(pseudofermion)
    D = Dirac_operator(U, pseudofermion, Dict(
        "Dirac_operator" => "staggered",
        "mass" => mass,
        "eps" => 1e-11,
        "MaxCGstep" => 10_000,
        "verbose_level" => 0,
    ))
    fermion = FermiAction(D, Dict("Nf" => Nf))
    fermion_md = NHYPSmearedFermiAction(
        fermion,
        pseudofermion;
        alpha_outer=0.5,
        alpha_middle=0.5,
        alpha_inner=0.4,
    )
    actions = MDActionSet(gauge=gauge, fermion=fermion_md)
    driver = md_driver(
        U,
        actions;
        steps=mdsteps,
        trajectory_length,
        integrator=QPQ(),
    )
    momentum = gauge_momenta(U)
    metropolis_rng = Xoshiro(seed + 1)
    accepted = 0
    last_result = nothing

    for trajectory in 1:trajectories
        gaussian_momenta!(
            momentum;
            seed=UInt64(seed + 2),
            sweep=trajectory,
        )
        Random.seed!(seed + 3trajectory)
        refresh_nhyp_pseudofermions!(gaussian, U, driver, :fermion)
        old_U = copy_configuration(U)
        result = md_trajectory!(U, momentum, driver)
        accept = log(rand(metropolis_rng)) <
                 min(0, -result.delta_hamiltonian)
        accept || copy_configuration!(U, old_U)
        accepted += accept
        last_result = result
        println(
            "trajectory=$trajectory accepted=$accept ",
            "deltaH=$(result.delta_hamiltonian) ",
            "plaquette=$(measure_plaquette(U))",
        )
    end
    return (;
        U,
        accepted=accepted / trajectories,
        delta_hamiltonian=last_result.delta_hamiltonian,
    )
end

end # module NHYPStaggeredHMCExample

if abspath(PROGRAM_FILE) == @__FILE__
    using .NHYPStaggeredHMCExample
    NHYPStaggeredHMCExample.run_nhyp_staggered_hmc()
end
