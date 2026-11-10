# Two-flavor staggered-fermion HMC (RHMC for the rooted determinant) on an SU(3)
# lattice with the v1 API. The original sample used a 16^4 lattice; the default
# here is LATTICE = (8, 8, 8, 8) so that it finishes in minutes on a laptop.
# Run: julia -t 4 --project=. run/staggeredhmc.jl

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

const LATTICE = (8, 8, 8, 8) # original sample: (16, 16, 16, 16)
const NC = 3

function staggered_hmc(; β=5.5, mass=0.1, Nf=2, trajectories=6, mdsteps=10, seed=123)
    U = gauge_configuration(LATTICE; colors=NC, halo=1, start=:cold, process_grid=(1, 1, 1, 1))

    gauge_action = GaugeAction(U)
    plaquettes = make_loops_fromname("plaquette", Dim=4)
    append!(plaquettes, plaquettes')
    push!(gauge_action, β / 2, plaquettes)

    x = Initialize_pseudofermion_fields(U[1], "staggered")
    parameters = Dict{String,Any}(
        "Dirac_operator" => "staggered",
        "mass" => mass,
        "eps_CG" => 1e-16,
        "MaxCGstep" => 3000,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, parameters)
    fermi_action = FermiAction(D, Dict{String,Any}("Nf" => Nf))

    phi = similar(x)
    noise = similar(x)
    fermion_md = PseudofermionMDAction(fermi_action, phi)
    actions = MDActionSet(; gauge=gauge_action, fermion=fermion_md)
    driver = md_driver(U, actions; steps=mdsteps, trajectory_length=1.0, integrator=QPQ())

    p = gauge_momenta(U)
    old_U = copy_configuration(U)
    metropolis_rng = Xoshiro(seed)
    accepted = 0

    println("staggered HMC: lattice=$LATTICE NC=$NC β=$β mass=$mass Nf=$Nf mdsteps=$mdsteps")
    println("0 plaquette = ", measure_plaquette(U))
    # After the cold start, deltaH is large and negative while the plaquette relaxes.
    for trajectory in 1:trajectories
        gaussian_momenta!(p; seed=UInt64(seed + 1), sweep=trajectory)
        refresh_pseudofermion!(fermion_md, U, noise; seed=UInt64(seed + 2), sweep=trajectory, subgroup=1)
        copy_configuration!(old_U, U)

        elapsed = @elapsed result = md_trajectory!(U, p, driver)
        accept = log(rand(metropolis_rng)) < min(0, -result.delta_hamiltonian)
        if accept
            accepted += 1
        else
            copy_configuration!(U, old_U)
        end
        println(trajectory, " deltaH = ", result.delta_hamiltonian, " accepted = ", accept,
            " plaquette = ", measure_plaquette(U), " acceptance = ", accepted / trajectory,
            " time = ", round(elapsed; digits=2), " s")
    end
    return U
end

staggered_hmc()
