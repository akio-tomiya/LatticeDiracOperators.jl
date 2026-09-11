module HISQHMC4x4Example

import JACC
JACC.@init_backend

using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Random

const DIM = 4
const NC = 3
const LATTICE_SIZE = (4, 4, 4, 4)
const PROCESS_GRID = (1, 1, 1, 1)

function gauge_action(U, beta)
    action = GaugeAction(U)
    plaquette_loops = make_loops_fromname("plaquette")
    append!(plaquette_loops, plaquette_loops')
    push!(action, beta / 2, plaquette_loops)
    return action
end

function hamiltonian(action, U, momentum, fermion_action_value)
    gauge = -evaluate_GaugeAction(action, U) / U[1].NC
    kinetic = momentum * momentum / 2
    return real(gauge + kinetic + fermion_action_value)
end

function update_links!(U, momentum, step, action)
    temps = get_temporary_gaugefields(action)
    temp1, temp2, exponential, work = temps[1:4]
    for mu in 1:DIM
        exptU!(exponential, step, momentum[mu], [temp1, temp2])
        mul!(work, exponential, U[mu])
        substitute_U!(U[mu], work)
    end
    return U
end

function update_gauge_momentum!(momentum, U, step, action)
    temps = get_temporary_gaugefields(action)
    derivative = temps[end]
    factor = -step / U[1].NC
    for mu in 1:DIM
        calc_dSdUμ!(derivative, action, mu, U)
        mul!(temps[1], U[mu], derivative)
        Traceless_antihermitian_add!(momentum[mu], factor, temps[1])
    end
    return momentum
end

function update_fermion_momentum!(
    momentum, U, step, gauge, fermion, pseudofermion,
)
    derivatives = get_temporary_gaugefields(gauge)[1:DIM]
    calc_UdSfdU!(derivatives, fermion, U, pseudofermion)
    for mu in 1:DIM
        Traceless_antihermitian_add!(momentum[mu], -step, derivatives[mu])
    end
    return momentum
end

function trajectory!(
    U,
    momentum,
    old_U,
    pseudofermion,
    gaussian_fermion,
    gauge,
    fermion;
    mdsteps,
    trajectory_length,
)
    step = trajectory_length / mdsteps

    gauss_distribution!(momentum)
    gauss_sampling_in_action!(gaussian_fermion, U, fermion)
    sample_pseudofermions!(pseudofermion, U, fermion, gaussian_fermion)
    old_fermion_action = evaluate_FermiAction(
        fermion, U, pseudofermion)

    substitute_U!(old_U, U)
    old_hamiltonian = hamiltonian(gauge, U, momentum, old_fermion_action)

    for _ in 1:mdsteps
        update_links!(U, momentum, step / 2, gauge)
        update_gauge_momentum!(momentum, U, step, gauge)
        update_fermion_momentum!(
            momentum, U, step, gauge, fermion, pseudofermion)
        update_links!(U, momentum, step / 2, gauge)
    end

    new_fermion_action = evaluate_FermiAction(fermion, U, pseudofermion)
    new_hamiltonian = hamiltonian(gauge, U, momentum, new_fermion_action)
    delta_h = new_hamiltonian - old_hamiltonian
    acceptance_draw = rand()
    accepted = log(acceptance_draw) < -delta_h
    accepted || substitute_U!(U, old_U)
    return (; accepted, delta_h, old_hamiltonian, new_hamiltonian)
end

function plaquette(U, gauge)
    temps = get_temporary_gaugefields(gauge)
    normalization = 1 / (6 * U[1].NV * U[1].NC)
    return real(calculate_Plaquette(U, temps[1], temps[2]) * normalization)
end

"""
    run_hisq_hmc(; kwargs...)

Run a small, single-process four-flavor HISQ HMC demonstration on a fixed
`4^4` lattice using the analytic `StaggeredFermiAction` force.
"""
function run_hisq_hmc(;
    trajectories=1,
    mdsteps=2,
    trajectory_length=0.02,
    beta=6.0,
    mass=0.1,
    naik_epsilon=0.0,
    eps_CG=1e-10,
    maxsteps=10_000,
    seed=1234,
    verbose=true,
)
    trajectories >= 1 || throw(ArgumentError("trajectories must be positive"))
    mdsteps >= 1 || throw(ArgumentError("mdsteps must be positive"))
    trajectory_length > 0 || throw(ArgumentError("trajectory_length must be positive"))

    communicator = SerialCommunicator()
    Random.seed!(seed)

    # Full HISQ smearing and the three-link Naik term need three halo layers.
    U = Initialize_Gaugefields(
        NC,
        3,
        LATTICE_SIZE...;
        condition="cold",
        isMPILattice=true,
        PEs=PROCESS_GRID,
        comm=communicator,
        verbose_level=0,
    )
    gauge = gauge_action(U, beta)

    fermion_template = GeneralFermion(
        NC,
        1,
        LATTICE_SIZE,
        PROCESS_GRID;
        nw=3,
        phases=(1, 1, 1, -1),
        elementtype=ComplexF64,
        comm0=communicator,
    )
    fermion = StaggeredFermiAction(
        U,
        fermion_template;
        mass,
        Nf=4,
        discretization=:hisq,
        naik_epsilon,
        eps_CG,
        maxsteps,
        verbose_level=0,
    )

    pseudofermion = fermion_template
    gaussian_fermion = similar(fermion_template)
    momentum = initialize_TA_Gaugefields(U)
    old_U = similar(U)
    substitute_U!(old_U, U)

    report = verbose
    report && println(
        "HISQ HMC: lattice=4^4 PEs=$PROCESS_GRID ",
        "beta=$beta mass=$mass mdsteps=$mdsteps",
    )
    report && println("initial plaquette = ", plaquette(U, gauge))

    accepted_count = 0
    last_result = nothing
    for trajectory_number in 1:trajectories
        result = trajectory!(
            U,
            momentum,
            old_U,
            pseudofermion,
            gaussian_fermion,
            gauge,
            fermion;
            mdsteps,
            trajectory_length,
        )
        accepted_count += result.accepted
        last_result = result
        report && println(
            "trajectory $trajectory_number: accepted=$(result.accepted) ",
            "deltaH=$(result.delta_h) plaquette=$(plaquette(U, gauge))",
        )
    end

    final_plaquette = plaquette(U, gauge)
    acceptance_rate = accepted_count / trajectories
    report && println("acceptance rate = $acceptance_rate")
    return (;
        accepted=last_result.accepted,
        delta_h=last_result.delta_h,
        plaquette=final_plaquette,
        acceptance_rate,
    )
end

function main()
    trajectories = parse(Int, get(ENV, "LDO_HISQ_HMC_TRAJECTORIES", "1"))
    mdsteps = parse(Int, get(ENV, "LDO_HISQ_HMC_MDSTEPS", "2"))
    trajectory_length = parse(
        Float64,
        get(ENV, "LDO_HISQ_HMC_TRAJECTORY_LENGTH", "0.02"),
    )
    return run_hisq_hmc(; trajectories, mdsteps, trajectory_length)
end

end # module HISQHMC4x4Example

if abspath(PROGRAM_FILE) == @__FILE__
    HISQHMC4x4Example.main()
end
