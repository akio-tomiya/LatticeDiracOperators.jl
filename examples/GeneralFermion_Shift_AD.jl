module GeneralFermionShiftADExample

import JACC
JACC.@init_backend

using Enzyme # Activates LatticeDiracOperatorsEnzymeExt.
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

export ShiftDefinedWilson, run_shift_defined_ad

"""
    ShiftDefinedWilson{Adjoint}(kappa)

Callable Wilson-like Dirac application assembled directly from shifted lattice
operations.  It deliberately does not construct or wrap a
`WilsonDiracOperator4D`; `GeneralFermionAction` differentiates this callback.
"""
struct ShiftDefinedWilson{Adjoint,T}
    kappa::T
end

ShiftDefinedWilson{Adjoint}(kappa::T) where {Adjoint,T} =
    ShiftDefinedWilson{Adjoint,T}(kappa)

function _add_shifted_hop!(
    result,
    link,
    source,
    gamma,
    shift_forward,
    shift_backward,
    temporary1,
    temporary2,
    kappa,
    spin_sign,
)
    # U_mu(x) (1 - spin_sign * gamma_mu) psi(x + mu)
    mul_AshiftB!(temporary1, link, source, shift_forward)
    mul!(temporary2, temporary1, transpose(I(4) - spin_sign * gamma))
    add_fermion!(result, -kappa, temporary2)

    # U_mu(x-mu)' (1 + spin_sign * gamma_mu) psi(x - mu)
    mul_shiftAshiftB!(
        temporary1,
        adjoint(link),
        source,
        shift_backward,
        shift_backward,
    )
    mul!(temporary2, temporary1, transpose(I(4) + spin_sign * gamma))
    add_fermion!(result, -kappa, temporary2)
    return nothing
end

function (apply::ShiftDefinedWilson{Adjoint})(
    result,
    U1,
    U2,
    U3,
    U4,
    source,
    fermion_temps,
    gauge_temps,
) where {Adjoint}
    clear_fermion!(result)
    add_fermion!(result, 1, source)

    temporary1 = fermion_temps[1]
    temporary2 = fermion_temps[2]
    spin_sign = Adjoint ? -1 : 1

    _add_shifted_hop!(
        result, U1, source, γ1,
        (1, 0, 0, 0), (-1, 0, 0, 0),
        temporary1, temporary2, apply.kappa, spin_sign,
    )
    _add_shifted_hop!(
        result, U2, source, γ2,
        (0, 1, 0, 0), (0, -1, 0, 0),
        temporary1, temporary2, apply.kappa, spin_sign,
    )
    _add_shifted_hop!(
        result, U3, source, γ3,
        (0, 0, 1, 0), (0, 0, -1, 0),
        temporary1, temporary2, apply.kappa, spin_sign,
    )
    _add_shifted_hop!(
        result, U4, source, γ4,
        (0, 0, 0, 1), (0, 0, 0, -1),
        temporary1, temporary2, apply.kappa, spin_sign,
    )
    return result
end

"""
    run_shift_defined_ad(; ...)

Build a `GeneralFermionAction` from the shift-defined callbacks, evaluate
`D' * D` and the pseudofermion action, and calculate its gauge force with
Enzyme.  The returned `force_norm` is nonzero when the callback has been
differentiated successfully.

Pass an MPI communicator and matching `process_grid` to run distributed. The
default is serial and does not require MPI.jl.
"""
function run_shift_defined_ad(;
    local_x=4,
    transverse_size=4,
    number_of_colors=3,
    kappa=0.1,
    seed=123,
    eps_CG=1e-10,
    maxsteps=10_000,
    process_grid=(1, 1, 1, 1),
    comm=nothing,
)
    global_size = (
        local_x * process_grid[1],
        transverse_size * process_grid[2],
        transverse_size * process_grid[3],
        transverse_size * process_grid[4],
    )

    gauge = gauge_configuration(
        global_size;
        colors=number_of_colors,
        halo=1,
        start=:cold,
        process_grid,
        comm,
        verbose=0,
    )

    Random.seed!(seed)
    source = GeneralFermion(
        number_of_colors,
        4,
        global_size,
        process_grid;
        nw=1,
        elementtype=ComplexF64,
        numtemps=8,
        comm0=comm,
    )
    gauss_distribution_fermion!(source)
    set_wing_fermion!(source)

    action = GeneralFermionAction(
        gauge,
        source,
        ShiftDefinedWilson{false}(kappa),
        ShiftDefinedWilson{true}(kappa);
        # Two callback work fields plus one output field used by the AD force.
        numtemp=3,
        num=6,
        numg=4,
        numcg=10,
        eps_CG,
        maxsteps,
        verbose_level=0,
    )

    ddagd_source = similar(source)
    mul!(ddagd_source, action.DdagD, source)
    potential = evaluate_FermiAction(action, gauge, source)

    force = similar(gauge)
    calc_UdSfdU!(force, action, gauge, source)
    force_norm = sum(link -> real(dot(link.U, link.U)), force)

    return (;
        action,
        gauge,
        source,
        ddagd_source,
        potential,
        force,
        force_norm,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_shift_defined_ad()
    println("pseudofermion action = ", result.potential)
    println("AD force norm       = ", result.force_norm)
end

end # module GeneralFermionShiftADExample
