import JACC
JACC.@init_backend

using Enzyme # Load the optional automatic-differentiation extension.
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

# A small Hermitian nearest-neighbour operator,
#
#   D psi(x) = m psi(x)
#              - kappa U_1(x) psi(x+1)
#              - kappa U_1(x-1)' psi(x-1).
#
# This is only an example: users can replace this body with their own shifts.
const QUICKSTART_MASS = 1.0
const QUICKSTART_KAPPA = 0.1
const QUICKSTART_SHIFT_PLUS = (1, 0, 0, 0)
const QUICKSTART_SHIFT_MINUS = (-1, 0, 0, 0)

function apply_D!(result, U1, U2, U3, U4, source, fermion_temps, gauge_temps)
    work = fermion_temps[1]

    clear_fermion!(result)
    add_fermion!(result, QUICKSTART_MASS, source)

    mul_AshiftB!(work, U1, source, QUICKSTART_SHIFT_PLUS)
    add_fermion!(result, -QUICKSTART_KAPPA, work)

    mul_shiftAshiftB!(
        work,
        adjoint(U1),
        source,
        QUICKSTART_SHIFT_MINUS,
        QUICKSTART_SHIFT_MINUS,
    )
    add_fermion!(result, -QUICKSTART_KAPPA, work)
    return result
end

function run_quickstart(;
    seed=123,
    process_grid=(1, 1, 1, 1),
    comm=nothing,
)
    global_size = ntuple(direction -> 2 * process_grid[direction], 4)

    gauge = gauge_configuration(
        global_size;
        colors=2,
        halo=1,
        start=:cold,
        process_grid,
        comm,
        verbose=0,
    )

    Random.seed!(seed)
    source = GeneralFermion(
        2,
        1,
        global_size,
        process_grid;
        nw=1,
        numtemps=4,
        comm0=comm,
    )
    gauss_distribution_fermion!(source)
    set_wing_fermion!(source)

    # This toy D is Hermitian, so the same callback is also apply_Ddag!.
    # For a non-Hermitian D, pass a separately defined adjoint callback here.
    action = GeneralFermionAction(
        gauge,
        source,
        apply_D!,
        apply_D!;
        numtemp=2, # work is [1]; [2] is reserved by the AD force calculation.
        num=4,
        numg=4,
        numcg=10,
        eps_CG=1e-12,
        verbose_level=0,
    )

    DdagD_source = similar(source)
    mul!(DdagD_source, action.DdagD, source)

    force = similar(gauge)
    calc_UdSfdU!(force, action, gauge, source)

    force_norm = sum(link -> real(dot(link.U, link.U)), force)
    return (; action, gauge, source, DdagD_source, force, force_norm)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_quickstart()
    println("AD force norm = ", result.force_norm)
end
