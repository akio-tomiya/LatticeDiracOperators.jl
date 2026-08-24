import JACC
JACC.@init_backend

using Enzyme # Load the optional automatic-differentiation extension.
using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random

# A spinless covariant nearest-neighbour operator in all four directions:
#
#   D psi(x) = m psi(x)
#              - kappa sum_mu [U_mu(x) psi(x+mu)
#                              + U_mu(x-mu)' psi(x-mu)].
const ALL_DIRECTIONS_MASS = 1.0
const ALL_DIRECTIONS_KAPPA = 0.05

function add_nearest_neighbour!(result, link, source, shift_plus, shift_minus, work)
    mul_AshiftB!(work, link, source, shift_plus)
    add_fermion!(result, -ALL_DIRECTIONS_KAPPA, work)

    mul_shiftAshiftB!(work, adjoint(link), source, shift_minus, shift_minus)
    add_fermion!(result, -ALL_DIRECTIONS_KAPPA, work)
    return nothing
end

function apply_D_all_directions!(
    result,
    U1,
    U2,
    U3,
    U4,
    source,
    fermion_temps,
    gauge_temps,
)
    work = fermion_temps[1]

    clear_fermion!(result)
    add_fermion!(result, ALL_DIRECTIONS_MASS, source)

    add_nearest_neighbour!(
        result, U1, source, (1, 0, 0, 0), (-1, 0, 0, 0), work)
    add_nearest_neighbour!(
        result, U2, source, (0, 1, 0, 0), (0, -1, 0, 0), work)
    add_nearest_neighbour!(
        result, U3, source, (0, 0, 1, 0), (0, 0, -1, 0), work)
    add_nearest_neighbour!(
        result, U4, source, (0, 0, 0, 1), (0, 0, 0, -1), work)
    return result
end

function run_all_directions(;
    seed=456,
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

    # This D is Hermitian. A non-Hermitian operator needs a separate
    # apply_Ddag! callback as the fourth positional argument.
    action = GeneralFermionAction(
        gauge,
        source,
        apply_D_all_directions!,
        apply_D_all_directions!;
        numtemp=2,
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
    force_norms = ntuple(
        direction -> real(dot(force[direction].U, force[direction].U)),
        4,
    )
    return (; action, gauge, source, DdagD_source, force, force_norms)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_all_directions()
    println("AD force norms (U1, U2, U3, U4) = ", result.force_norms)
end
