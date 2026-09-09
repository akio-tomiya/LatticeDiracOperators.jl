module LatticeStaggeredActionCompare

import JACC
JACC.@init_backend

using Enzyme
using Gaugefields
using LatticeDiracOperators
using LatticeMatrices
using LinearAlgebra
using Random

struct ApplyOneLinkStaggered{Adjoint,T}
    mass::T
end

function (apply::ApplyOneLinkStaggered{Adjoint})(
    result, U1, U2, U3, U4, input, _fermion_temporaries,
    _gauge_temporaries,
) where Adjoint
    operator = StaggeredDiracOperator4D(
        [U1.U, U2.U, U3.U, U4.U], apply.mass)
    mul!(result.field, Adjoint ? adjoint(operator) : operator, input.field)
    return result
end

struct ApplyHISQStaggered{Adjoint,C}
    cache::C
end

function (apply::ApplyHISQStaggered{false})(
    result, U1, U2, U3, U4, input, _fermion_temporaries,
    _gauge_temporaries,
)
    mul_cached_hisq!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, input.field)
    return result
end

function (apply::ApplyHISQStaggered{true})(
    result, U1, U2, U3, U4, input, _fermion_temporaries,
    _gauge_temporaries,
)
    mul_cached_hisq_adjoint!(
        result.field, apply.cache,
        U1.U, U2.U, U3.U, U4.U, input.field)
    return result
end

function _general_action(U, fermion, discretization, mass, naik_epsilon)
    if discretization == :staggered
        apply_D = ApplyOneLinkStaggered{false,typeof(mass)}(mass)
        apply_Ddag = ApplyOneLinkStaggered{true,typeof(mass)}(mass)
    elseif discretization == :hisq
        cache = HISQDiracCache4D(
            getproperty.(U, :U), mass; naik_epsilon)
        apply_D = ApplyHISQStaggered{false,typeof(cache)}(cache)
        apply_Ddag = ApplyHISQStaggered{true,typeof(cache)}(cache)
    else
        throw(ArgumentError(
            "discretization must be :staggered or :hisq"))
    end
    return GeneralFermionAction(
        U, fermion, apply_D, apply_Ddag;
        numtemp=8,
        num=12,
        numg=18,
        numcg=12,
        eps_CG=1e-11,
        maxsteps=10_000,
        verbose_level=0,
    )
end

function _maximum_force_difference(left, right)
    return maximum(
        maximum(abs, Array(left[direction].U.A) .-
                     Array(right[direction].U.A))
        for direction in eachindex(left)
    )
end

function _maximum_force_magnitude(force)
    return maximum(
        maximum(abs, Array(field.U.A)) for field in force)
end

function _force_directional_derivative(force, U, direction)
    gradient = similar(U[1])
    derivative = 0.0
    for mu in eachindex(U)
        mul!(gradient, force[mu]', U[mu])
        derivative += real(dot(gradient.U, direction[mu].U))
    end
    return -2derivative
end

function _timed_force!(force, action, U, pseudofermion)
    elapsed = @elapsed begin
        calc_UdSfdU!(force, action, U, pseudofermion)
        JACC.synchronize()
    end
    return 1_000elapsed
end

function _median(values)
    ordered = sort(values)
    count = length(ordered)
    midpoint = fld(count, 2) + 1
    return isodd(count) ? ordered[midpoint] :
           (ordered[midpoint - 1] + ordered[midpoint]) / 2
end

"""
    run_comparison(; discretization=:staggered, lattice=nothing, repeats=5)

Compare the dedicated analytic `StaggeredFermiAction` with the equivalent
`GeneralFermionAction`/Enzyme path on the active JACC backend. `Nf=8` is used
so that both actions represent the same full-lattice pseudofermion action.
The first Enzyme reverse compilation is excluded from the reported samples
and can take several minutes, especially for HISQ.
"""
function run_comparison(;
    discretization=:staggered,
    lattice=nothing,
    repeats=5,
    mass=0.5,
    naik_epsilon=-0.083,
)
    repeats >= 1 || throw(ArgumentError("repeats must be positive"))
    nw = discretization == :hisq ? 3 : 1
    default_extent = discretization == :hisq ? 4 : 8
    selected_lattice = isnothing(lattice) ?
                       ntuple(_ -> default_extent, 4) : Tuple(lattice)
    U = gauge_configuration(
        selected_lattice;
        colors=3,
        halo=nw,
        start=:hot,
        seed=UInt64(0x434f4d50 + nw),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    fermion = GeneralFermion(
        3, 1, selected_lattice, (1, 1, 1, 1);
        nw,
        phases=(1, 1, 1, -1),
    )
    pseudofermion = similar(fermion)
    Random.seed!(0x53544147 + nw)
    gauss_distribution_fermion!(pseudofermion)
    set_wing_fermion!(pseudofermion)

    dedicated = StaggeredFermiAction(
        U, fermion;
        mass,
        Nf=8,
        discretization,
        naik_epsilon,
        eps_CG=1e-11,
        maxsteps=10_000,
        verbose_level=0,
    )
    general = _general_action(
        U, fermion, discretization, mass, naik_epsilon)

    dedicated_potential = evaluate_FermiAction(
        dedicated, U, pseudofermion)
    general_potential = evaluate_FermiAction(
        general, U, pseudofermion)
    dedicated_force = similar(U)
    general_force = similar(U)
    calc_UdSfdU!(dedicated_force, dedicated, U, pseudofermion)
    calc_UdSfdU!(general_force, general, U, pseudofermion)
    JACC.synchronize()

    force_difference = _maximum_force_difference(
        dedicated_force, general_force)
    force_scale = max(
        _maximum_force_magnitude(dedicated_force), eps(Float64))
    direction = gauge_configuration(
        selected_lattice;
        colors=3,
        halo=nw,
        start=:hot,
        seed=UInt64(0x44495200 + nw),
        process_grid=(1, 1, 1, 1),
        verbose=0,
    )
    finite_difference_step = 3e-4
    plus_U = copy_configuration(U)
    minus_U = copy_configuration(U)
    for mu in eachindex(U)
        add_U!(plus_U[mu], finite_difference_step, direction[mu])
        add_U!(minus_U[mu], -finite_difference_step, direction[mu])
    end
    finite_difference = (
        evaluate_FermiAction(general, plus_U, pseudofermion) -
        evaluate_FermiAction(general, minus_U, pseudofermion)
    ) / (2finite_difference_step)
    dedicated_directional = _force_directional_derivative(
        dedicated_force, U, direction)
    general_directional = _force_directional_derivative(
        general_force, U, direction)
    dedicated_times = [
        _timed_force!(dedicated_force, dedicated, U, pseudofermion)
        for _ in 1:repeats
    ]
    general_times = [
        _timed_force!(general_force, general, U, pseudofermion)
        for _ in 1:repeats
    ]
    dedicated_median_ms = _median(dedicated_times)
    general_median_ms = _median(general_times)
    result = (;
        discretization,
        lattice=selected_lattice,
        potential_difference=abs(dedicated_potential - general_potential),
        force_maximum_difference=force_difference,
        force_relative_difference=force_difference / force_scale,
        finite_difference,
        dedicated_directional,
        general_directional,
        dedicated_median_ms,
        general_median_ms,
        speedup=general_median_ms / dedicated_median_ms,
    )
    println(result)
    return result
end

function main()
    discretization = isempty(ARGS) ? :staggered : Symbol(ARGS[1])
    repeats = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 5
    return run_comparison(; discretization, repeats)
end

end # module LatticeStaggeredActionCompare

if abspath(PROGRAM_FILE) == @__FILE__
    LatticeStaggeredActionCompare.main()
end
