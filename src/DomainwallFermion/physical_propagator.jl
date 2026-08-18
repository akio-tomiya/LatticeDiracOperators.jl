function _require_shamir_physical_propagator(D::MobiusDomainwall_Dirac_operator)
    D.D5DW isa D5DW_MobiusDomainwall_operator_MPILattice || throw(ArgumentError(
        "physical domain-wall propagators require the LatticeMatrices-backed MPILattice operator"))
    D.b == 1 && D.c == 1 || throw(ArgumentError(
        "this physical-source API currently supports Shamir domain-wall fermions " *
        "(b=1, c=1); got b=$(D.b), c=$(D.c)"))
    return nothing
end

function _validate_domainwall_propagator_fields(solution5, solution4, source5, source4, D)
    for (name, field) in
        (("solution5", solution5), ("source5", source5))
        field isa DomainwallFermion_5D_MPILattice || throw(ArgumentError(
            "$name must be a DomainwallFermion_5D_MPILattice"))
        field.L5 == D.D5DW.L5 || throw(DimensionMismatch(
            "$name has L5=$(field.L5), but the operator has L5=$(D.D5DW.L5)"))
    end
    solution4 isa WilsonFermion_4D_MPILattice || throw(ArgumentError(
        "solution4 must be a WilsonFermion_4D_MPILattice container"))
    source4 isa WilsonFermion_4D_MPILattice || throw(ArgumentError(
        "source4 must be a WilsonFermion_4D_MPILattice container"))
    return nothing
end

"""
    solve_domainwall_physical_propagator!(solution5, solution4, source5, D, source4)

Solve the raw five-dimensional Shamir equation for a four-dimensional
physical source. `source5` is reusable workspace. The source embedding and
solution projection follow Grid's `ImportPhysicalFermionSource` and
`ExportPhysicalFermionSolution` conventions. The Pauli--Villars-composed outer
operator is deliberately not used for this valence propagator solve.

Returns the `SolverDiagnostics` from the raw five-dimensional solve.
"""
function solve_domainwall_physical_propagator!(
    solution5::DomainwallFermion_5D_MPILattice,
    solution4::WilsonFermion_4D_MPILattice,
    source5::DomainwallFermion_5D_MPILattice,
    D::MobiusDomainwall_Dirac_operator,
    source4::WilsonFermion_4D_MPILattice,
)
    _require_shamir_physical_propagator(D)
    _validate_domainwall_propagator_fields(
        solution5, solution4, source5, source4, D)
    LatticeMatrices.domainwall_import_physical_source!(source5.f, source4.f)
    diagnostics = solve_DinvX!(solution5, D.D5DW, source5)
    LatticeMatrices.domainwall_export_physical_solution!(solution4.f, solution5.f)
    set_wing_fermion!(solution4, collect(D.boundarycondition[1:4]))
    return diagnostics
end

function solve_domainwall_physical_propagator!(
    solution5::DomainwallFermion_5D_MPILattice,
    solution4::WilsonFermion_4D_MPILattice,
    D::MobiusDomainwall_Dirac_operator,
    source4::WilsonFermion_4D_MPILattice,
)
    source5 = similar(solution5)
    return solve_domainwall_physical_propagator!(
        solution5, solution4, source5, D, source4)
end

"""
    domainwall_physical_point_propagators(D, template5; source_position=(1,1,1,1))

Compute all twelve color-spin columns of a Shamir physical point propagator.
The result contains `five_dimensional`, `physical`, and `diagnostics` tuples in
color-major, spin-minor order, plus the one-based global `source_position`.
"""
function domainwall_physical_point_propagators(
    D::MobiusDomainwall_Dirac_operator,
    template5::DomainwallFermion_5D_MPILattice;
    source_position::NTuple{4,<:Integer}=(1, 1, 1, 1),
)
    _require_shamir_physical_propagator(D)
    template5.L5 == D.D5DW.L5 || throw(DimensionMismatch(
        "template field L5=$(template5.L5) differs from operator L5=$(D.D5DW.L5)"))
    source4 = Initialize_WilsonFermion(
        D.U[1]; nowing=true, boundarycondition=collect(D.boundarycondition[1:4]))
    source5 = similar(template5)
    five_dimensional = Vector{typeof(template5)}(undef, 12)
    physical_template = similar(source4)
    physical = Vector{typeof(physical_template)}(undef, 12)
    diagnostics = Vector{SolverDiagnostics}(undef, 12)

    column = 0
    for source_color in 1:3, source_spin in 1:4
        column += 1
        clear_fermion!(source4)
        LatticeMatrices.set_global_component!(
            source4.f, one(eltype(source4.f.A)), source_color, source_spin,
            source_position)
        five_dimensional[column] = similar(template5)
        physical[column] = similar(physical_template)
        diagnostics[column] = solve_domainwall_physical_propagator!(
            five_dimensional[column], physical[column], source5, D, source4)
    end
    return (
        five_dimensional=Tuple(five_dimensional),
        physical=Tuple(physical),
        diagnostics=Tuple(diagnostics),
        source_position=ntuple(d -> Int(source_position[d]), 4),
    )
end

"""
    domainwall_residual_mass_correlator(propagators5; kwargs...)

Form the zero-momentum pseudoscalar correlator `PP`, the midpoint correlator
`J5qP`, and their timeslice ratio from twelve five-dimensional point-source
propagators. This is the Furman--Shamir/UKQCD residual-mass ratio used by
Grid's `ContractJ5q` convention; a plateau fit is intentionally left to the
measurement layer.
"""
function domainwall_residual_mass_correlator(
    propagators5::NTuple{12,T};
    axis::Integer=4,
    origin::NTuple{4,<:Integer}=(1, 1, 1, 1),
    momentum::NTuple{4,<:Integer}=(0, 0, 0, 0),
) where {T<:DomainwallFermion_5D_MPILattice}
    element_type = eltype(propagators5[1].f.A)
    identity_spin = Matrix{element_type}(I, 4, 4)
    extent = propagators5[1].f.gsize[axis]
    PP = zeros(element_type, extent)
    J5qP = zeros(element_type, extent)
    for source_color in 1:3
        block = ntuple(
            source_spin -> propagators5[(source_color - 1) * 4 + source_spin].f,
            4)
        PP .+= LatticeMatrices.domainwall_projected_bilinear_slices(
            block, block, identity_spin, identity_spin;
            projection=:physical, axis, origin, momentum, coefficient=1)
        J5qP .+= LatticeMatrices.domainwall_projected_bilinear_slices(
            block, block, identity_spin, identity_spin;
            projection=:midpoint, axis, origin, momentum, coefficient=1)
    end
    return (PP=PP, J5qP=J5qP, ratio=J5qP ./ PP)
end

export solve_domainwall_physical_propagator!
export domainwall_physical_point_propagators, domainwall_residual_mass_correlator
