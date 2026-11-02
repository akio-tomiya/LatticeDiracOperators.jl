const PhysicalDomainwallOperator = Union{
    MobiusDomainwall_Dirac_operator, GeneralizedDomainwall_Dirac_operator}

function _require_domainwall_physical_propagator(D::PhysicalDomainwallOperator)
    D.D5DW isa Union{D5DW_MobiusDomainwall_operator_MPILattice,
                    D5DW_GeneralizedDomainwall_operator_MPILattice} || throw(ArgumentError(
        "physical domain-wall propagators require the LatticeMatrices-backed MPILattice operator"))
    if D.D5DW isa D5DW_GeneralizedDomainwall_operator_MPILattice
        all(!iszero, D.D5DW.as) || throw(ArgumentError(
            "physical propagators require nonzero row coefficients as"))
    end
    return nothing
end

function _domainwall_wilson!(result, raw, source)
    # D4x_5D! adds the hopping term to its destination.
    axpby!(4 + raw.M, source, 0, result)
    LatticeMatrices.D4x_5D!(result.f, raw.D.U, source.f, -1)
    return result
end

function _domainwall_source_coefficients(raw::D5DW_GeneralizedDomainwall_operator_MPILattice)
    return raw.D.a, raw.D.c, all(isone, raw.as) && all(iszero, raw.cs)
end

function _domainwall_source_coefficients(raw::D5DW_MobiusDomainwall_operator_MPILattice)
    # The scalar wrapper uses b=bs+cs, c=bs-cs, unlike the vector API.
    return ntuple(_ -> 1.0, raw.L5), ntuple(_ -> (raw.b-raw.c)/2, raw.L5), raw.b == raw.c
end

function _kernel_domainwall_physical_source!(i, source, wilson, a, c, indexer,
    ::Val{NC}, ::Val{nw}) where {NC,nw}
    indices = delinearize(indexer, i, nw)
    s = indices[5] - nw
    @inbounds for spin in 1:4, color in 1:NC
        source[color, spin, indices...] = a[s] * (
            source[color, spin, indices...] - c[s] * wilson[color, spin, indices...])
    end
    return nothing
end

"""
    import_domainwall_physical_source!(source5, D, source4)

Embed a physical source and apply `A * (I - C * D_W)`. For the generalized
operator, `A=diag(as)` and `C=diag(cs)`; coefficients may vary with the fifth
coordinate. The scalar Mobius wrapper instead uses `cs=(b-c)/2`.
The row factor `A` matches the row scaling of `D.D5DW` and cancels from the
physical solution. This is Grid's Cayley `Dminus` source convention.
"""
function import_domainwall_physical_source!(source5::DomainwallFermion_5D_MPILattice,
    D::PhysicalDomainwallOperator, source4::WilsonFermion_4D_MPILattice)
    _require_domainwall_physical_propagator(D)
    source5.L5 == D.D5DW.L5 || throw(DimensionMismatch("source and operator L5 differ"))
    LatticeMatrices.domainwall_import_physical_source!(source5.f, source4.f)
    a, c, identity_source = _domainwall_source_coefficients(D.D5DW)
    identity_source && return source5
    pool = D.D5DW._temporary_fermi
    wilson, token = get_temp(pool)
    try
        _domainwall_wilson!(wilson, D.D5DW, source5)
        LatticeMatrices.parallel_for_mutating!(source5.f, prod(source5.f.PN),
            _kernel_domainwall_physical_source!, source5.f.A, wilson.f.A,
            a, c, source5.f.indexer, Val(source5.f.NC1), Val(source5.f.nw))
    finally
        unused!(pool, token)
    end
    return source5
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

Solve the raw five-dimensional domain-wall equation for a four-dimensional
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
    D::PhysicalDomainwallOperator,
    source4::WilsonFermion_4D_MPILattice,
)
    _require_domainwall_physical_propagator(D)
    _validate_domainwall_propagator_fields(
        solution5, solution4, source5, source4, D)
    solution5.f.A === source5.f.A && throw(ArgumentError("solution5 and source5 must not alias"))
    import_domainwall_physical_source!(source5, D, source4)
    diagnostics = solve_DinvX!(solution5, D.D5DW, source5)
    LatticeMatrices.domainwall_export_physical_solution!(solution4.f, solution5.f)
    set_wing_fermion!(solution4, collect(D.boundarycondition[1:4]))
    return diagnostics
end

function solve_domainwall_physical_propagator!(
    solution5::DomainwallFermion_5D_MPILattice,
    solution4::WilsonFermion_4D_MPILattice,
    D::PhysicalDomainwallOperator,
    source4::WilsonFermion_4D_MPILattice,
)
    source5 = similar(solution5)
    return solve_domainwall_physical_propagator!(
        solution5, solution4, source5, D, source4)
end

"""
    domainwall_physical_point_propagators(D, template5; source_position=(1,1,1,1))

Compute all twelve color-spin columns of a domain-wall physical point propagator.
The result contains `five_dimensional`, `physical`, and `diagnostics` tuples in
color-major, spin-minor order, plus the one-based global `source_position`.
"""
function domainwall_physical_point_propagators(
    D::PhysicalDomainwallOperator,
    template5::DomainwallFermion_5D_MPILattice;
    source_position::NTuple{4,<:Integer}=(1, 1, 1, 1),
)
    _require_domainwall_physical_propagator(D)
    template5.f.NC1 == 3 || throw(ArgumentError("point propagators require three colors"))
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

export import_domainwall_physical_source!, solve_domainwall_physical_propagator!
export domainwall_physical_point_propagators, domainwall_residual_mass_correlator
