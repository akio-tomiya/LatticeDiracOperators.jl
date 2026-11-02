function _kernel_domainwall_ratio_source!(i, output, psi, weights, denominator,
    ratio, indexer, coords, PN, extent, origin, L5, ::Val{NC}, ::Val{nw}) where {NC,nw}
    indices = delinearize(indexer, i, nw)
    s = indices[5] - nw
    t = coords[4] * PN[4] + indices[4] - nw
    separation = mod(t - origin, extent) + 1
    @inbounds for spin in 1:4, color in 1:NC
        physical_s = ifelse(spin <= 2, L5, 1)
        midpoint_s = ifelse(spin <= 2, L5 ÷ 2, L5 ÷ 2 + 1)
        factor = weights[separation] * ((s == midpoint_s) - ratio * (s == physical_s)) / denominator
        output[color, spin, indices...] = factor * psi[color, spin, indices...]
    end
    return nothing
end

function _kernel_domainwall_slice_inner_product(i, left, right, indexer, s,
    ::Val{NC}, ::Val{nw}) where {NC,nw}
    position = delinearize(indexer, i, nw)
    indices = (position..., s + nw)
    value = zero(real(zero(eltype(left))))
    @inbounds for spin in 1:4, color in 1:NC
        value += real(conj(left[color, spin, indices...]) * right[color, spin, indices...])
    end
    return value
end

"""
    domainwall_residual_mass_gradient(D, template5; timeslices, source_position=(1,1,1,1))

Compute the zero-momentum ratio `sum(J5qP[timeslices])/sum(PP[timeslices])`
and its derivatives `db`, `dc` with respect to every real generalized coefficient.
`timeslices` must be explicitly specified: index 1 is zero separation from
`source_position`. This is a ratio of sums, not the mean of timeslice ratios.
The returned `correlators` also contains the unaveraged PP, J5qP and ratio.

The gradient differentiates both the raw operator and the physical source
`A*(I-C*D_W)*eta`. Twelve point-source solves and twelve adjoint solves are
performed; no finite differencing or differentiation through CG iterations is
used. This exact point-source observable is different from the legacy noisy
`N/D - mass` objective. It does not update coefficients or optimizer state.
"""
function domainwall_residual_mass_gradient(
    D::GeneralizedDomainwall_Dirac_operator,
    template5::DomainwallFermion_5D_MPILattice;
    timeslices,
    source_position::NTuple{4,<:Integer}=(1, 1, 1, 1),
)
    _require_domainwall_physical_propagator(D)
    iseven(template5.L5) || throw(ArgumentError("midpoint mres requires even L5"))
    extent = template5.f.gsize[4]
    slices = collect(timeslices)
    !isempty(slices) && all(t -> t isa Integer && 1 <= t <= extent, slices) ||
        throw(ArgumentError("timeslices must be nonempty integer indices in 1:$extent"))
    length(unique(slices)) == length(slices) || throw(ArgumentError("timeslices contains duplicates"))
    propagators = domainwall_physical_point_propagators(D, template5; source_position)
    correlators = domainwall_residual_mass_correlator(
        propagators.five_dimensional; origin=source_position)
    denominator = real(sum(correlators.PP[slices]))
    numerator = real(sum(correlators.J5qP[slices]))
    isfinite(denominator) && denominator > 0 && isfinite(numerator) ||
        throw(ArgumentError("mres ratio requires finite correlators and a positive PP sum"))
    ratio = numerator / denominator
    weights_host = zeros(typeof(denominator), extent)
    weights_host[slices] .= 1
    weights = JACC.array(weights_host)
    raw = D.D5DW
    L5 = raw.L5
    local_gradient = zeros(typeof(denominator), L5, 2)
    rhs, lambda, embedded, hopped, wilson = (similar(template5) for _ in 1:5)
    source4 = similar(propagators.physical[1])
    indexer4 = LatticeMatrices.DIndexer(ntuple(d -> template5.f.PN[d], 4))
    volume4 = prod(template5.f.PN[1:4])
    diagnostics = SolverDiagnostics[]
    for source_color in 1:3, source_spin in 1:4
        psi = propagators.five_dimensional[4 * (source_color-1) + source_spin]
        LatticeMatrices.parallel_for_mutating!(rhs.f, prod(rhs.f.PN),
            _kernel_domainwall_ratio_source!, rhs.f.A, psi.f.A, weights,
            denominator, ratio, rhs.f.indexer, rhs.f.coords, rhs.f.PN,
            extent, source_position[4], L5, Val(rhs.f.NC1), Val(rhs.f.nw))
        clear_fermion!(lambda)
        push!(diagnostics, solve_DinvX!(lambda, adjoint(raw), rhs))

        # dM/db_s = a_s * D_W on slice s.
        _domainwall_wilson!(wilson, raw, psi)
        for s in 1:L5
            local_gradient[s, 1] -= 2raw.as[s] * JACC.parallel_reduce(
                volume4, _kernel_domainwall_slice_inner_product,
                lambda.f.A, wilson.f.A, indexer4, s, Val(rhs.f.NC1), Val(rhs.f.nw);
                init=zero(denominator), op=+)
        end

        # d(eta5-M*psi)/dc_s = -a_s*D_W*(embedded_eta+F_m*psi)_s.
        clear_fermion!(source4)
        LatticeMatrices.set_global_component!(source4.f, one(eltype(source4.f.A)),
            source_color, source_spin, source_position)
        LatticeMatrices.domainwall_import_physical_source!(embedded.f, source4.f)
        clear_fermion!(hopped)
        set_wing_fermion!(psi)
        LatticeMatrices.apply_F_5D!(hopped.f, raw.mass, L5, psi.f)
        axpby!(1, embedded, 1, hopped)
        _domainwall_wilson!(wilson, raw, hopped)
        for s in 1:L5
            local_gradient[s, 2] -= 2raw.as[s] * JACC.parallel_reduce(
                volume4, _kernel_domainwall_slice_inner_product,
                lambda.f.A, wilson.f.A, indexer4, s, Val(rhs.f.NC1), Val(rhs.f.nw);
                init=zero(denominator), op=+)
        end
    end
    gradient = LatticeMatrices._allreduce_sum(local_gradient, template5.f.comm)
    return (value=ratio, db=gradient[:, 1], dc=gradient[:, 2], correlators=correlators,
        timeslices=Int.(slices), source_position=propagators.source_position,
        diagnostics=propagators.diagnostics, adjoint_diagnostics=Tuple(diagnostics))
end

export domainwall_residual_mass_gradient
