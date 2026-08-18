# Conversion helpers between the deprecated w[s] storage and the v1 common
# five-dimensional LatticeMatrix field.

function substitute_fermion!(
    destination::DomainwallFermion_5D_MPILattice,
    source::MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion<:WilsonFermion_4D_MPILattice}
    destination.L5 == length(source.w) || throw(DimensionMismatch(
        "source and destination must have the same L5"))
    nw = destination.f.nw
    _mark_halo_dirty!(destination.f)
    for s in 1:destination.L5
        destination.f.A[:, :, :, :, :, :, s + nw] .= source.w[s].f.A
    end
    set_halo!(destination.f)
    return destination
end

function substitute_fermion!(
    destination::DomainwallFermion_5D_MPILattice,
    source::MobiusDomainwallFermion_5D{NC,WilsonFermion},
) where {NC,WilsonFermion}
    destination.L5 == length(source.w) || throw(DimensionMismatch(
        "source and destination must have the same L5"))
    _, NX, NY, NZ, NT, NG = size(source.w[1].f)
    work = zeros(
        eltype(source.w[1].f), NC, NG, NX, NY, NZ, NT, destination.L5)
    for s in 1:destination.L5
        work[:, :, :, :, :, :, s] .=
            permutedims(source.w[s].f, (1, 6, 2, 3, 4, 5))
    end
    temporary = LatticeMatrix(
        work, 5, destination.f.dims;
        nw=destination.f.nw,
        phases=destination.f.phases,
        comm0=destination.f.comm,
    )
    substitute!(destination.f, temporary)
    set_halo!(destination.f)
    return destination
end
