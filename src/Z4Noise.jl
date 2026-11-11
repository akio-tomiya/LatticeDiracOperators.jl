const _Z4_STREAM_TAG = UInt32(0x00005a34)
const _Z2_STREAM_TAG = UInt32(0x00005a32)

@inline function _z4_root(::Type{Complex{T}}, value::UInt32) where {T}
    z = zero(T)
    o = one(T)
    value == UInt32(0) && return Complex{T}(o, z)
    value == UInt32(1) && return Complex{T}(z, o)
    value == UInt32(2) && return Complex{T}(-o, z)
    return Complex{T}(z, -o)
end

@inline _z4_root(value::Integer) = _z4_root(ComplexF64, UInt32(value))

@inline _discrete_noise_root(::Val{4}, ::Type{T}, value) where {T} = _z4_root(T, value)
@inline _discrete_noise_root(::Val{2}, ::Type{T}, value) where {T} =
    iszero(value) ? one(T) : -one(T)

@inline function _kernel_discrete_noise!(
    site_index,
    data::AbstractArray{T},
    indexer,
    ::Val{NC1},
    ::Val{NC2},
    ::Val{nw},
    coordinates,
    local_size,
    global_size,
    key,
    algorithm,
    ::Val{FirstFifthSliceOnly},
    order::Val{Order},
) where {T<:Complex,NC1,NC2,nw,FirstFifthSliceOnly,Order}
    local_indices = LatticeMatrices.delinearize(indexer, site_index, 0)
    storage_indices = LatticeMatrices.delinearize(indexer, site_index, nw)
    global_indices = LatticeMatrices.global_site_coordinates(
        local_indices,
        coordinates,
        local_size,
    )
    active_site = !FirstFifthSliceOnly || global_indices[5] == 1
    global_site = LatticeMatrices.global_site_id(global_indices, global_size)
    rng = LatticeMatrices.site_rng(key, global_site, algorithm)

    @inbounds for component in 1:NC2, color in 1:NC1
        if active_site
            rng, value = LatticeMatrices.rand_bounded(rng, UInt32(Order))
            data[color, component, storage_indices...] = _discrete_noise_root(order, T, value)
        else
            data[color, component, storage_indices...] = zero(T)
        end
    end
    return nothing
end

function _discrete_distribution_lattice!(
    lattice::LatticeMatrices.LatticeMatrix{D,T,AT,NC1,NC2,nw,DI},
    order::Val{Order};
    seed=nothing,
    sweep::Integer=0,
    rng_algorithm=LatticeMatrices.Philox4x32(),
    first_fifth_slice_only::Bool=false,
) where {D,T<:Complex,AT,NC1,NC2,nw,DI,Order}
    first_fifth_slice_only && D != 5 && throw(ArgumentError(
        "first_fifth_slice_only requires a five-dimensional field",
    ))
    shared_seed = _shared_fermion_noise_seed(lattice, seed)
    key = LatticeMatrices.RNGStreamKey(
        shared_seed,
        sweep,
        0,
        0,
        Order == 2 ? _Z2_STREAM_TAG : _Z4_STREAM_TAG,
    )
    _mark_halo_dirty!(lattice)
    JACC.parallel_for(
        prod(lattice.PN),
        _kernel_discrete_noise!,
        lattice.A,
        lattice.indexer,
        Val(NC1),
        Val(NC2),
        Val(nw),
        lattice.coords,
        lattice.PN,
        lattice.gsize,
        key,
        rng_algorithm,
        Val(first_fifth_slice_only),
        order,
    )
    LatticeMatrices.set_halo!(lattice)
    return nothing
end

_z4_distribution_lattice!(lattice; kwargs...) =
    _discrete_distribution_lattice!(lattice, Val(4); kwargs...)

"""
    Z2_distribution_fermion!(field::WilsonFermion_4D_MPILattice;
                             seed=nothing, sweep=0, rng_algorithm=Philox4x32())

Fill the global interior with real Z2 noise `{-1, 1}` and refresh halos.
All ranks call together. An explicit seed and sweep give identical fields
across CPU, GPU, and MPI decompositions; otherwise rank zero draws a seed
and broadcasts it.
"""
function Z2_distribution_fermion!(field::WilsonFermion_4D_MPILattice; kwargs...)
    _discrete_distribution_lattice!(field.f, Val(2); kwargs...)
    return field
end

"""
    Z4_distribution_fermi!(field; seed=nothing, sweep=0,
                           rng_algorithm=Philox4x32())

Fill a standard LatticeMatrices-backed fermion field with exact Z4 roots
`{1, im, -1, -im}`.  An explicit `seed` and `sweep` produce the same global
field across CPU/GPU backends and MPI decompositions.  With `seed=nothing`,
rank zero draws and broadcasts a fresh seed.

For domain-wall fields, the physical first fifth-dimensional slice is filled
and the remaining slices are cleared, matching the historical API.
"""
function Z4_distribution_fermi!(
    x::WilsonFermion_4D_MPILattice;
    kwargs...,
)
    _z4_distribution_lattice!(x.f; kwargs...)
    return x
end


function Z4_distribution_fermi!(
    x::StaggeredFermion_4D_MPILattice;
    kwargs...,
)
    _z4_distribution_lattice!(x.f; kwargs...)
    return x
end


function Z4_distribution_fermi!(
    x::DomainwallFermion_5D_MPILattice;
    kwargs...,
)
    _z4_distribution_lattice!(
        x.f;
        first_fifth_slice_only=true,
        kwargs...,
    )
    return x
end
