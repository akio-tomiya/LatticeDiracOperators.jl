"""
    setvalue_fermion!(field::WilsonFermion_4D_MPILattice, value, color, x, y, z, t, spin)

Set one component using global, one-based coordinates on CPU, GPU, or MPI
fields. Every rank supplies the same coordinates and value; only the owner
writes. Call `set_wing_fermion!(field)` collectively after all component writes
and before applying a stencil. `field[color, x, y, z, t, spin] = value` has the
same write and halo contract.
"""
function setvalue_fermion!(field::WilsonFermion_4D_MPILattice, value,
                          color::Integer, x::Integer, y::Integer, z::Integer,
                          t::Integer, spin::Integer)
    f = field.f
    index = LinearIndices((f.NC1, f.NC2, f.gsize...))[color, spin, x, y, z, t]
    GeneralFermion(f)[index] = value
    return value
end

function Base.setindex!(field::WilsonFermion_4D_MPILattice, value,
                        color::Integer, x::Integer, y::Integer, z::Integer,
                        t::Integer, spin::Integer)
    return setvalue_fermion!(field, value, color, x, y, z, t, spin)
end

"""
    substitute_fermion!(field::WilsonFermion_4D_MPILattice, values::AbstractVector)

Load a replicated global, halo-free vector in the historical Wilson order
`(color, x, y, z, t, spin)` and refresh halos. All ranks supply the same vector
and call together. GPU fields use bulk transfers through JACC.
"""
function substitute_fermion!(field::WilsonFermion_4D_MPILattice, values::AbstractVector)
    length(values) == length(field) || throw(DimensionMismatch(
        "expected $(length(field)) global entries"))
    global_values = reshape(Array(values), size(field))
    lattice_values = permutedims(global_values, (1, 6, 2, 3, 4, 5))
    substitute_fermion!(GeneralFermion(field.f), vec(lattice_values))
    return field
end

"""
    convert_to_normalvector(field::WilsonFermion_4D_MPILattice)

Collect the global interior on every rank in `(color, x, y, z, t, spin)` order,
matching `substitute_fermion!`. This operation is collective on the field's
communicator and uses bulk host transfers for GPU fields.
"""
function convert_to_normalvector(field::WilsonFermion_4D_MPILattice)
    values = gather_and_bcast_matrix(field.f)
    return vec(permutedims(values, (1, 3, 4, 5, 6, 2)))
end
