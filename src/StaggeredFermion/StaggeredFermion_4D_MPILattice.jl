import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_MPILattice, Fields_4D_MPILattice
import Gaugefields.MPILattice:
    add_matrix!, clear_matrix!, get_PEs, set_halo!, substitute!
import LatticeMatrices:
    Adjoint_Lattice, LatticeMatrix, Shifted_Lattice,
    randomize_gaussian_matrix!, release!

@inline function _staggered_boundary_phases_match(requested, actual)
    length(requested) == length(actual) || return false
    return all(isapprox(requested[mu], actual[mu]) for mu in eachindex(actual))
end

abstract type StaggeredFields_4D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,NG
} <: AbstractFermionfields_4D{NC} end

"""
    StaggeredFermion_4D_MPILattice(NC, NX, NY, NZ, NT; ...)

Four-dimensional staggered fermion field backed by
`LatticeMatrices.LatticeMatrix`.  The per-site shape is `NC × 1`; MPI
decomposition, halo width, precision, communicator, and boundary phases are
owned by the wrapped lattice matrix.
"""
struct StaggeredFermion_4D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,TF
} <: StaggeredFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::TF
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    singleprecision::Bool

    function StaggeredFermion_4D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT;
        NDW=1,
        singleprecision=false,
        elementtype=nothing,
        boundarycondition=(1, 1, 1, -1),
        PEs=nothing,
        comm=MPI.COMM_WORLD,
    )
        MPI.Initialized() || MPI.Init()
        NDW >= 0 || throw(ArgumentError("NDW must be non-negative"))

        complex_type = if elementtype === nothing
            singleprecision ? ComplexF32 : ComplexF64
        else
            elementtype
        end
        complex_type in (ComplexF32, ComplexF64) || throw(ArgumentError(
            "staggered MPILattice fields require ComplexF32 or ComplexF64"))
        singleprecision = complex_type === ComplexF32

        nprocs = MPI.Comm_size(comm)
        process_grid = isnothing(PEs) ? (1, 1, 1, nprocs) : Tuple(PEs)
        length(process_grid) == 4 || throw(ArgumentError(
            "PEs must contain four process-grid dimensions"))
        prod(process_grid) == nprocs || throw(ArgumentError(
            "prod(PEs) must equal the communicator size"))

        phases = Tuple(boundarycondition)
        length(phases) == 4 || throw(ArgumentError(
            "boundarycondition must contain four phases"))
        f = LatticeMatrix(
            NC, 1, 4, (NX, NY, NZ, NT), process_grid;
            nw=NDW,
            elementtype=complex_type,
            phases,
            comm0=comm,
        )
        T = eltype(f.A)
        AT = typeof(f.A)
        DI = typeof(f.indexer)
        NG = 1
        return new{NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,typeof(f)}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NG,
            NDW,
            "staggered",
            singleprecision,
        )
    end
end

function Initialize_StaggeredFermion(
    u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW};
    nowing=(NDW == 0),
    boundarycondition=(1, 1, 1, -1),
) where {NC,NX,NY,NZ,NT,T,AT,NDW}
    return StaggeredFermion_4D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT;
        NDW,
        singleprecision=u.singleprecision,
        elementtype=eltype(u.U.A),
        boundarycondition,
        PEs=get_PEs(u.U),
        comm=u.U.comm,
    )
end

Base.size(x::StaggeredFermion_4D_MPILattice) =
    (x.NC, x.NX, x.NY, x.NZ, x.NT, x.NG)
Base.length(x::StaggeredFermion_4D_MPILattice) =
    x.NC * x.NX * x.NY * x.NZ * x.NT

function Base.similar(x::StaggeredFermion_4D_MPILattice)
    return StaggeredFermion_4D_MPILattice(
        x.NC,
        x.NX,
        x.NY,
        x.NZ,
        x.NT;
        NDW=x.NDW,
        singleprecision=x.singleprecision,
        elementtype=eltype(x.f.A),
        boundarycondition=x.f.phases,
        PEs=get_PEs(x.f),
        comm=x.f.comm,
    )
end

get_myrank(x::StaggeredFermion_4D_MPILattice) = MPI.Comm_rank(x.f.comm)
get_nprocs(x::StaggeredFermion_4D_MPILattice) = MPI.Comm_size(x.f.comm)
barrier(x::StaggeredFermion_4D_MPILattice) = MPI.Barrier(x.f.comm)

function gauss_distribution_fermion!(x::StaggeredFermion_4D_MPILattice)
    real_type = typeof(real(zero(eltype(x.f.A))))
    sigma = sqrt(real_type(0.5))
    randomize_gaussian_matrix!(x.f; sigma, seed=rand(UInt64))
    return x
end

gauss_distribution_fermion!(
    x::StaggeredFermion_4D_MPILattice,
    randomfunc,
) = gauss_distribution_fermion!(x)

function clear_fermion!(x::StaggeredFermion_4D_MPILattice; sethalo=false)
    clear_matrix!(x.f)
    sethalo && set_halo!(x.f)
    return x
end

function clear_fermion!(
    x::StaggeredFermion_4D_MPILattice,
    target_even::Bool,
)
    clear_matrix!(x.f, target_even)
    return x
end

struct Shifted_StaggeredFermion_4D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
} <: StaggeredFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Shifted_Lattice{TF,4}
end

function Shifted_StaggeredFermion_4D_MPILattice(
    x::StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,TF
    },
    shift,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,TF}
    shifted = Shifted_Lattice(x.f, shift)
    return Shifted_StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
    }(shifted)
end

struct Adjoint_StaggeredFermion_4D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
} <: StaggeredFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Adjoint_Lattice{TF}
end

struct Adjoint_Shifted_StaggeredFermion_4D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
} <: StaggeredFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Adjoint_Lattice{Shifted_Lattice{TF,4}}
end

function Base.adjoint(
    x::StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,TF
    },
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,DI,TF}
    return Adjoint_StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
    }(x.f')
end

function Base.adjoint(
    x::Shifted_StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
    },
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF}
    return Adjoint_Shifted_StaggeredFermion_4D_MPILattice{
        NC,NX,NY,NZ,NT,T,AT,NDW,NG,TF
    }(x.f')
end

function shift_fermion(
    x::StaggeredFermion_4D_MPILattice,
    direction::Integer;
    boundarycondition=nothing,
)
    abs(direction) in 1:4 || throw(ArgumentError(
        "staggered shift direction must be ±1, ±2, ±3, or ±4"))
    boundarycondition === nothing ||
        _staggered_boundary_phases_match(boundarycondition, x.f.phases) || throw(ArgumentError(
            "fermion boundary phases do not match boundarycondition"))
    shift = ntuple(mu -> mu == abs(direction) ? sign(direction) : 0, 4)
    return shift_fermion(x, shift)
end

shift_fermion(
    x::StaggeredFermion_4D_MPILattice,
    shift::NTuple{4,<:Integer},
) = Shifted_StaggeredFermion_4D_MPILattice(x, shift)

function add_fermion!(
    destination::StaggeredFermion_4D_MPILattice,
    alpha::Number,
    source::StaggeredFields_4D_MPILattice,
)
    add_matrix!(destination.f, source.f, alpha)
    return destination
end

function add_fermion!(
    destination::StaggeredFermion_4D_MPILattice,
    alpha::Number,
    source1::StaggeredFields_4D_MPILattice,
    beta::Number,
    source2::StaggeredFields_4D_MPILattice,
)
    add_matrix!(destination.f, source1.f, alpha)
    add_matrix!(destination.f, source2.f, beta)
    return destination
end

function set_wing_fermion!(x::StaggeredFermion_4D_MPILattice)
    set_halo!(x.f)
    return x
end

function set_wing_fermion!(
    x::StaggeredFermion_4D_MPILattice,
    boundarycondition,
)
    _staggered_boundary_phases_match(boundarycondition, x.f.phases) || throw(ArgumentError(
        "fermion boundary phases do not match boundarycondition"))
    set_halo!(x.f)
    return x
end

LinearAlgebra.dot(
    x::StaggeredFermion_4D_MPILattice,
    y::StaggeredFermion_4D_MPILattice,
) = dot(x.f, y.f)

function substitute_fermion!(
    destination::StaggeredFermion_4D_MPILattice,
    source::StaggeredFields_4D_MPILattice,
)
    substitute!(destination.f, source.f)
    set_halo!(destination.f)
    return destination
end

function LinearAlgebra.axpby!(
    alpha::Number,
    x::StaggeredFields_4D_MPILattice,
    beta::Number,
    y::StaggeredFermion_4D_MPILattice,
)
    axpby!(alpha, x.f, beta, y.f)
    set_halo!(y.f)
    return y
end

function LinearAlgebra.mul!(
    destination::StaggeredFermion_4D_MPILattice,
    alpha::Number,
    source::StaggeredFermion_4D_MPILattice,
)
    mul!(destination.f, alpha, source.f)
    return destination
end

@static if isdefined(LatticeMatrices, :release!)
    release!(x::Shifted_StaggeredFermion_4D_MPILattice) = release!(x.f)
    release!(x::Adjoint_Shifted_StaggeredFermion_4D_MPILattice) = release!(x.f)
    Base.close(x::Shifted_StaggeredFermion_4D_MPILattice) = release!(x)
    Base.close(x::Adjoint_Shifted_StaggeredFermion_4D_MPILattice) = release!(x)
    Base.isopen(x::Shifted_StaggeredFermion_4D_MPILattice) = isopen(x.f)
    Base.isopen(x::Adjoint_Shifted_StaggeredFermion_4D_MPILattice) = isopen(x.f)
end
