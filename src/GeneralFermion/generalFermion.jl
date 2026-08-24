using LatticeMatrices
using LinearAlgebra
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice, Fields_4D_MPILattice
using PreallocatedArrays


function dSFdU! end
export dSFdU!


function defaultphase(dim)
    phase = ones(dim)
    phase[end] = -1
    return phase
end

"""
    GeneralFermion(field::LatticeMatrix)
    GeneralFermion(NC, NG, global_size, process_grid; kwargs...)

Callback-oriented fermion field backed directly by a
`LatticeMatrices.LatticeMatrix`. Use it with [`GeneralFermionAction`](@ref)
when a user-defined `apply_D!`/`apply_Ddag!` pair is more convenient than one
of LDO's standard formulations.
"""
struct GeneralFermion{TF,D,T,AT,NC,NG,nw,DI} <: AbstractFermionfields{NC,D}
    field::TF

    function GeneralFermion(field::L) where {D,T,AT,NC,NG,nw,DI,L<:LatticeMatrix{D,T,AT,NC,NG,nw,DI}}
        return new{L,D,T,AT,NC,NG,nw,DI}(field)
    end

    function GeneralFermion(NC, NG, gsize, PEs; nw=1, elementtype=ComplexF64, phases=defaultphase(length(gsize)),
        comm0=MPI.COMM_WORLD, numtemps=1)
        dim = length(gsize)
        field = LatticeMatrix(NC, NG, dim, gsize, PEs; nw, elementtype, phases,
            comm0, numtemps)
        return GeneralFermion(field)
    end
end
export GeneralFermion

@inline function LinearAlgebra.dot(
    A::L1,
    B::L2,
) where {L1<:GeneralFermion,L2<:GeneralFermion}

    s = dot(A.field, B.field)
    return s
end

@inline function LinearAlgebra.mul!(
    c::L1,
    a::TG,
    b::L2,
) where {TG<:Fields_4D_MPILattice,L1<:GeneralFermion,L2<:GeneralFermion}
    mul!(c.field, a.U, b.field)
end

function Base.zero(a::L1) where {L1<:GeneralFermion}
    bf = similar(a.field)
    b = GeneralFermion(bf)
    return b
end

function Base.similar(a::L1) where {L1<:GeneralFermion}
    return zero(a)
end

"""
    mul!(result::GeneralFermion, operator, input::GeneralFermion)

Apply a `LatticeMatrices` operator directly to `GeneralFermion` fields.
The underlying `LatticeMatrix` storage is passed to the operator, while the
`GeneralFermion` wrapper is retained for use by LatticeDiracOperators actions.
"""
function LinearAlgebra.mul!(
    result::GeneralFermion,
    operator::LatticeMatrices.OperatorOnKernel,
    input::GeneralFermion,
)
    mul!(result.field, operator, input.field)
    return result
end

function substitute_fermion!(
    A::L1,
    B::L2,
) where {L1<:GeneralFermion,L2<:GeneralFermion}
    substitute!(A.field, B.field)
    set_halo!(A.field)
end


function substitute_fermion!(A::L1, B::AbstractFermionfields_4D) where {L1<:GeneralFermion}
    dim = 4
    PEs = A.field.dims
    phases = A.field.phases
    nw = A.field.nw
    comm0 = A.field.comm
    D = permutedims(B.f, (1, 6, 2, 3, 4, 5))

    tempf = LatticeMatrix(D, dim, PEs;
        nw,
        phases,
        comm0)
    substitute!(A.field, tempf)
    set_halo!(A.field)

end

function substitute_fermion!(A::L1, B::WilsonFields_4D_MPILattice) where {L1<:GeneralFermion}
    substitute!(A.field, B.f)
    set_halo!(A.field)
end



function gauss_distribution_fermion!(
    x::L1;
    seed=nothing,
    sweep::Integer=0,
    direction::Integer=0,
    color::Integer=0,
    subgroup::Integer=_PSEUDOFERMION_STREAM_TAG,
    rng_algorithm=LatticeMatrices.Philox4x32(),
) where {TF,D,T,AT,NC,NG,nw,DI,L1<:GeneralFermion{TF,D,T,AT,NC,NG,nw,DI}}
    real_type = typeof(real(zero(eltype(x.field.A))))
    sigma = sqrt(real_type(0.5))
    shared_seed = _shared_fermion_noise_seed(x.field, seed)
    LatticeMatrices.randomize_gaussian_matrix!(
        x.field;
        sigma,
        seed=shared_seed,
        sweep,
        direction,
        color,
        subgroup,
        rng_algorithm,
    )
    return x
end

@inline function LatticeMatrices.mul_AshiftB!(
    C::L1,
    A::TG,
    B::L2, shift) where {TG<:Fields_4D_MPILattice,L1<:GeneralFermion,L2<:GeneralFermion}

    LatticeMatrices.mul_AshiftB!(C.field, A.U, B.field, shift)

end

@inline function LatticeMatrices.mul_shiftAshiftB!(
    C::L1,
    A::TG,
    B::L2, shiftA, shiftB) where {TG<:Fields_4D_MPILattice,L1<:GeneralFermion,L2<:GeneralFermion}

    LatticeMatrices.mul_shiftAshiftB!(C.field, A.U, B.field, shiftA, shiftB)
end


@inline function add_fermion!(
    c::Tc,
    α::Number,
    a::Ta,
) where {
    Tc<:GeneralFermion,
    Ta<:GeneralFermion}#c += alpha*a 

    add_matrix!(c.field, a.field, α)
end

@inline function add_fermion!(
    c::Tc,
    a::Ta,
    α::Number) where {
    Tc<:GeneralFermion,
    Ta<:GeneralFermion}#c += alpha*a 

    add_matrix!(c.field, a.field, α)
end


@inline function add_fermion!(
    c::Tc,
    α::Number,
    a::Ta,
    β::Number,
    b::Tb) where {
    Tc<:GeneralFermion,
    Ta<:GeneralFermion,
    Tb<:GeneralFermion}

    add_matrix!(c.field, a.field, b.field, α, β)
end

@inline function clear_fermion!(a::L1; sethalo=false) where {L1<:GeneralFermion}
    clear_matrix!(a.field)
    if sethalo
        set_halo!(a.field)
    end
end

#C = A*B A is a matrix.
@inline function LinearAlgebra.mul!(C::L1,
    A::TA, B::L2) where {L1<:GeneralFermion,L2<:GeneralFermion,TA<:AbstractMatrix}
    mul!(C.field, A, B.field)
end


#C = A*B, B is a matrix.
@inline function LinearAlgebra.mul!(C::L1,
    A::L2, B::TB) where {L1<:GeneralFermion,L2<:GeneralFermion,TB<:AbstractMatrix}
    mul!(C.field, A.field, B)
end


"""
    DdagDgeneral(U, field, apply_D!, apply_Ddag!; kwargs...)

Composite `D†D` operator built from user callbacks. Callbacks have the
signature `apply!(out, U1, U2, U3, U4, input, fermion_temps, gauge_temps)` and
must overwrite `out`.
"""
struct DdagDgeneral{TmulD,TmulDdag,TG,TF} <: DdagD_operator
    apply_D::TmulD #(phitemp1, DdagD.U[1], DdagD.U[2], DdagD.U[3], DdagD.U[4], x, phitemp, temp)
    apply_Ddag::TmulDdag #(phitemp1, DdagD.U[1], DdagD.U[2], DdagD.U[3], DdagD.U[4], x, phitemp, temp)
    U::Vector{TG}
    _temporary_gaugefield::PreallocatedArray{TG}
    _temporary_fermion::PreallocatedArray{TF}
    _temporary_fermion_forCG::PreallocatedArray{TF}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_print::Verbose_print
    boundarycondition::Vector{Int8}
    numtemp::Int64


    function DdagDgeneral(U::Vector{TG}, x::TF, apply_D::TmulD, apply_Ddag::TmulDdag;
        numcg=4, num=5, numg=4, eps_CG=1e-12, maxsteps=10000, verbose_level=2, numtemp=4) where {
        TmulD,TmulDdag,TG<:Fields_4D_MPILattice,TF<:GeneralFermion}

        _temporary_gaugefield = PreallocatedArray(U[1]; num=numg)
        _temporary_fermion_forCG = PreallocatedArray(x; num=numcg)
        _temporary_fermion = PreallocatedArray(x; num=num)
        boundarycondition = zeros(Int8, length(x.field.phases))
        boundarycondition .= x.field.phases

        verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))

        return new{TmulD,TmulDdag,TG,TF}(apply_D, apply_Ddag, U,
            _temporary_gaugefield, _temporary_fermion, _temporary_fermion_forCG,
            eps_CG, maxsteps, verbose_print, boundarycondition, numtemp)
    end

    function DdagDgeneral(apply_D::TmulD, apply_Ddag::TmulDdag, U::Vector{TG},
        _temporary_gaugefield, _temporary_fermion::PreallocatedArray{TF}, _temporary_fermion_forCG,
        eps_CG, MaxCGstep, verbose_print, boundarycondition, numtemp) where {
        TmulD,TmulDdag,TG<:Fields_4D_MPILattice,TF<:GeneralFermion}

        return new{TmulD,TmulDdag,TG,TF}(apply_D, apply_Ddag, U,
            _temporary_gaugefield, _temporary_fermion, _temporary_fermion_forCG,
            eps_CG, MaxCGstep, verbose_print, boundarycondition, numtemp)
    end
end
export DdagDgeneral

function (D::DdagDgeneral)(U::Vector{TG}) where {TG<:Fields_4D_MPILattice}
    return DdagDgeneral(D.apply_D, D.apply_Ddag, U,
        D._temporary_gaugefield, D._temporary_fermion, D._temporary_fermion_forCG,
        D.eps_CG, D.MaxCGstep, D.verbose_print, D.boundarycondition, D.numtemp)
end

function update_U!(DdagD::DdagDgeneral, U)
    dim = length(U)
    for μ = 1:dim
        DdagD.U[μ] = U
    end
end


function get_eps(A::T2) where {T2<:DdagDgeneral}
    return A.eps_CG
end

function get_maxsteps(A::T2) where {T2<:DdagDgeneral}
    return A.MaxCGstep
end

function get_verbose(A::T2) where {T2<:DdagDgeneral}
    return A.verbose_print
end

function get_temporaryvectors_forCG(A::T) where {T<:DdagDgeneral}
    return A._temporary_fermion_forCG
end

function get_boundarycondition(A::T2) where {T2<:DdagDgeneral}
    return A.boundarycondition
end

function LinearAlgebra.mul!(y::L1, DdagD::DdagDgeneral, x::L1) where {L1<:GeneralFermion}
    phitemp1, it_phitemp1 = get_block(DdagD._temporary_fermion)
    numtemp = DdagD.numtemp
    temp, ittemp = get_block(DdagD._temporary_gaugefield, numtemp)
    phitemp, itphitemp = get_block(DdagD._temporary_fermion, numtemp)

    DdagD.apply_D(phitemp1, DdagD.U[1], DdagD.U[2], DdagD.U[3], DdagD.U[4], x, phitemp, temp)
    set_wing_fermion!(phitemp1)

    DdagD.apply_Ddag(y, DdagD.U[1], DdagD.U[2], DdagD.U[3], DdagD.U[4], phitemp1, phitemp, temp)
    set_wing_fermion!(y)

    unused!(DdagD._temporary_fermion, it_phitemp1)
    unused!(DdagD._temporary_fermion, itphitemp)
    unused!(DdagD._temporary_gaugefield, ittemp)
end


function set_wing_fermion!(F::L1) where {L1<:GeneralFermion}
    set_halo!(F.field)
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:GeneralFermion}
    @assert boundarycondition ≈ F.field.phases "boundarycondition = $boundarycondition $(F.field.phases)"
    set_halo!(F.field)
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::L1,
    b::Number,
    Y::L2,
) where {L1<:GeneralFermion,L2<:GeneralFermion}

    axpby!(a, X.field, b, Y.field)
    set_halo!(Y.field)
end
