import LatticeMatrices: apply_F_5D!, apply_δF_5D!, D4x_5D!

abstract type DomainwallField_5D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,
} <: AbstractFermionfields_5D{NC} end

"""Common five-dimensional LatticeMatrices-backed domain-wall field."""
struct DomainwallFermion_5D_MPILattice{
    NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,
} <: DomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    f::Tf#LatticeMatrix{5,T,AT,NC,4}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    Dirac_operator::String
    singleprecision::Bool
    L5::Int64

    function DomainwallFermion_5D_MPILattice(
        NC::Tn,
        NX::Tn,
        NY::Tn,
        NZ::Tn,
        NT::Tn,
        L5; NDW=1,
        singleprecision=false,
        operator_name="MobiusDomainwall",
        boundarycondition=[1, 1, 1, -1, 1],
        PEs=nothing,
        comm=nothing, kwargs...) where {Tn<:Integer}




        Dirac_operator = String(operator_name)
        NG = 4

        comm0 = prepare_communicator(resolve_communicator(comm))

        gsize = (NX, NY, NZ, NT, L5)
        dim = 5
        nw = NDW
        @assert NDW > 0 "NDW should be larger than 0. We use a halo area."
        elementtype = ifelse(singleprecision, ComplexF32, ComplexF64)
        phases = boundarycondition
        #@info phases

        nprocs = comm_size(comm0)
        if isnothing(PEs)
            PEs_in = (1, 1, 1, nprocs, 1)
        else
            PEs_in = deepcopy(PEs)
        end
        if length(PEs_in) == 4
            PEs_in = (PEs_in[1], PEs_in[2], PEs_in[3], PEs_in[4], 1)
        end
        PEs_in[5] == 1 || throw(ArgumentError(
            "DomainwallFermion_5D_MPILattice does not distribute the fifth coordinate"))

        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        @assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        @assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"
        @assert L5 % PEs_in[5] == 0 "L5 % PEs[5] should be 0. Now L5 = $L5 and PEs = $PEs_in"

        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        f = LatticeMatrix(NC, NG, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        #println("fpn", f.PN)
        T = elementtype
        AT = typeof(f.A)
        Tf = typeof(f)


        return new{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NDW,
            Dirac_operator,
            singleprecision,
            L5)
    end

    function DomainwallFermion_5D_MPILattice(
        u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW}, L5
        ; boundarycondition=[1, 1, 1, -1, 1],
        operator_name="MobiusDomainwall", kwargs...) where {NC,NX,NY,NZ,NT,T,AT,NDW}


        x = DomainwallFermion_5D_MPILattice(
            NC,
            NX,
            NY,
            NZ,
            NT,
            L5;
            NDW=u.NDW,
            singleprecision=u.singleprecision,
            operator_name,
            boundarycondition,
            PEs=get_PEs(u.U),
            comm=u.U.comm)

        return x
    end

    function DomainwallFermion_5D_MPILattice(
        u::AbstractGaugefields, L5
        ; boundarycondition=[1, 1, 1, -1, 1],
        operator_name="MobiusDomainwall", kwargs...)

        NC = u.NC
        NX = u.NX
        NY = u.NY
        NZ = u.NZ
        NT = u.NT

        x = DomainwallFermion_5D_MPILattice(
            NC,
            NX,
            NY,
            NZ,
            NT,
            L5;
            operator_name,
            boundarycondition)

        return x
    end

end

get_myrank(x::DomainwallFermion_5D_MPILattice) = comm_rank(x.f.comm)
get_nprocs(x::DomainwallFermion_5D_MPILattice) = comm_size(x.f.comm)
barrier(x::DomainwallFermion_5D_MPILattice) = communicator_barrier(x.f.comm)

const MobiusDomainwallField_5D_MPILattice = DomainwallField_5D_MPILattice
const MobiusDomainwallFermion_5D_MPILattice = DomainwallFermion_5D_MPILattice

export DomainwallField_5D_MPILattice,
    DomainwallFermion_5D_MPILattice,
    MobiusDomainwallFermion_5D_MPILattice

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::TX,
    b::Number,
    Y::TY,
) where {TX<:MobiusDomainwallField_5D_MPILattice,TY<:MobiusDomainwallField_5D_MPILattice}

    axpby!(a, X.f, b, Y.f)
    set_halo!(Y.f)
end

function substitute_fermion!(A::TA, B::TB) where {TA<:MobiusDomainwallFermion_5D_MPILattice,
    TB<:MobiusDomainwallFermion_5D_MPILattice}
    substitute!(A.f, B.f)
    set_halo!(A.f)
end



function gauss_distribution_fermion!(
    x::Tx;
    seed=nothing,
    sweep::Integer=0,
    direction::Integer=0,
    color::Integer=0,
    subgroup::Integer=_PSEUDOFERMION_STREAM_TAG,
    rng_algorithm=LatticeMatrices.Philox4x32(),
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}
    real_type = typeof(real(zero(eltype(x.f.A))))
    sigma = sqrt(real_type(0.5))
    shared_seed = _shared_fermion_noise_seed(x.f, seed)
    LatticeMatrices.randomize_gaussian_matrix!(
        x.f;
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

function gauss_distribution_fermion!(
    x::Tx,
    randomfunc,
    σ,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}

    work = zeros(T, NC, 4, NX, NY, NZ, NT, L5)
    work = map(i -> gauss_distribution(σ), work)
    PEs = get_PEs(x.f)
    a = LatticeMatrix(
        work, 5, PEs; nw=x.NDW, phases=x.f.phases, comm0=x.f.comm)
    substitute!(x.f, a)
end





function Base.similar(x::Tx) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}

    return DomainwallFermion_5D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT,
        L5;
        NDW=x.NDW,
        singleprecision=x.singleprecision,
        operator_name=x.Dirac_operator,
        boundarycondition=x.f.phases,
        PEs=get_PEs(x.f),
        comm=x.f.comm)
end

"""
    Initialize_DomainwallFermion(u::Gaugefields_4D_MPILattice, L5; kwargs...)

Construct the standard five-dimensional LatticeMatrices-backed field used by
the Shamir, Möbius, and generalized domain-wall operators.  The fifth
direction is local by default (`PEs[5] == 1`).
"""
function Initialize_DomainwallFermion(
    u::Gaugefields_4D_MPILattice,
    L5;
    nowing=true,
    kwargs...,
)
    return DomainwallFermion_5D_MPILattice(
        u, L5; operator_name="Domainwall", kwargs...)
end

function Initialize_MobiusDomainwallFermion(
    u::Gaugefields_4D_MPILattice,
    L5;
    nowing=true,
    kwargs...,
)
    return DomainwallFermion_5D_MPILattice(
        u, L5; operator_name="MobiusDomainwall", kwargs...)
end

struct Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    f::Shifted_Lattice{Tf,5}
end

function Shifted_MobiusDomainwallFermion_5D_MPILattice(x::Tx, shift) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}
    sx = Shifted_Lattice(x.f, shift)
    s = Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(sx)
    return s
end


#lattice shift
function shift_fermion(F::TF, ν::T1; boundarycondition=nothing) where {TF<:MobiusDomainwallFermion_5D_MPILattice,T1<:Integer}

    if boundarycondition != nothing
        @assert F.f.phases ≈ boundarycondition "boundary condition is wrong now the boudnary condition of the fermions is $(F.f.phases) but you want to use $boundarycondition"
    end
    #println(F.f.phases)

    if ν == 1
        shift = (1, 0, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1, 0)
    elseif ν == 5
        shift = (0, 0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1, 0)
    elseif ν == -5
        shift = (0, 0, 0, 0, -1)
    end

    s = shift_fermion(F, shift)

    return s
end


function shift_fermion(
    F::TF,
    shift::NTuple{5,T1},
) where {T1<:Integer,TF<:MobiusDomainwallFermion_5D_MPILattice}

    s = Shifted_MobiusDomainwallFermion_5D_MPILattice(F, shift)

    return s
end

struct Adjoint_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    f::Adjoint_Lattice{Tf}
end

struct Adjoint_Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    f::Adjoint_Lattice{Shifted_Lattice{Tf,5}}
end

function Base.adjoint(x::MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    Adjoint_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(x.f')
end

function Base.adjoint(x::Tx) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}
    Adjoint_Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(x.f')
end

@static if isdefined(LatticeMatrices, :release!)
    function LatticeMatrices.release!(x::Shifted_MobiusDomainwallFermion_5D_MPILattice)
        LatticeMatrices.release!(x.f)
        return nothing
    end

    function LatticeMatrices.release!(x::Adjoint_Shifted_MobiusDomainwallFermion_5D_MPILattice)
        LatticeMatrices.release!(x.f)
        return nothing
    end

    Base.close(x::Shifted_MobiusDomainwallFermion_5D_MPILattice) = LatticeMatrices.release!(x)
    Base.close(x::Adjoint_Shifted_MobiusDomainwallFermion_5D_MPILattice) = LatticeMatrices.release!(x)
    Base.isopen(x::Shifted_MobiusDomainwallFermion_5D_MPILattice) = isopen(x.f)
    Base.isopen(x::Adjoint_Shifted_MobiusDomainwallFermion_5D_MPILattice) = isopen(x.f)
end

function Base.adjoint(x::Adjoint_MobiusDomainwallFermion_5D_MPILattice)
    Adjoint_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(x.f')
end

include("linearalgebra_5D.jl")

function LinearAlgebra.dot(
    A::TA,
    B::TB,
) where {TA<:MobiusDomainwallFermion_5D_MPILattice,TB<:MobiusDomainwallFermion_5D_MPILattice}

    s = dot(A.f, B.f)
    return s
end



struct D5DW_MobiusDomainwall_operator_MPILattice{Dim,TU,fermion,TD} <:
       Dirac_operator{Dim} where {TU<:AbstractGaugefields}
    U::Array{TU,1}
    D::TD #D5DW_MobiusDomainwallOperator5D{T,L5}
    mass::Float64
    _temporary_fermi::Temporalfields{fermion}#Array{fermion,1}
    L5::Int64
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}# Vector{fermion}
    #_temporary_fermion_forCG::Vector{fermion}
    boundarycondition::Vector{ComplexF64}
    b::Float64 #coefficient for MobiusDomainwall
    c::Float64 #coefficient for MobiusDomainwall
    M::Float64
end

function D5DW_MobiusDomainwall_operator_MPILattice(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x::Tx,
    parameters,
    mass,
    b,
    c,
) where {NC,Dim,Tx<:MobiusDomainwallFermion_5D_MPILattice}
    @assert haskey(parameters, "L5") "parameters should have the keyword L5"
    L5 = parameters["L5"]
    L5 == x.L5 || throw(DimensionMismatch(
        "operator L5=$L5 does not match field L5=$(x.L5)"))
    @assert Dim == 4 "Dim should be 4!"

    M = check_parameters(parameters, "M", -1)

    TU = eltype(U)
    UL = [U[1].U, U[2].U, U[3].U, U[4].U]
    D = D5DW_MobiusDomainwallOperator5D(UL, L5, mass, M, b, c)
    TD = typeof(D)

    num = 4
    _temporary_fermi = Temporalfields(x; num)

    numcg = 7
    _temporary_fermion_forCG = Temporalfields(x; num=numcg)#Array{xtype,1}(undef, numcg)


    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    #println("eps_CG = ",eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

    verbose_level = check_parameters(parameters, "verbose_level", 2)
    verbose_print = Verbose_print(verbose_level)

    method_CG = check_parameters(parameters, "method_CG", "bicg")
    #println("xtype ",xtype)


    boundarycondition = x.f.phases
    #println(x.f.phases)


    return D5DW_MobiusDomainwall_operator_MPILattice{Dim,TU,Tx,TD}(
        U,
        D,
        mass,
        _temporary_fermi,
        L5,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG,
        boundarycondition,
        b,
        c,
        M
    )
    #=
    U::Array{TU,1}
    D::TD #D5DW_MobiusDomainwallOperator5D{T,L5}
    mass::Float64
    _temporary_fermi::Temporalfields{fermion}#Array{fermion,1}
    L5::Int64
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}# Vector{fermion}
    #_temporary_fermion_forCG::Vector{fermion}
    boundarycondition::Vector{Int8}
    b::Float64 #coefficient for MobiusDomainwall
    c::Float64 #coefficient for MobiusDomainwall
    =#

end
export D5DW_MobiusDomainwall_operator_MPILattice

function (D::D5DW_MobiusDomainwall_operator_MPILattice{Dim,TU,fermion,TD})(
    U,
) where {Dim,TU,fermion,TD}
    WD = D5DW_MobiusDomainwallOperator5D([U[1].U, U[2].U, U[3].U, U[4].U], D.L5, D.mass, D.M, D.b, D.c)
    #WD = WilsonDiracOperator4D([U[1].U, U[2].U, U[3].U, U[4].U], D.κ)
    return D5DW_MobiusDomainwall_operator_MPILattice{Dim,TU,fermion,TD}(
        U,
        WD,
        D.mass,
        D._temporary_fermi,
        D.L5,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG,
        D.boundarycondition,
        D.b,
        D.c,
        D.M
    )
end

struct Adjoint_D5DW_MobiusDomainwall_operator_MPILattice{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_MobiusDomainwall_operator_MPILattice}
    Adjoint_D5DW_MobiusDomainwall_operator_MPILattice{typeof(A)}(A)
end

"""
    combine_generalized_domainwall_force!(X, X0, Z, a, b, c)

Apply the slice-dependent coefficients required by the generalized
domain-wall force:

```
X[s] = a[s] * (c[s] * X[s] + b[s] * X0[s])
Z[s] = a[s] * c[s] * Z[s].
```

The coefficient vectors must live on the active JACC backend. The fifth
coordinate is deliberately not MPI-distributed for this field type.
"""
function combine_generalized_domainwall_force!(
    X::DomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5},
    X0::DomainwallFermion_5D_MPILattice,
    Z::DomainwallFermion_5D_MPILattice,
    a,
    b,
    c,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    for (name, coefficients) in (("a", a), ("b", b), ("c", c))
        length(coefficients) == L5 || throw(DimensionMismatch(
            "$name must have length L5=$L5"))
    end
    X0.L5 == L5 && Z.L5 == L5 || throw(DimensionMismatch(
        "all generalized domain-wall force fields must have the same L5"))

    _mark_halo_dirty!(X.f)
    _mark_halo_dirty!(Z.f)
    JACC.parallel_for(
        prod(X.f.PN),
        kernel_combine_generalized_domainwall_force!,
        X.f.A,
        X0.f.A,
        Z.f.A,
        a,
        b,
        c,
        Val(NC),
        Val(NDW),
        X.f.indexer,
    )
    set_halo!(X.f)
    set_halo!(Z.f)
    return nothing
end

@inline function kernel_combine_generalized_domainwall_force!(
    i,
    X,
    X0,
    Z,
    a,
    b,
    c,
    ::Val{NC},
    ::Val{nw},
    dindexer,
) where {NC,nw}
    indices = delinearize(dindexer, i, nw)
    s = indices[5] - nw
    ab = a[s] * b[s]
    ac = a[s] * c[s]
    @inbounds for spin in 1:4, color in 1:NC
        X[color, spin, indices...] =
            ac * X[color, spin, indices...] +
            ab * X0[color, spin, indices...]
        Z[color, spin, indices...] = ac * Z[color, spin, indices...]
    end
    return nothing
end

function LinearAlgebra.mul!(
    c::Tc,
    a::Ta,
    b::Tb,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tc<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5},
    Tb<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5},Ta<:D5DW_MobiusDomainwall_operator_MPILattice}
    #println(typeof(c.f))
    #println(typeof(b.f))
    mul!(c.f, a.D, b.f)

    set_halo!(c.f)

    #@code_warntype mul!(c.f, a.U, b.f)
    #error("dd")
end

function LinearAlgebra.mul!(
    c::Tc,
    a::Ta,
    b::Tb,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tc<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5},
    Tb<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5},Ta<:Adjoint_D5DW_MobiusDomainwall_operator_MPILattice}
    #println(typeof(c.f))
    #println(typeof(b.f))
    mul!(c.f, a.parent.D', b.f)
    #@code_warntype mul!(c.f, a.U, b.f)
    set_halo!(c.f)

end

function LinearAlgebra.mul!(
    c::MobiusDomainwallField_5D_MPILattice,
    a::T,
    b::MobiusDomainwallField_5D_MPILattice,
) where {T<:Number}
    #println(typeof(c.f))
    #println(typeof(a.U))
    #println(typeof(b.f))

    mul!(c.f, a, b.f)
    #@code_warntype mul!(c.f, a.U, b.f)
end

function set_wing_fermion!(F::MobiusDomainwallFermion_5D_MPILattice)
    set_halo!(F.f)
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:MobiusDomainwallFermion_5D_MPILattice}
    #@info boundarycondition
    #@info F.f.phases
    @assert boundarycondition ≈ F.f.phases "boundarycondition = $boundarycondition $(F.f.phases)"
    set_halo!(F.f)
end


function clear_fermion!(a::MobiusDomainwallFermion_5D_MPILattice; sethalo=false)
    clear_matrix!(a.f)
    if sethalo
        set_halo!(a.f)
    end
end


function add_fermion!(
    c::Tc,
    α::Number,
    a::Ta,
) where {
    Tc<:MobiusDomainwallFermion_5D_MPILattice,
    Ta<:MobiusDomainwallField_5D_MPILattice}#c += alpha*a 

    add_matrix!(c.f, a.f, α)
end

function add_fermion!(
    c::Tc,
    α::Number,
    a::Ta,
    β::Number,
    b::Tb) where {
    Tc<:MobiusDomainwallFermion_5D_MPILattice,
    Ta<:MobiusDomainwallField_5D_MPILattice,
    Tb<:MobiusDomainwallField_5D_MPILattice}

    add_matrix!(c.f, a.f, b.f, α, β)
end

function apply_F!(
    xout::Txout,
    L5,
    m,
    x::Tx,
    temp1,
) where {Tx<:MobiusDomainwallFermion_5D_MPILattice,
    Txout<:MobiusDomainwallFermion_5D_MPILattice}
    clear_fermion!(xout)
    @assert L5 == xout.L5 "L5 should be same"

    apply_F_5D!(xout.f, m, L5, x.f)
    set_halo!(xout.f)

end


function apply_δF!(
    xout::Txout,
    L5,
    m,
    x::Tx,
    temp1,
) where {Tx<:MobiusDomainwallFermion_5D_MPILattice,
    Txout<:MobiusDomainwallFermion_5D_MPILattice}
    clear_fermion!(xout)
    @assert L5 == xout.L5 "L5 should be same"

    apply_δF_5D!(xout.f, m, L5, x.f)
    set_halo!(xout.f)

end




function LinearAlgebra.mul!(
    c::Tc,
    a::Ta,
    b::Tb,
) where {Tc<:MobiusDomainwallFermion_5D_MPILattice,
    Tb<:MobiusDomainwallField_5D_MPILattice,Ta<:Fields_4D_MPILattice}

    mul!(c.f, a.U, b.f)

    #set_wing_fermion!(c)
end

function mul_1plusγμx!(temp1_f::Tf1, temp0_f::Tf0, μ) where {Tf1<:MobiusDomainwallFermion_5D_MPILattice,
    Tf0<:MobiusDomainwallFermion_5D_MPILattice}
    mul!(temp1_f.f, Oneγ(μ, true), temp0_f.f)
end

function mul_1minusγμx!(temp1_f::Tf1, temp0_f::Tf0, μ) where {Tf1<:MobiusDomainwallFermion_5D_MPILattice,
    Tf0<:MobiusDomainwallFermion_5D_MPILattice}
    mul!(temp1_f.f, Oneγ(μ, false), temp0_f.f)
end

function mul_x1plusγμ!(y::MobiusDomainwallFermion_5D_MPILattice, x::MobiusDomainwallFermion_5D_MPILattice, μ)
    #mul_1plusγμx!(y, x, μ)
    if μ == 1
        mul_1minusγμx!(y, x, 1)
    elseif μ == 2
        mul_1plusγμx!(y, x, 2)
    elseif μ == 3
        mul_1minusγμx!(y, x, 3)
    elseif μ == 4
        mul_1plusγμx!(y, x, 4)
    end
end


function muladd_U!(UdSfdU, coeff, temp0_g, temp0_f, f::Tf, temp1_f) where {
    Tf<:Adjoint_MobiusDomainwallFermion_5D_MPILattice}

    #s1 = dot(f.f',f.f')
    #s2 = dot(temp0_f,temp0_f)

    mul_sum!(temp0_g.U, temp0_f.f, f.f, temp1_f.f)
    #println(tr(temp0_g.U)," $s1 $s2")
    #display(temp0_g.U.A[:,:,2,2,2,2])
    add_U!(UdSfdU, coeff, temp0_g)
end

#C = A*B'
function mul_sum!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw,DIC},
    A::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}, B::Adjoint_Lattice{L},
    temp::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}) where {T1,T2,T3,AT1,AT2,AT3,
    NC1,NC2,NC3,nw,DIA,DIB,DIC,
    L<:LatticeMatrix{5,T3,AT3,NC2,NC3,nw,DIB}}


    clear_matrix!(C)
    #=
    println("C = A * B'")
    Ai = A.A[:,:,2,2,2,2,2]
    Bi = B.data.A[:,:,2,2,2,2,2]
    display(Bi)
    ABi = zero(C.A[:,:,2,2,2,2])
    _,_,N1,N2,N3,N4,Lsize = size(A.A)
    println(size(A.A))
    for i=1+nw:Lsize-nw
        ai = A.A[:,:,2,2,2,2,i]
        bi = B.data.A[:,:,2,2,2,2,i]
        println("i = $i")
        display(ai*bi')
        ABi += A.A[:,:,2,2,2,2,i]*B.data.A[:,:,2,2,2,2,i]'
    end
    display(ABi)
    =#

    #JACC.parallel_for(
    #    prod(A.PN), kernel_Dmatrix_mul_455ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), A.indexer
    #)
    _mark_halo_dirty!(temp)
    JACC.parallel_for(
        prod(temp.PN), kernel_Dmatrix_mulsum_455ABdag!, temp.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), temp.indexer
    )
    #=
    tempi = temp.A[1:NC1,1:NC2,2,2,2,2,2]
    display(tempi)
    =#

    _, _, N1, N2, N3, N4, Lsize = size(A.A)

    for i = 1+nw:Lsize-nw
        C.A[1:NC1, 1:NC2, 1:N1, 1:N2, 1:N3, 1:N4] .+= view(temp.A, 1:NC1, 1:NC2, 1:N1, 1:N2, 1:N3, 1:N4, i)
    end
    #=
    Ci = C.A[1:NC1,1:NC2,2,2,2,2]
    display(Ci)
    =#
    #set_halo!(C)
end


@inline function kernel_Dmatrix_mulsum_455ABdag!(i, temp, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix, iy, iz, it, i5 = indices
    #println(indices)

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            temp[ic, jc, ix, iy, iz, it, i5] = zero(eltype(temp))
        end

        for ic = 1:NC1
            for kc = 1:NC3
                temp[ic, jc, ix, iy, iz, it, i5] += A[ic, kc, indices...] * B[jc, kc, indices...]'
            end
        end
    end
end



function muladd_U!(UdSfdU, coeff, temp0_g, temp0_f, f::Tf, temp1_f) where {TTf<:MobiusDomainwallFermion_5D_MPILattice,
    Tf<:TTf}
    mul_sum!(temp0_g.U, temp0_f.f, f.f, temp1_f.f)
    add_U!(UdSfdU, coeff, temp0_g)
end

#C = A*B^T
function mul_sum!(C::LatticeMatrix{4,T1,AT1,NC1,NC2,nw,DIC},
    A::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}, B::L,
    temp::LatticeMatrix{5,T2,AT2,NC1,NC3,nw,DIA}) where {T1,T2,T3,AT1,AT2,AT3,
    NC1,NC2,NC3,nw,DIA,DIB,DIC,
    L<:LatticeMatrix{5,T3,AT3,NC2,NC3,nw,DIB}}


    clear_matrix!(C)
    #println("C = A * B'")
    #display(A.A[:,:,2,2,2,2,2])
    #display(B.data.A[:,:,2,2,2,2,2])

    #JACC.parallel_for(
    #    prod(A.PN), kernel_Dmatrix_mul_455ABdag!, C.A, A.A, B.data.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), A.indexer
    #)
    _mark_halo_dirty!(temp)
    JACC.parallel_for(
        prod(A.PN), kernel_Dmatrix_mulsum_455AB!, temp.A, A.A, B.A, Val(NC1), Val(NC2), Val(NC3), Val(nw), A.indexer
    )
    _, _, N1, N2, N3, N4, Lsize = size(A.A)

    for i = 1+nw:Lsize-nw
        C.A[1:NC1, 1:NC2, :, :, :, :] .+= view(temp.A, 1:NC1, 1:NC2, :, :, :, :, i)
    end
    #set_halo!(C)
end

@inline function kernel_Dmatrix_mulsum_455AB!(i, temp, A, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    ix, iy, iz, it, i5 = indices

    @inbounds for jc = 1:NC2
        for ic = 1:NC1
            temp[ic, jc, ix, iy, iz, it, i5] = zero(eltype(temp))
        end

        for ic = 1:NC1
            for kc = 1:NC3
                temp[ic, jc, ix, iy, iz, it, i5] += A[ic, kc, indices...] * B[jc, kc, indices...]
            end
        end
    end
end



function LinearAlgebra.mul!(
    c::Tc,
    a::Ta,
    b::Tb,
) where {Ta<:MobiusDomainwallField_5D_MPILattice,
    Tc<:MobiusDomainwallField_5D_MPILattice,
    Tb<:Abstractfields}
    mul!(c.f, a.f, b.U)
    #set_wing_fermion!(c)
end

function D4x_5D!(C::Tc, U::Vector{Tu}, ψ::Tp, coeff) where {
    Tc<:MobiusDomainwallFermion_5D_MPILattice,
    Tu<:AbstractGaugefields,
    Tp<:MobiusDomainwallFermion_5D_MPILattice}

    D4x_5D!(C.f, [U[1].U, U[2].U, U[3].U, U[4].U], ψ.f, coeff)
    set_halo!(C.f)
end

function apply_F_5D!(C::Tc, mass, L5, ψ::Tp) where {
    Tc<:MobiusDomainwallFermion_5D_MPILattice,
    Tp<:MobiusDomainwallFermion_5D_MPILattice}

    apply_F_5D!(C.f, mass, L5, ψ.f)
    set_halo!(C.f)
end

function Z4_distribution_fermi!(x::MobiusDomainwallField_5D_MPILattice)
    ZN_distribution_fermi!(x, 4)
end

function ZN_distribution_fermi!(
    x::Tx, N
) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}
    NG = 4
    work = zeros(ComplexF64, NC, NG, NX, NY, NZ, NT, L5)
    for it = 1:NT
        for iz = 1:NZ
            for iy = 1:NY
                for ix = 1:NX
                    for ialpha = 1:NG
                        @inbounds @simd for ic = 1:NC
                            work[ic, ialpha, ix, iy, iz, it, 1] =
                                N == 4 ? _z4_root(rand(0:3)) : cis(2π * rand(0:N-1) / N)
                        end
                    end
                end
            end
        end
    end
    PEs = get_PEs(x.f)
    a = LatticeMatrix(work, 5, PEs; nw=1, phases=x.f.phases, comm0=x.f.comm)
    substitute!(x.f, a)
    return
end
