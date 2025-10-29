



abstract type MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: Abstract_MobiusDomainwallFermion_5D{NC,nothing} end

"""
Struct for MobiusDomainwallFermion
"""
struct MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
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

    function MobiusDomainwallFermion_5D_MPILattice(
        NC::Tn,
        NX::Tn,
        NY::Tn,
        NZ::Tn,
        NT::Tn,
        L5; NDW=1,
        singleprecision=false,
        boundarycondition=[1, 1, 1, -1, 1],
        PEs=nothing,
        comm=MPI.COMM_WORLD, kwargs...) where {Tn<:Integer}




        Dirac_operator = "MobiusDomainwall"
        NG = 4

        if MPI.Initialized() == false
            MPI.Init()
            mpiinit = true
        end

        comm0 = comm

        gsize = (NX, NY, NZ, NT, L5)
        dim = 5
        nw = NDW
        @assert NDW > 0 "NDW should be larger than 0. We use a halo area."
        elementtype = ifelse(singleprecision, ComplexF32, ComplexF64)
        phases = boundarycondition
        nprocs = MPI.Comm_size(comm)
        if isnothing(PEs)
            PEs_in = (1, 1, 1, nprocs, 1)
        else
            PEs_in = deepcopy(PEs)
        end
        if length(PEs_in) == 4
            PEs_in = (PEs_in[1], PEs_in[2], PEs_in[3], PEs_in[4], 1)
        end

        @assert NX > PEs_in[1] "PEs[1] is larger than NX. Now NX = $NX and PEs = $PEs_in"
        @assert NY > PEs_in[2] "PEs[2] is larger than NY. Now NX = $NY and PEs = $PEs_in"
        @assert NZ > PEs_in[3] "PEs[3] is larger than NZ. Now NX = $NZ and PEs = $PEs_in"
        @assert NT > PEs_in[4] "PEs[4] is larger than NT. Now NX = $NT and PEs = $PEs_in"
        @assert L5 > PEs_in[5] "PEs[5] is larger than NT. Now NX = $L5 and PEs = $PEs_in"


        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        @assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        @assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"
        @assert L5 % PEs_in[4] == 0 "L5 % PEs[5] should be 0. Now NT = $L5 and PEs = $PEs_in"

        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        #verbose_print = Verbose_print(verbose_level, myid=myrank)

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

    function MobiusDomainwallFermion_5D_MPILattice(
        u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW}, L5
        ; boundarycondition=[1, 1, 1, -1, 1], kwargs...) where {NC,NX,NY,NZ,NT,T,AT,NDW}


        x = MobiusDomainwallFermion_5D_MPILattice(
            NC,
            NX,
            NY,
            NZ,
            NT,
            L5;
            NDW=u.NDW,
            singleprecision=u.singleprecision,
            boundarycondition,
            PEs=get_PEs(u.U),
            comm=u.U.comm)

        return x
    end

end

export MobiusDomainwallFermion_5D_MPILattice

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





function substitute_fermion!(A::TA, B::TB) where {NC,WilsonFermion<:WilsonFermion_4D_MPILattice,TA<:MobiusDomainwallFermion_5D_MPILattice,TB<:MobiusDomainwallFermion_5D{NC,WilsonFermion}}
    #dim = 5
    #PEs = A.f.dims
    #phases = A.f.phases
    nw = A.f.nw
    #comm0 = A.f.comm
    L5 = A.L5
    D = similar(A.f)

    for i = 1:L5
        xi = B.w[i]#permutedims(B.w[i][:, :, :, :, :, :, i], (1, 6, 2, 3, 4, 5))
        D.A[:, :, :, :, :, :, i+nw] .= xi.f.A
    end

    substitute!(A.f, D)
    set_halo!(A.f)
end

function substitute_fermion!(A::TA, B::TB) where {NC,WilsonFermion,TA<:MobiusDomainwallFermion_5D_MPILattice,TB<:MobiusDomainwallFermion_5D{NC,WilsonFermion}}
    dim = 5
    PEs = A.f.dims
    phases = A.f.phases
    nw = A.f.nw
    comm0 = A.f.comm
    L5 = A.L5
    _, NX, NY, NZ, NT, NG = size(B.w[1].f)
    D = zeros(eltype(B.w[1].f), NC, NG, NX, NY, NZ, NT, L5)

    for i = 1:L5
        xi = permutedims(B.w[i].f[:, :, :, :, :, :], (1, 6, 2, 3, 4, 5))
        #println(size(xi))
        #println(size(D))
        D[:, :, :, :, :, :, i] .= xi
    end

    tempf = LatticeMatrix(D, dim, PEs;
        nw,
        phases,
        comm0)
    substitute!(A.f, tempf)
    set_halo!(A.f)
end



function Base.similar(x::Tx) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}

    return MobiusDomainwallFermion_5D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT,
        L5;
        NDW=x.NDW,
        singleprecision=x.singleprecision,
        boundarycondition=x.f.phases,
        PEs=get_PEs(x.f),
        comm=x.f.comm)
end

struct Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5} <: MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}
    f::Shifted_Lattice{Tf,5}
end

function Shifted_MobiusDomainwallFermion_5D_MPILattice(x::Tx, shift) where {NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5,Tx<:MobiusDomainwallField_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}}
    sx = Shifted_Lattice(x.f, shift)
    s = Shifted_MobiusDomainwallFermion_5D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,Tf,L5}(sx)
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
    if L5 != x.L5
        @error "L5 in Dirac operator and fermion fields is not same. Now L5 = $L5 and x.L5 = $(x.L5)"
        #@assert L5 == x.L5 "L5 in Dirac operator and fermion fields should be same. Now L5 = $L5 and x.L5 = $(x.L5)"
    end
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

struct Adjoint_D5DW_MobiusDomainwall_operator_MPILattice{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:D5DW_MobiusDomainwall_operator_MPILattice}
    Adjoint_D5DW_MobiusDomainwall_operator_MPILattice{typeof(A)}(A)
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

function set_wing_fermion!(F::MobiusDomainwallFermion_5D_MPILattice)
    set_halo!(F.f)
end