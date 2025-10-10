using JACC
using StaticArrays
import Gaugefields.MPILattice: LatticeMatrix,
    Shifted_Lattice,
    Adjoint_Lattice,
    TALattice,
    makeidentity_matrix!,
    set_halo!,
    substitute!,
    partial_trace,
    get_PEs,
    clear_matrix!,
    add_matrix!,
    expt!,
    get_4Dindex,
    traceless_antihermitian_add!,
    normalize_matrix!,
    randomize_matrix!,
    get_shift,
    gather_and_bcast_matrix,
    traceless_antihermitian!
import Gaugefields.AbstractGaugefields_module: Gaugefields_4D_MPILattice, Fields_4D_MPILattice
import LatticeMatrices: mulT!

include("linearalgebra_4D.jl")

abstract type WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG} <: WilsonFermion_4D{NC} end


struct WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,Tf} <: WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Tf#LatticeMatrix{4,T,AT,NC,NG}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    singleprecision::Bool

    function WilsonFermion_4D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT;
        NDW=1,
        singleprecision=false,
        boundarycondition=[1, 1, 1, -1],
        PEs=nothing,
        comm=MPI.COMM_WORLD)

        Dirac_operator = "Wilson"
        NG = 4

        if MPI.Initialized() == false
            MPI.Init()
            mpiinit = true
        end


        comm0 = comm

        gsize = (NX, NY, NZ, NT)
        dim = 4
        nw = NDW
        @assert NDW > 0 "NDW should be larger than 0. We use a halo area."
        elementtype = ifelse(singleprecision, ComplexF32, ComplexF64)
        phases = boundarycondition
        nprocs = MPI.Comm_size(comm)
        if isnothing(PEs)
            PEs_in = (1, 1, 1, nprocs)
        else
            PEs_in = deepcopy(PEs)
        end

        @assert NX > PEs_in[1] "PEs[1] is larger than NX. Now NX = $NX and PEs = $PEs_in"
        @assert NY > PEs_in[2] "PEs[2] is larger than NY. Now NX = $NY and PEs = $PEs_in"
        @assert NZ > PEs_in[3] "PEs[3] is larger than NZ. Now NX = $NZ and PEs = $PEs_in"
        @assert NT > PEs_in[4] "PEs[4] is larger than NT. Now NX = $NT and PEs = $PEs_in"

        @assert NX % PEs_in[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs_in"
        @assert NY % PEs_in[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs_in"
        @assert NZ % PEs_in[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs_in"
        @assert NT % PEs_in[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs_in"


        @assert prod(PEs_in) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        #verbose_print = Verbose_print(verbose_level, myid=myrank)

        f = LatticeMatrix(NC, NG, dim, gsize, PEs_in;
            nw, elementtype, phases, comm0)
        T = elementtype
        AT = typeof(f.A)


        return new{NC,NX,NY,NZ,NT,T,AT,NDW,NG,typeof(f)}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NG,
            NDW,
            Dirac_operator,
            singleprecision)
    end
end

function Initialize_WilsonFermion(
    u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW}
    ; nowing=false,boundarycondition=[1, 1, 1, -1]) where {NC,NX,NY,NZ,NT,T,AT,NDW}
    @assert nowing == true "nowing should be false."
    x = WilsonFermion_4D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT;
        NDW=u.NDW,
        singleprecision=u.singleprecision,
        boundarycondition,
        PEs=get_PEs(u.U),
        comm=u.U.comm)

    return x
end


function Initialize_pseudofermion_fields(
    u::Gaugefields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW},
    Dirac_operator::String;
    L5=2,
    nowing=true, kwargs...
) where {NC,NX,NY,NZ,NT,T,AT,NDW}

#=
    if Dirac_operator == "staggered"
        error(
            "Dirac_operator  = $Dirac_operator witn nowing = $nowing is not supported",
        )
    elseif Dirac_operator == "Wilson"
        x = Initialize_WilsonFermion(u)
    else
        error("Dirac_operator  = $Dirac_operator is not supported")
    end
    return x
    =#
    Dim = 4
    if Dim == 4
        if Dirac_operator == "staggered"
            x = Initialize_StaggeredFermion(u, nowing=nowing)
        elseif Dirac_operator == "Wilson"
            x = Initialize_WilsonFermion(u, nowing=nowing)
        elseif Dirac_operator == "Domainwall"
            #@warn "Domainwall fermion is not well tested!!"
            x = Initialize_DomainwallFermion(u, L5, nowing=nowing)
        elseif Dirac_operator == "MobiusDomainwall"
            #@warn "MobiusDomainwall fermion is not well tested!!"
            x = Initialize_MobiusDomainwallFermion(u, L5, nowing=nowing)
        elseif Dirac_operator == "GeneralizedDomainwall"
            #@warn "GeneralizedDomainwall fermion is not well tested!!"
            x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)

        else
            error("Dirac_operator = $Dirac_operator is not supported")
        end
    elseif Dim == 2
        if Dirac_operator == "staggered"
            x = Initialize_StaggeredFermion(u)
        elseif Dirac_operator == "Wilson"
            x = Initialize_WilsonFermion(u)
        elseif Dirac_operator == "Domainwall"
            #@warn "Domainwall fermion is not well tested!!"
            x = Initialize_DomainwallFermion(u, L5)
        elseif Dirac_operator == "MobiusDomainwall"
            @warn "MobiusDomainwall fermion is not well tested!!"
            x = Initialize_MobiusDomainwallFermion(u, L5)
        elseif Dirac_operator == "GeneralizedDomainwall"
            @warn "GeneralizedDomainwall fermion is not tested!!"
            x = Initialize_GeneralizedDomainwallFermion(u, L5, nowing=nowing)
        else
            error("Dirac_operator = $Dirac_operator is not supported")
        end
    else
        error("Dim = $Dim is not supported")
    end
    return x

end


function Base.similar(x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    return WilsonFermion_4D_MPILattice(
        NC,
        NX,
        NY,
        NZ,
        NT;
        NDW,
        singleprecision=x.singleprecision,
        boundarycondition=x.f.phases,
        PEs=get_PEs(x.f),
        comm=x.f.comm)
end

function gauss_distribution_fermion!(
    x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    work = zeros(ComplexF64, NC, NG, NX, NY, NZ, NT)
    work = map(i -> gauss_distribution(), work)
    PEs = get_PEs(x.f)
    a = LatticeMatrix(work, 4, PEs; nw=1, phases=x.f.phases, comm0=x.f.comm)
    substitute!(x.f, a)

    return
end

function gauss_distribution_fermion!(
     x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    randomfunc,
    σ,
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    work = zeros(ComplexF64, NC, NG, NX, NY, NZ, NT)
    work = map(i -> gauss_distribution(σ), work)
    PEs = get_PEs(x.f)
    a = LatticeMatrix(work, 4, PEs; nw=1, phases=x.f.phases, comm0=x.f.comm)
    substitute!(x.f, a)
end

function gauss_distribution(σ=1)
    v1 = sqrt(-log(rand() + 1e-10))
    v2 = 2pi * rand()
    xr = v1 * cos(v2)
    xi = v1 * sin(v2)
    return σ * xr + σ * im * xi
end


function clear_fermion!(a::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    clear_matrix!(a.f)
    set_halo!(a.f)
end

struct Shifted_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,shift,Tf} <: WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Shifted_Lattice{Tf,shift}

    function Shifted_WilsonFermion_4D_MPILattice(x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}, shift) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
        #sU = Shifted_Lattice{typeof(U.U),shift}(U.U)
        sx = Shifted_Lattice(x.f, shift)
        shiftin = get_shift(sx)
        return new{NC,NX,NY,NZ,NT,T,AT,NDW,NG,shiftin,typeof(x.f)}(sx)
    end
end

struct Adjoint_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,Tf} <: WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Adjoint_Lattice{Tf}
end

struct Adjoint_Shifted_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,shift,Tf} <: WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    f::Adjoint_Lattice{Shifted_Lattice{Tf,shift}}
end

function Base.adjoint(x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    Adjoint_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,typeof(x.f)}(x.f')
end

function Base.adjoint(x::Shifted_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,shift,Tf}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,shift,Tf}
    Adjoint_Shifted_WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG,shift,Tf}(x.f')
end


function LinearAlgebra.mul!(
    c::WilsonFields_4D_MPILattice,
    a::Fields_4D_MPILattice,
    b::WilsonFields_4D_MPILattice,
)
    #println(typeof(c.f))
    #println(typeof(a.U))
    #println(typeof(b.f))

    mul!(c.f, a.U, b.f)
    #@code_warntype mul!(c.f, a.U, b.f)
end

function LinearAlgebra.mul!(C::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    A::TA) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,TA<:AbstractMatrix}

    mul!(C.f, A)
    #set_halo!(C)
end

function LinearAlgebra.mul!(C::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    A::TA,x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,TA<:AbstractMatrix}

    mul!(C.f, A,x.f)
    #set_halo!(C)
end

function LinearAlgebra.mul!(C::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},A::TA,) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,TA<:AbstractMatrix}

    mul!(C.f, x.f,A)
    #set_halo!(C)
end

#lattice shift
function shift_fermion(F::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}, ν::T1; boundarycondition=nothing) where {T1<:Integer,NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    if boundarycondition != nothing
        @assert F.f.phases ≈ boundarycondition "boundary condition is wrong now the boudnary condition of the fermions is $(F.f.phases) but you want to use $boundarycondition"
    end
    #println(F.f.phases)

    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return shift_fermion(F, shift)
end


function shift_fermion(
    F::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    shift::NTuple{4,T1},
) where {T1<:Integer,NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    return Shifted_WilsonFermion_4D_MPILattice(F, shift)
end

function add_fermion!(
    c::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    α::Number,
    a::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}#c += alpha*a 
    add_matrix!(c.f, a.f, α)
end

function add_fermion!(
    c::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    α::Number,
    a::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    β::Number,
    b::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    add_matrix!(c.f, a.f, b.f, α, β)
end

function set_wing_fermion!(F::WilsonFermion_4D_MPILattice)
    #println("wing")
    #display(F.f.A[:, :, 2, 2, 2, 2])
    #println("wing-------")
    set_halo!(F.f)
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:WilsonFermion_4D_MPILattice}
    @assert boundarycondition ≈ F.f.phases "boundarycondition = $boundarycondition $(F.f.phases)"
    set_halo!(F.f)
end

function LinearAlgebra.dot(
    A::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    B::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    s = dot(A.f, B.f)
    return s
end


function substitute_fermion!(
    A::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    B::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    substitute!(A.f, B.f)
    set_halo!(A.f)

end

function substitute_fermion!(A::WilsonFermion_4D_MPILattice, B::WilsonFermion_4D)
    dim = 4
    PEs = A.f.dims
    phases = A.f.phases
    nw = A.f.nw
    comm0 = A.f.comm
    D = permutedims(B.f, (1, 6, 2, 3, 4, 5))

    tempf = LatticeMatrix(D, dim, PEs;
        nw,
        phases,
        comm0)
    substitute!(A.f, tempf)
    set_halo!(A.f)
end

#Overwrite Y with X*a + Y*b, where a and b are scalars. Return Y.
function LinearAlgebra.axpby!(
    a::Number,
    X::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    b::Number,
    Y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    axpby!(a, X.f, b, Y.f)
    set_halo!(Y.f)
end

function mul_x1plusγμ!(y::WilsonFermion_4D_MPILattice, x, μ)
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

function mul_x1minusγμ!(y::WilsonFermion_4D_MPILattice, x, μ)
    if μ == 1
        mul_1plusγμx!(y, x, 1)
    elseif μ == 2
        mul_1minusγμx!(y, x, 2)
    elseif μ == 3
        mul_1plusγμx!(y, x, 3)
    elseif μ == 4
        mul_1minusγμx!(y, x, 4)
    end
end

function mul_1plusγμx!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x, μ) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    #println("mul")
    #display(x.f.A[:, :, 2, 2, 2, 2])
    substitute!(y.f,x.f)
    set_halo!(y.f)
    mul!(y.f, Oneγμ{:plus,μ}())
    #mul!(y.f, Oneγμ{:plus,μ}(), x.f)
    #error("mul_1plus")
end


function mul_1minusγμx!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x, μ) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    substitute!(y.f,x.f)
    mul!(y.f, Oneγμ{:minus,μ}())
    #mul!(y.f, Oneγμ{:minus,μ}(), x.f)
    #error("mul_1minus")
end

function mul_1plusγ1x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1plusγμx!(y, x, 1)
end

function mul_1plusγ2x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1plusγμx!(y, x, 2)
end

function mul_1plusγ3x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1plusγμx!(y, x, 3)
end

function mul_1plusγ4x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1plusγμx!(y, x, 4)
end

function mul_1plusγ5x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1plusγμx!(y, x, 5)
end

function mul_1minusγ1x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1minusγμx!(y, x, 1)
end

function mul_1minusγ2x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1minusγμx!(y, x, 2)
end

function mul_1minusγ3x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1minusγμx!(y, x, 3)
end

function mul_1minusγ4x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1minusγμx!(y, x, 4)
end

function mul_1minusγ5x!(y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    mul_1minusγμx!(y, x, 5)
end





#C = a*x
function LinearAlgebra.mul!(C::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    a::TA, x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG,TA<:Number}

    mul!(C.f, a, x.f)
    #set_halo!(C)
end


function LinearAlgebra.mul!(
    u::Gaugefields_4D_MPILattice,
    x::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    y::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}, ; clear=true
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    #clear_U!(u)
    if clear
        clear_matrix!(u.U)
    else
        #    println(sum(abs.(u.U)))
    end

    mul!(u.U, x.f, y.f)

end

function LinearAlgebra.mul!(
    y::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x::WilsonFields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    u::Fields_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW}
) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}

    #y[k1, ix, iy, iz, it, ialpha] +=
    #      x[k2, ix, iy, iz, it, ialpha] *
    #      A[k2, k1, ix, iy, iz, it]

    #y[i,a] = x[k,a]*u[k,i]
    #mul!(y.f, x.f, u.U)
    mulT!(y.f, x.f, u.U)

end


function Wdagx_noclover!(xout::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}, U::Array{G,1},
    x::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG}, A, Dim) where {
    G<:AbstractGaugefields,NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    #,temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D,G <: AbstractGaugefields}
    #temp = A._temporary_fermi[4] #temps[4]
    #temp1 = A._temporary_fermi[1] #temps[1]
    #temp2 = A._temporary_fermi[2] #temps[2]
    temp, it_temp = get_temp(A._temporary_fermi)#[4] #temps[4]
    temp1, it_temp1 = get_temp(A._temporary_fermi)#i[1] #temps[1]
    temp2, it_temp2 = get_temp(A._temporary_fermi)#[2] #temps[2]

    #println("MPILattice")

    clear_fermion!(temp)
    #set_wing_fermion!(x)
    for ν = 1:Dim
        Wdagx_noclover_ν!(temp, x, ν, U[ν], A.hopp[ν], A.hopm[ν], temp1, temp2)

        #Wdagx_noclover_ν_p!(temp, x, ν, U[ν], A.hopp[ν], A.hopm[ν])

    end

    clear_fermion!(xout)
    add_fermion!(xout, 1, x, -1, temp)
    set_wing_fermion!(xout, A.boundarycondition)

    unused!(A._temporary_fermi, it_temp)
    unused!(A._temporary_fermi, it_temp1)
    unused!(A._temporary_fermi, it_temp2)

    #display(xout)
    #    exit()
    return
end

function Wdagx_noclover_ν!(temp, x, ν, Uν, Ahoppν, Ahopmν, temp1, temp2)
    xplus = shift_fermion(x, ν)
    mul!(temp1, Uν, xplus)

    #mul!(temp1, view(Arplusγ, :, :, ν))
    mul!(temp1, Oneγμ{:plus,ν}())

    xminus = shift_fermion(x, -ν)
    Uminus = shift_U(Uν, -ν)

    mul!(temp2, Uminus', xminus)
    #fermion_shift!(temp2,U,-ν,x)
    #mul!(temp2,view(x.rminusγ,:,:,ν),temp2)
    #mul!(temp2, view(Arminusγ, :, :, ν))
    mul!(temp2, Oneγμ{:minus,ν}())

    add_fermion!(temp, Ahoppν, temp1, Ahopmν, temp2)
end


function Wdagx_noclover_ν_p!(temp::WilsonFermion_4D_MPILattice{NC,NX,NY,NZ,NT,T,AT,NDW,NG},
    x, ν, Uν, Ahoppν, Ahopmν) where {NC,NX,NY,NZ,NT,T,AT,NDW,NG}
    JACC.parallel_for(
        prod(temp.f.PN), kernel_Wdagx_noclover_ν!, temp.f.A, x.f.A, Val(ν),
        Uν.U.A, Ahoppν, Ahopmν, temp.f.PN, Val(NC), Val(NG), Val(NDW))
end

function kernel_Wdagx_noclover_ν!(i, temp,
    x::AbstractArray{T}, ::Val{ν}, Uν, Ahoppν, Ahopmν, PN,
    ::Val{NC}, ::Val{NG}, ::Val{nw}) where {NC,NG,ν,T,nw}

    ix, iy, iz, it = get_4Dindex(i, PN)
    ix += nw
    iy += nw
    iz += nw
    it += nw
    ixp = ix + ifelse(ν == 1, 1, 0)
    iyp = iy + ifelse(ν == 2, 1, 0)
    izp = iz + ifelse(ν == 3, 1, 0)
    itp = it + ifelse(ν == 4, 1, 0)
    #xplus = shift_fermion(x, ν)
    xplus = MMatrix{NC,NG,T}(undef)
    @inbounds for jc = 1:NG
        for ic = 1:NC
            xplus[ic, jc] = x[ic, jc, ixp, iyp, izp, itp]
        end
    end
    temp1 = MMatrix{NC,NG,T}(undef)
    temp2 = MMatrix{NC,NG,T}(undef)

    # mul!(temp1, Uν, xplus)

    @inbounds for jc = 1:NG
        for ic = 1:NC
            temp1[ic, jc] = zero(eltype(temp))
            for kc = 1:NC
                temp1[ic, jc] += Uν[ic, kc, ix, iy, iz, it] * xplus[kc, jc]
            end
        end
    end

    #mul!(temp1, Oneγμ{:plus,ν}())
    mul!(temp1, Oneγμ{:plus,ν}())

    xminus = MMatrix{NC,NG,T}(undef)
    ixm = ix + ifelse(ν == 1, -1, 0)
    iym = iy + ifelse(ν == 2, -1, 0)
    izm = iz + ifelse(ν == 3, -1, 0)
    itm = it + ifelse(ν == 4, -1, 0)

    @inbounds for jc = 1:NG
        for ic = 1:NC
            xminus[ic, jc] = x[ic, jc, ixm, iym, izm, itm]
        end
    end
    Uminus = MMatrix{NC,NC,T}(undef)
    @inbounds for jc = 1:NC
        for ic = 1:NC
            Uminus[ic, jc] = Uν[ic, jc, ixm, iym, izm, itm]
        end
    end

    @inbounds for jc = 1:NG
        for ic = 1:NC
            temp2[ic, jc] = zero(eltype(temp))
            for kc = 1:NC
                temp2[ic, jc] += Uminus[kc, ic]' * xminus[kc, jc]
            end
        end
    end
    mul!(temp2, Oneγμ{:minus,ν}())

    @inbounds for jc = 1:NG
        for ic = 1:NC
            temp[ic, jc, ix, iy, iz, it] += Ahoppν * temp1[ic, jc] + Ahopmν * temp2[ic, jc]
        end
    end
    #xminus = shift_fermion(x, -ν)
    #Uminus = shift_U(Uν, -ν)

    #mul!(temp2, Uminus', xminus)
    #mul!(temp2, Oneγμ{:minus,ν}())

    #add_fermion!(temp, Ahoppν, temp1, Ahopmν, temp2)
end



function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:plus,1}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 - im * x4
        C[ic, 2] = x2 - im * x3
        C[ic, 3] = x3 + im * x2
        C[ic, 4] = x4 + im * x1
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:minus,1}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 + im * x4
        C[ic, 2] = x2 + im * x3
        C[ic, 3] = x3 - im * x2
        C[ic, 4] = x4 - im * x1
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:plus,2}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 - x4
        C[ic, 2] = x2 + x3
        C[ic, 3] = x3 + x2
        C[ic, 4] = x4 - x1
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:minus,2}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 + x4
        C[ic, 2] = x2 - x3
        C[ic, 3] = x3 - x2
        C[ic, 4] = x4 + x1
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:plus,3}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 - im * x3
        C[ic, 2] = x2 + im * x4
        C[ic, 3] = x3 + im * x1
        C[ic, 4] = x4 - im * x2
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:minus,3}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 + im * x3
        C[ic, 2] = x2 - im * x4
        C[ic, 3] = x3 - im * x1
        C[ic, 4] = x4 + im * x2
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:plus,4}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 - x3
        C[ic, 2] = x2 - x4
        C[ic, 3] = x3 - x1
        C[ic, 4] = x4 - x2
    end
end

function LinearAlgebra.mul!(C::MMatrix{NC,NG,T}, ::Oneγμ{:minus,4}) where {NC,NG,T}
    @inbounds for ic = 1:NC
        x1 = C[ic, 1]
        x2 = C[ic, 2]
        x3 = C[ic, 3]
        x4 = C[ic, 4]
        C[ic, 1] = x1 + x3
        C[ic, 2] = x2 + x4
        C[ic, 3] = x3 + x1
        C[ic, 4] = x4 + x2
    end
end

