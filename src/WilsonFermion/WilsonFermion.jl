struct Wilson_Dirac_operator{Dim,T,fermion} <: Dirac_operator{Dim} 
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    _temporary_fermi::Vector{fermion}

    γ::Array{ComplexF64,3}
    rplusγ::Array{ComplexF64,3}
    rminusγ::Array{ComplexF64,3}
    κ::Float64 #Hopping parameter
    r::Float64 #Wilson term
    hopp::Array{ComplexF64,1}
    hopm::Array{ComplexF64,1}
    eps_CG::Float64
    MaxCGstep::Float64
    verbose::Union{Verbose_1,Verbose_2,Verbose_3}
end

struct DdagD_Wilson_operator <: DdagD_operator 
    dirac::Wilson_Dirac_operator
    function DdagD_Wilson_operator(U::Array{T,1},x,parameters) where  T <: AbstractGaugefields
        return new(Wilson_Dirac_operator(U,x,parameters))
    end

    function DdagD_Wilson_operator(D::Wilson_Dirac_operator{Dim,T,fermion}) where {Dim,T,fermion}
        return new(D)
    end
end


include("./WilsonFermion_4D_wing.jl")


function Wilson_Dirac_operator(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where {NC,Dim}
    xtype = typeof(x)
    num = 6
    _temporary_fermi = Array{xtype,1}(undef,num)

    @assert haskey(parameters,"κ") "parameters should have the keyword κ"
    κ = parameters["κ"]
    boundarycondition = check_parameters(parameters,"boundarycondition",[1,1,1,-1])

    r = check_parameters(parameters,"r",1.0)

    γ,rplusγ,rminusγ = mk_gamma(r)
    hopp = zeros(ComplexF64,4)
    hopm = zeros(ComplexF64,4)
    hopp .= κ
    hopm .= κ

    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    eps_CG = check_parameters(parameters,"eps",default_eps_CG)
    MaxCGstep = check_parameters(parameters,"MaxCGstep",default_MaxCGstep)

    verbose_level = check_parameters(parameters,"verbose_level",2)


    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    if verbose_level == 1 
        verbose = Verbose_1()
    elseif verbose_level == 2
        verbose = Verbose_2()
    elseif verbose_level == 3
        verbose = Verbose_3()
    else
        error("verbose_level = $verbose_level is not supported")
    end 

    return Wilson_Dirac_operator{Dim,eltype(U),xtype}(U,boundarycondition,_temporary_fermi,
        γ,
        rplusγ,
        rminusγ,
        κ,
        r,
        hopp,
        hopm,
        eps_CG,MaxCGstep,verbose
        )
end

function (D::Wilson_Dirac_operator{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Wilson_Dirac_operator{Dim,T,fermion}(U,D.boundarycondition,D._temporary_fermi,
        D.γ,
        D.rplusγ,
        D.rminusγ,
        D.κ,
        D.r,
        D.hopp,
        D.hopm,
        D.eps_CG,D.MaxCGstep,D.verbose
        )
end

struct Adjoint_Wilson_operator{T} <: Adjoint_Dirac_operator
    parent::T
end


function Base.adjoint(A::T) where T <: Wilson_Dirac_operator
    Adjoint_Wilson_operator{typeof(A)}(A)
end

function Initialize_WilsonFermion(u::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    _,_,NN... = size(u)
    return Initialize_WilsonFermion(NC,NN...) 
end

function Initialize_WilsonFermion(NC,NN...) 
    Dim = length(NN)
    if Dim == 4
        fermion = WilsonFermion_4D_wing(NC,NN...)
    else
        error("Dimension $Dim is not supported")
    end
    return fermion
end



function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1 <:AbstractFermionfields,T2 <: Wilson_Dirac_operator, T3 <:AbstractFermionfields}
    Wx!(y,A.U,x,A) 
    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1 <:AbstractFermionfields,T2 <: Adjoint_Wilson_operator, T3 <:  AbstractFermionfields}
    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
    Wdagx!(y,A.parent.U,x,A.parent) 
    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")

    return
end



"""
mk_gamma()
c----------------------------------------------------------------------c
c     Make gamma matrix
c----------------------------------------------------------------------c
C     THE CONVENTION OF THE GAMMA MATRIX HERE
C     ( EUCLIDEAN CHIRAL REPRESENTATION )
C
C               (       -i )              (       -1 )
C     GAMMA1 =  (     -i   )     GAMMA2 = (     +1   )
C               (   +i     )              (   +1     )
C               ( +i       )              ( -1       )
C
C               (     -i   )              (     -1   )
C     GAMMA3 =  (       +i )     GAMMA4 = (       -1 )
C               ( +i       )              ( -1       )
C               (   -i     )              (   -1     )
C
C               ( -1       )
C     GAMMA5 =  (   -1     )
C               (     +1   )
C               (       +1 )
C
C     ( GAMMA_MU, GAMMA_NU ) = 2*DEL_MU,NU   FOR MU,NU=1,2,3,4   
c----------------------------------------------------------------------c
"""
function mk_gamma(r)
    g0 = zeros(ComplexF64,4,4)
    g1 = zero(g0)
    g2 = zero(g1)
    g3 = zero(g1)
    g4 = zero(g1)
    g5 = zero(g1)
    gamma = zeros(ComplexF64,4,4,5)
    rpg = zero(gamma)
    rmg = zero(gamma)


    g0[1,1]=1.0; g0[1,2]=0.0; g0[1,3]=0.0; g0[1,4]=0.0
    g0[2,1]=0.0; g0[2,2]=1.0; g0[2,3]=0.0; g0[2,4]=0.0
    g0[3,1]=0.0; g0[3,2]=0.0; g0[3,3]=1.0; g0[3,4]=0.0
    g0[4,1]=0.0; g0[4,2]=0.0; g0[4,3]=0.0; g0[4,4]=1.0

    g1[1,1]=0.0; g1[1,2]=0.0; g1[1,3]=0.0; g1[1,4]=-im
    g1[2,1]=0.0; g1[2,2]=0.0; g1[2,3]=-im;  g1[2,4]=0.0
    g1[3,1]=0.0; g1[3,2]=+im;  g1[3,3]=0.0; g1[3,4]=0.0
    g1[4,1]=+im;  g1[4,2]=0.0; g1[4,3]=0.0; g1[4,4]=0.0

    g2[1,1]=0.0; g2[1,2]=0.0; g2[1,3]=0.0; g2[1,4]=-1.0
    g2[2,1]=0.0; g2[2,2]=0.0; g2[2,3]=1.0; g2[2,4]=0.0
    g2[3,1]=0.0; g2[3,2]=1.0; g2[3,3]=0.0; g2[3,4]=0.0
    g2[4,1]=-1.0;g2[4,2]=0.0; g2[4,3]=0.0; g2[4,4]=0.0

    g3[1,1]=0.0; g3[1,2]=0.0; g3[1,3]=-im;  g3[1,4]=0.0
    g3[2,1]=0.0; g3[2,2]=0.0; g3[2,3]=0.0; g3[2,4]=+im
    g3[3,1]=+im;  g3[3,2]=0.0; g3[3,3]=0.0; g3[3,4]=0.0
    g3[4,1]=0.0; g3[4,2]=-im;  g3[4,3]=0.0; g3[4,4]=0.0

    g4[1,1]=0.0; g4[1,2]=0.0; g4[1,3]=-1.0;g4[1,4]=0.0
    g4[2,1]=0.0; g4[2,2]=0.0; g4[2,3]=0.0; g4[2,4]=-1.0
    g4[3,1]=-1.0;g4[3,2]=0.0; g4[3,3]=0.0; g4[3,4]=0.0
    g4[4,1]=0.0; g4[4,2]=-1.0;g4[4,3]=0.0; g4[4,4]=0.0

    g5[1,1]=-1.0;g5[1,2]=0.0; g5[1,3]=0.0; g5[1,4]=0.0
    g5[2,1]=0.0; g5[2,2]=-1.0;g5[2,3]=0.0; g5[2,4]=0.0
    g5[3,1]=0.0; g5[3,2]=0.0; g5[3,3]=1.0; g5[3,4]=0.0
    g5[4,1]=0.0; g5[4,2]=0.0; g5[4,3]=0.0; g5[4,4]=1.0

    gamma[:,:,1] = g1[:,:]
    gamma[:,:,2] = g2[:,:]
    gamma[:,:,3] = g3[:,:]
    gamma[:,:,4] = g4[:,:]
    gamma[:,:,5] = g5[:,:]

    for mu=1:4
        for j=1:4
            for i=1:4
                rpg[i,j,mu] = r*g0[i,j] + gamma[i,j,mu]
                rmg[i,j,mu] = r*g0[i,j] - gamma[i,j,mu]
            end
        end
    end 

    return gamma,rpg,rmg


end