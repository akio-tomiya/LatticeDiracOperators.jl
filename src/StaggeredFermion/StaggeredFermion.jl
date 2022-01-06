struct Staggered_Dirac_operator{Dim,T,fermion} <: Dirac_operator{Dim} 
    U::Array{T,1}
    boundarycondition::Vector{Int8}
    mass::Float64
    _temporary_fermi::Vector{fermion}
end

include("./StaggeredFermion_4D_wing.jl")

function Staggered_Dirac_operator(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where {NC,Dim}
    xtype = typeof(x)
    num = 9
    _temporary_fermi = Array{xtype,1}(undef,num)

    @assert haskey(parameters,"mass") "parameters should have the keyword mass"
    mass = parameters["mass"]
    boundarycondition = check_parameters(parameters,"boundarycondition",[1,1,1,-1])
 
    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    return Staggered_Dirac_operator{Dim,eltype(U),xtype}(U,boundarycondition,mass,_temporary_fermi)
end

struct Adjoint_Staggered_operator{T} <: Adjoint_Dirac_operator
    parent::T
end


function Base.adjoint(A::T) where T <: Staggered_Dirac_operator
    Adjoint_Staggered_operator{typeof(A)}(A)
end


function Initialize_StaggeredFermion(u::AbstractGaugefields{NC,Dim}) where {NC,Dim}
    _,_,NN... = size(u)
    return Initialize_StaggeredFermion(NC,NN...) 
end

function Initialize_StaggeredFermion(NC,NN...) 
    Dim = length(NN)
    if Dim == 4
        fermion = StaggeredFermion_4D_wing(NC,NN...)
    else
        error("Dimension $Dim is not supported")
    end
    return fermion
end


function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1,T2 <: Staggered_Dirac_operator, T3}
    
    @assert typeof(A._temporary_fermi[1]) == typeof(x) "type of A._temporary_fermi[1] $(typeof(A._temporary_fermi[1])) should be type of x: $(typeof(x))"
    temps = A._temporary_fermi
    temp = temps[4]
    Dx!(temp,A.U,x,[temps[1],temps[2],temps[3]],A.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y,A.mass,x,1,temp)
    set_wing_fermion!(y,A.boundarycondition)

    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")
end

function LinearAlgebra.mul!(y::T1,A::T2,x::T3) where {T1 <:AbstractFermionfields,T2 <: Adjoint_Staggered_operator, T3 <:  AbstractFermionfields}
    #error("LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))")

    temps = A.parent._temporary_fermi
    temp = temps[4]
    Dx!(temp,A.parent.U,x,[temps[1],temps[2],temps[3]],A.parent.boundarycondition)
    clear_fermion!(y)
    add_fermion!(y,A.parent.mass,x,-1,temp)
    set_wing_fermion!(y,A.parent.boundarycondition)
    #println(y[1,1,1,1,1,1])
    return
end






