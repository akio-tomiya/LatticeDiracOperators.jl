module Dirac_operators

import Gaugefields:
    AbstractGaugefields,
    Abstractfields,
    CovNeuralnet,
    shift_U,
    clear_U!,
    set_wing_U!,
    add_U!
#import Gaugefields:Verbose_level,Verbose_3,Verbose_2,Verbose_1,println_verbose3
#import Gaugefields:Verbose_level,Verbose_3,Verbose_2,Verbose_1,println_verbose3
using LinearAlgebra
import Gaugefields.Verboseprint_mpi:
    Verbose_print, println_verbose_level1, println_verbose_level2, println_verbose_level3
using SparseArrays



include("./cgmethods.jl")
include("./SakuraiSugiura/SakuraiSugiuramethod.jl")

abstract type Operator#<: AbstractMatrix{ComplexF64}
end

abstract type Dirac_operator{Dim} <: Operator end

abstract type DdagD_operator <: Operator end

abstract type γ5D_operator <: Operator #hermitian matrix
end

struct γ5D{Dirac} <: γ5D_operator
    dirac::Dirac
    function γ5D(D)
        return new{typeof(D)}(D)
    end
end

has_cloverterm(D::Dirac_operator) = false

abstract type Adjoint_Dirac_operator <: Operator end

function Base.size(D::Operator)
    error("Base.size(D::T)  is not implemented in type $(typeof(D))")
end

function Base.size(D::Dirac_operator)
    x = get_temporaryvectors_forCG(D)[1]
    NN = length(x)
    return (NN, NN)
end

function Base.size(D::DdagD_operator)
    return size(D.dirac)
end

function Base.size(D::γ5D_operator)
    return size(D.dirac)
end


function Base.adjoint(A::Dirac_operator{Dim}) where {Dim}
    error("Base.adjoint(A::T)  is not implemented in type $(typeof(A))")
end

Base.adjoint(A::Adjoint_Dirac_operator) = A.parent

Base.adjoint(A::γ5D_operator) = A


function get_temporaryvectors_forCG(A::T) where {T<:Dirac_operator}
    return A._temporary_fermion_forCG
end

function get_temporaryvectors_forCG(A::T) where {T<:Adjoint_Dirac_operator}
    return A.parent._temporary_fermion_forCG
end

function get_temporaryvectors_forCG(A::T) where {T<:γ5D_operator}
    return A.dirac._temporary_fermion_forCG
end

const default_eps_CG = 1e-19
const default_MaxCGstep = 3000


include("./AbstractFermions.jl")
include("./StaggeredFermion/StaggeredFermion.jl")
include("./WilsonFermion/WilsonFermion.jl")
include("./DomainwallFermion/DomainwallFermion.jl")
include("./MobiusDomainwallFermion/MobiusDomainwallFermion.jl")
include("./action/FermiAction.jl")

function Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "Dirac_operator") "parameters should have Dirac_operator keyword!"
    if parameters["Dirac_operator"] == "staggered"
        Staggered_Dirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Wilson"
        fasterversion = check_parameters(parameters, "faster version", false)
        if fasterversion
            Wilson_Dirac_operator_faster(U, x, parameters)
        else
            Wilson_Dirac_operator(U, x, parameters)
        end
    elseif parameters["Dirac_operator"] == "WilsonClover"
        @warn "not implemented completely!!"
        fasterversion = check_parameters(parameters, "faster version", false)
        if fasterversion 
            @warn "The faster version is not supported but now \"faster version\" is true. We ignore it. "
        end
        parameters["hasclover"] = true
        Wilson_Dirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Wilson_general"
        Wilson_GeneralDirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Domainwall"
        Domainwall_Dirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "MobiusDomainwall"
        MobiusDomainwall_Dirac_operator(U, x, parameters)
    else
        error("$(parameters["Dirac_operator"]) is not supported")
    end
end

function DdagD_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "Dirac_operator") "parameters should have Dirac_operator keyword!"
    if parameters["Dirac_operator"] == "staggered"
        DdagD_Staggered_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Wilson"
        DdagD_Wilson_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Domainwall"
        DdagD_Domainwall_operator(U, x, parameters)
    else
        error("$(parameters["Dirac_operator"]) is not supported")
    end
end

function get_temporaryvectors_forCG(A::T) where {T<:DdagD_operator}
    return A.dirac._temporary_fermion_forCG
end

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Operator,T3<:AbstractFermionfields}
    error(
        "solve_DinvX!(y,A,x) (y = A^{-1} x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))",
    )
end

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:γ5D_operator,T3<:AbstractFermionfields}
    x2 = deepcopy(x)
    apply_γ5!(x2)
    solve_DinvX!(y, A.dirac, x2)
end

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Dirac_operator,T3<:AbstractFermionfields}
    if A.method_CG == "bicg"
        bicg(y, A, x; eps = A.eps_CG, maxsteps = A.MaxCGstep, verbose = A.verbose_print)#set_verbose(A.verbose_level)) 
        set_wing_fermion!(y, A.boundarycondition)
    elseif A.method_CG == "bicgstab"
        bicgstab(y, A, x; eps = A.eps_CG, maxsteps = A.MaxCGstep, verbose = A.verbose_print)
        set_wing_fermion!(y, A.boundarycondition)
    elseif A.method_CG == "preconditiond_bicgstab"
        #@assert A.Dirac_operator == "Wilson" "preconditiond_bicgstab is supported only in Wilson Dirac operator"
        WW = Wilson_Dirac_operator_evenodd(A)
        #b = A._temporary_fermi[6]
        #substitute_fermion!(beff,x)
        bout = A._temporary_fermi[7]
        calc_beff!(bout, A.U, x, A)
        iseven = true
        isodd = false

        #bout = x
        #bicgstab(y,WW,bout;eps=A.eps_CG,maxsteps = A.MaxCGstep,verbose = set_verbose(A.verbose_level)) 
        bicgstab_evenodd(
            y,
            WW,
            bout,
            iseven;
            eps = A.eps_CG,
            maxsteps = A.MaxCGstep,
            verbose = A.verbose_print,
        )
        Tx = A._temporary_fermi[6]
        set_wing_fermion!(y, A.boundarycondition, iseven)

        Toex!(Tx, A.U, y, A, iseven)
        #set_wing_fermion!(Tx,A.boundarycondition)
        add_fermion!(y, 1, x, 1, Tx, isodd)
        set_wing_fermion!(y, A.boundarycondition, isodd)

        #Toex!(y,U,x,A,iseven)
        #xo = K Toe xe + b0
    else
        error("A.method_CG = $(A.method_CG) is not supported")
    end


end

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Adjoint_Dirac_operator,T3<:AbstractFermionfields}
    if A.parent.method_CG == "bicg"
        bicg(
            y,
            A,
            x;
            eps = A.parent.eps_CG,
            maxsteps = A.parent.MaxCGstep,
            verbose = A.parent.verbose_print,
        )
    elseif A.parent.method_CG == "bicgstab"
        #println("Adag")
        bicgstab(
            y,
            A,
            x;
            eps = A.parent.eps_CG,
            maxsteps = A.parent.MaxCGstep,
            verbose = A.parent.verbose_print,
        )
    elseif A.parent.method_CG == "preconditiond_bicgstab"
        bicgstab(
            y,
            A,
            x;
            eps = A.parent.eps_CG,
            maxsteps = A.parent.MaxCGstep,
            verbose = A.parent.verbose_print,
        )
        #=
        #@assert A.Dirac_operator == "Wilson" "preconditiond_bicgstab is supported only in Wilson Dirac operator"
        WWdag = Wilson_Dirac_operator_evenodd(A.parent)'
        #b = A._temporary_fermi[6]
        #substitute_fermion!(beff,x)
        bout = A.parent._temporary_fermi[7]
        #calc_beff!(bout,A.parent.U,x,A.parent)
        calc_beff_dag!(bout,A.parent.U,x,A.parent)
        iseven = true

        #bout = x
        #bicgstab(y,WW,bout;eps=A.eps_CG,maxsteps = A.MaxCGstep,verbose = set_verbose(A.verbose_level)) 
        bicgstab_evenodd(y,WW',bout,iseven;eps=A.parent.eps_CG,maxsteps = A.parent.MaxCGstep,verbose = set_verbose(A.parent.verbose_level)) 
        Tx = A.parent._temporary_fermi[6]
        set_wing_fermion!(y,A.parent.boundarycondition)
        Tdagoex!(Tx,A.parent.U,y,A.parent,iseven)
        #set_wing_fermion!(Tx,A.boundarycondition)
        add_fermion!(y,1,x,1,Tx,isodd)
        =#

    else
        error("A.method_CG = $(A.method_CG) is not supported")
    end

    set_wing_fermion!(y, A.parent.boundarycondition)
end

using InteractiveUtils

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:DdagD_operator,T3<:AbstractFermionfields}
    cg(
        y,
        A,
        x;
        eps = A.dirac.eps_CG,
        maxsteps = A.dirac.MaxCGstep,
        verbose = A.dirac.verbose_print,
    )
    set_wing_fermion!(y, A.dirac.boundarycondition)
end



function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:Operator,T3<:AbstractFermionfields}
    error(
        "LinearAlgebra.mul!(y,A,x) is not implemented in type y:$(typeof(y)),A:$(typeof(A)) and x:$(typeof(x))",
    )
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields{NC,Dim},
    A::T,
    x::AbstractFermionfields{NC,Dim},
) where {T<:DdagD_operator,NC,Dim} #y = A*x
    temp = A.dirac._temporary_fermi[5]

    mul!(temp, A.dirac, x)
    mul!(y, A.dirac', temp)

    return
end

function LinearAlgebra.mul!(
    y::AbstractFermionfields{NC,Dim},
    A::T,
    x::AbstractFermionfields{NC,Dim},
) where {T<:γ5D_operator,NC,Dim} #y = A*x
    mul!(y, A.dirac, x)
    apply_γ5!(y)
    return
end




function check_parameters(parameters, key, initial)
    if haskey(parameters, key)
        value = parameters[key]
    else
        value = initial
    end
    return value
end

function check_important_parameters(parameters, key, sample = nothing)
    if sample == nothing
        errstring = ""
    else
        errstring = "sample is $sample"
    end

    @assert haskey(parameters, key) "\"parameters\" should have the keyword $key. $errstring. Now \"parameters\" have $parameters"
    return parameters[key]
end

function set_verbose(verbose_level)
    if verbose_level == 1
        verbose = Verbose_1()
    elseif verbose_level == 2
        verbose = Verbose_2()
    elseif verbose_level == 3
        verbose = Verbose_3()
    else
        error("verbose_level = $verbose_level is not supported")
    end
    return verbose
end

function construct_sparsematrix(D::Operator) # D_ij = e_i D e_j
    NN, _ = size(D)
    mat_D = spzeros(ComplexF64, NN, NN)
    temp1 = get_temporaryvectors_forCG(D)[1]
    temp2 = get_temporaryvectors_forCG(D)[2]

    for j = 1:NN
        clear_fermion!(temp1)
        temp1[j] = 1
        set_wing_fermion!(temp1)
        mul!(temp2, D, temp1)
        for i = 1:NN
            if abs(temp2[i]) > 1e-20
                mat_D[i, j] = temp2[i]
            end
        end
    end
    return mat_D
end

function Base.eltype(A::Dirac_operator)
    return ComplexF64
end

function Base.:*(A::Dirac_operator,x::Vector{T}) where T
    x0 = get_temporaryvectors_forCG(A)[1]
    y0 = similar(x0)
    @assert length(y0) == length(x) "the size mismatch!"
    mul!(y0,A,x0)
    y = zero(x)
    for i = 1:length(y)
        y[i] = y0[i]
    end
    return y
end

function Base.:*(A::γ5D_operator,x::Vector{T}) where T
    x0 = get_temporaryvectors_forCG(A.dirac)[1]
    y0 = similar(x0)
    @assert length(y0) == length(x) "the size mismatch!"
    mul!(y0,A,x0)
    y = zero(x)
    for i = 1:length(y)
        y[i] = y0[i]
    end
    return y
end

end
