module Dirac_operators
using MPI
import Gaugefields.Temporalfields_module: Temporalfields, unused!, get_temp

import Gaugefields: get_myrank, get_nprocs

import Gaugefields:
    AbstractGaugefields,
    Abstractfields,
    CovNeuralnet,
    shift_U,
    clear_U!,
    set_wing_U!,
    add_U!,
    AbstractGaugefields_module.Gaugefields_4D_MPILattice
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
    temps = get_temporaryvectors_forCG(D)
    x, it_x = get_temp(temps)
    #x = get_temporaryvectors_forCG(D)[1]
    NN = length(x)
    unused!(temps, it_x)
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

#function get_temporaryvectors(A::T, ith) where {T<:Dirac_operator}
#    return A._temporary_fermion[ith]
#end

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
include("./GeneralizedDomainwallFermion/GeneralizedDomainwallFermion.jl")
include("./action/FermiAction.jl")
#include("./GeneralFermion/generalDiracoperators.jl")

function Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "Dirac_operator") "parameters should have Dirac_operator keyword!"
    if parameters["Dirac_operator"] == "staggered"
        Staggered_Dirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "Wilson"
        if Dim == 4
            fasterversion = check_parameters(parameters, "faster version", true)
        else
            fasterversion = check_parameters(parameters, "faster version", false)
            if fasterversion
                error("Faster version of the Wilson Dirac operator can be used only for Dim=4. Now Dim=$Dim")
            end
        end
        #if fasterversion == false
        #    @warn "now only fasterversion=true is supported. "
        #end

        #improved_gpu = check_parameters(parameters, "improved gpu", false)
        improved_gpu = check_parameters(parameters, "improved gpu", true)
        #@info fasterversion
        if improved_gpu && eltype(U) <: Gaugefields_4D_MPILattice
            Wilson_Dirac_operator_improved(U, x, parameters)
        else
            if fasterversion && Dim == 4
                #println("faster version is used")
                Wilson_Dirac_operator_faster(U, x, parameters)
            else
                #println("faster version is not used")
                Wilson_Dirac_operator(U, x, parameters)
            end
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
    elseif parameters["Dirac_operator"] == "GeneralizedDomainwall"
        GeneralizedDomainwall_Dirac_operator(U, x, parameters)
    elseif parameters["Dirac_operator"] == "GeneralDirac"
        General_Dirac_operator(U, x, parameters)
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
    elseif parameters["Dirac_operator"] == "GeneralDirac"
        DgagD_General_Dirac_operator(U, x, parameters)
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
    #println("print $(A.verbose_print)")
    if A.method_CG == "bicg"
        bicg(y, A, x; eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)#set_verbose(A.verbose_level)) 
        set_wing_fermion!(y, A.boundarycondition)
    elseif A.method_CG == "bicgstab"
        bicgstab(y, A, x; eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)
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
            eps=A.eps_CG,
            maxsteps=A.MaxCGstep,
            verbose=A.verbose_print,
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
            eps=A.parent.eps_CG,
            maxsteps=A.parent.MaxCGstep,
            verbose=A.parent.verbose_print,
        )
    elseif A.parent.method_CG == "bicgstab"
        #println("Adag")
        bicgstab(
            y,
            A,
            x;
            eps=A.parent.eps_CG,
            maxsteps=A.parent.MaxCGstep,
            verbose=A.parent.verbose_print,
        )
    elseif A.parent.method_CG == "preconditiond_bicgstab"
        bicgstab(
            y,
            A,
            x;
            eps=A.parent.eps_CG,
            maxsteps=A.parent.MaxCGstep,
            verbose=A.parent.verbose_print,
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

function get_eps(A::T2) where {T2<:DdagD_operator}
    return A.dirac.eps_CG
end

function get_maxsteps(A::T2) where {T2<:DdagD_operator}
    return A.dirac.MaxCGstep
end

function get_verbose(A::T2) where {T2<:DdagD_operator}
    return A.dirac.verbose_print
end

function get_boundarycondition(A::T2) where {T2<:DdagD_operator}
    return A.dirac.boundarycondition
end

function solve_DinvX!(
    y::T1,
    A::T2,
    x::T3,
) where {T1<:AbstractFermionfields,T2<:DdagD_operator,T3<:AbstractFermionfields}
    eps = get_eps(A)
    maxsteps = get_maxsteps(A)
    verbose = get_verbose(A)

    #cg(
    #    y,
    #    A,
    #    x;
    #    eps=A.dirac.eps_CG,
    #    maxsteps=A.dirac.MaxCGstep,
    #    verbose=A.dirac.verbose_print,
    #)
    cg(
        y,
        A,
        x;
        eps,
        maxsteps,
        verbose,
    )
    boundarycondition = get_boundarycondition(A)
    set_wing_fermion!(y, boundarycondition)
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
    #temp = get_temporaryvectors(A.dirac,5)
    temp, it_temp = get_temp(A.dirac._temporary_fermi)
    #temp = A.dirac._temporary_fermi[5]



    mul!(temp, A.dirac, x)
    set_wing_fermion!(temp)
    #println("dgadg")


    if any(isnan, temp.f)
        #println(temp.f.A)
        error("NaN detected in array temp!")
    end

    #println("temp ", dot(temp, temp))
    mul!(y, A.dirac', temp)
    #set_wing_fermion!(y)
    unused!(A.dirac._temporary_fermi, it_temp)

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

function check_important_parameters(parameters, key, sample=nothing)
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
    temps = get_temporaryvectors_forCG(D)
    temp1, it_temp1 = get_temp(temps)
    temp2, it_temp2 = get_temp(temps)
    #temp1 = get_temporaryvectors_forCG(D)[1]
    #temp2 = get_temporaryvectors_forCG(D)[2]

    for j = 1:NN
        clear_fermion!(temp1)
        temp1[j] = 1
        set_wing_fermion!(temp1)
        mul!(temp2, D, temp1)
        for i = 1:NN
            mat_D[i, j] = temp2[i]
        end
    end
    unused!(temps, it_temp1)
    unused!(temps, it_temp2)
    return mat_D
end


end
