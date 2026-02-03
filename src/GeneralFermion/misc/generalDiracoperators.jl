import LatticeMatrices: DiracOp
using PreallocatedArrays

#=

struct General_Dirac_operator{Dim,TG,TF,Dmul,Ddagmul,P} <: Dirac_operator{Dim}
    diracop::DiracOp{TG,TF,Dmul,Ddagmul,P}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    boundarycondition::Vector{Int8}
    _temporary_fermion_forCG::PreallocatedArray{TF}

    function General_Dirac_operator(
        U::Vector{<:AbstractGaugefields{NC,Dim}},
        phi,
        parameters,
    ) where {NC,Dim}

        @assert eltype(U) <: Gaugefields_4D_MPILattice "type of U should be Vector{Gaugefields_4D_MPILattice} now $(typeof(U))"
        numtemp = check_parameters(parameters, "numtemp", 4)
        numphitemp = check_parameters(parameters, "numphitemp", 4)
        @assert haskey(parameters, "apply_D") "parameters should have the keyword apply_D"
        apply = parameters["apply_D"]
        @assert haskey(parameters, "apply_Ddag") "parameters should have the keyword apply_Ddag"
        apply_dag = parameters["apply_Ddag"]
        if haskey(parameters, "parameters")
            params = parameters["parameters"]
        else
            params = ()
        end

        eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
        #println("eps_CG = ",eps_CG)
        MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

        verbose_level = check_parameters(parameters, "verbose_level", 2)
        verbose_print = Verbose_print(verbose_level, myid=get_myrank(phi))

        method_CG = check_parameters(parameters, "method_CG", "bicg")

        numcg = check_parameters(parameters, "numtempvec_CG", 12)
        #numcg = 7
        _temporary_fermion_forCG = PreallocatedArray(phi; num=numcg)

        println(phi.f.phases)
        boundarycondition = zeros(Int8, Dim)
        for i = 1:Dim
            boundarycondition[i] = phi.f.phases[i]
        end


        return General_Dirac_operator(U, apply, apply_dag, params, phi, eps_CG,
            MaxCGstep,#::Int64
            verbose_level,#::Int8
            method_CG, verbose_print, boundarycondition, _temporary_fermion_forCG; numtemp, numphitemp)
    end

    function General_Dirac_operator(U, apply, apply_dag, params, phi, eps_CG,
        MaxCGstep,#::Int64
        verbose_level,#::Int8
        method_CG, verbose_print, boundarycondition, _temporary_fermion_forCG;
        numtemp=4, numphitemp=4)

        diracop = DiracOp(U, apply, apply_dag, params, phi; numtemp, numphitemp)

        Dim = length(U)
        TG = eltype(U)
        Dmul = typeof(apply)
        Ddagmul = typeof(apply_dag)
        TF = typeof(phi)
        P = typeof(params)

        return new{Dim,TG,TF,Dmul,Ddagmul,P}(diracop,
            eps_CG,
            MaxCGstep,#::Int64
            verbose_level,#::Int8
            method_CG,
            verbose_print, boundarycondition, _temporary_fermion_forCG)#::String)
    end
end
export General_Dirac_operator

function LinearAlgebra.mul!(y::T1, GD::General_Dirac_operator{Dim,TG,TF,Dmul,Ddagmul,P}, x::T2) where {Dim,TG,TF,Dmul,Ddagmul,P,T1<:AbstractFermionfields,T2<:AbstractFermionfields}
    D = GD.diracop
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)
    D.apply(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end

struct Adjoint_General_Dirac_operator{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:General_Dirac_operator}
    Adjoint_General_Dirac_operator{typeof(A)}(A)
end


function LinearAlgebra.mul!(y::T1, A::Adjoint_General_Dirac_operator, x::T2) where {T1<:AbstractFermionfields,T2<:AbstractFermionfields}
    D = A.parent.diracop
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)

    D.apply_dag(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)

    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end


struct DgagD_General_Dirac_operator{T} <: DdagD_operator
    dirac::T
    function DgagD_General_Dirac_operator(d::T) where {T<:General_Dirac_operator}
        return new{T}(d)
    end

    function DgagD_General_Dirac_operator(
        U::Vector{<:AbstractGaugefields{NC,Dim}},
        phi,
        parameters
    ) where {NC,Dim}
        D = General_Dirac_operator(U, phi, parameters)
        return new{typeof(D)}(D)
    end

end

function LinearAlgebra.mul!(y::T1, A::T, x::T2) where {T<:DgagD_General_Dirac_operator,T1<:AbstractFermionfields,T2<:AbstractFermionfields}
    D = A.dirac.diracop
    phitemp1, itphitemp1 = get_block(D.phitemps)
    temp, ittemp = get_block(D.temps, 4)
    phitemp, itphitemp = get_block(D.phitemps, 4)

    D.apply(phitemp1, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp, temp)
    set_wing_fermion!(phitemp1)
    D.apply_dag(y, D.U[1], D.U[2], D.U[3], D.U[4], phitemp1, D.p, phitemp, temp)
    set_wing_fermion!(y)

    #DdagDmul!(y, D.U[1], D.U[2], D.U[3], D.U[4], x, D.p, phitemp1, temp, phitemp)
    unused!(D.phitemps, itphitemp1)
    unused!(D.temps, ittemp)
    unused!(D.phitemps, itphitemp)
end
=#


