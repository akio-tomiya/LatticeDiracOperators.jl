
using Requires

struct D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion} <:
       Dirac_operator{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    wilsonoperator::Union{
        Wilson_Dirac_operator{Dim,T,wilsonfermion},
        Wilson_Dirac_operator_faster{Dim,T,wilsonfermion},
    }
    mass::Float64
    # _temporary_fermi::Array{fermion,1}
    _temporary_fermi::Temporalfields{fermion}# Array{fermion,1}
    L5::Int64
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    # _temporary_fermion_forCG::Vector{fermion}
    _temporary_fermion_forCG::Temporalfields{fermion}# Vector{fermion}
    boundarycondition::Vector{Int8}
    bs::Vector{Float64} #coefficient for GeneralizedDomainwall
    cs::Vector{Float64} #coefficient for GeneralizedDomainwall

end


function D5DW_GeneralizedDomainwall_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
    mass,
    bs,
    cs,
) where {NC,Dim}
    @assert haskey(parameters, "L5") "parameters should have the keyword L5"
    L5 = parameters["L5"]
    if L5 != x.L5
        @warn "L5 in Dirac operator and fermion fields is not same. Now L5 = $L5 and x.L5 = $(x.L5)"
        #@assert L5 == x.L5 "L5 in Dirac operator and fermion fields should be same. Now L5 = $L5 and x.L5 = $(x.L5)"
    end

    if Dim == 4
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])
    elseif Dim == 2
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, -1])
    else
        error("Dim should be 2 or 4!")
    end

    #boundarycondition = check_parameters(parameters,"boundarycondition",[1,1,1,-1])
    T = eltype(U)

    #Parameters for Wilson operator
    r = 1
    M = check_parameters(parameters, "M", -1)
    #@assert haskey(parameters,"M") "parameters should have the keyword M"
    #M = parameters["M"]
    κ_wilson = 1 / (2 * Dim * r + 2M)
    parameters_wilson = Dict()
    parameters_wilson["κ"] = κ_wilson
    parameters_wilson["numtempvec"] = 4
    parameters_wilson["numtempvec_CG"] = 1
    parameters_wilson["boundarycondition"] = boundarycondition
    x_wilson = x.w[1]
    fasterversion = check_parameters(parameters, "faster version", false)
    if fasterversion
        wilsonoperator = Wilson_Dirac_operator_faster(U, x_wilson, parameters_wilson)
        #println("faster version incorporated!")
    else
        wilsonoperator = Wilson_Dirac_operator(U, x_wilson, parameters_wilson)
    end
    #--------------------------------


    xtype = typeof(x)
    num = 4
    # _temporary_fermi = Array{xtype,1}(undef, num)
    _temporary_fermi = Temporalfields(x; num)
    # for i = 1:num
        # _temporary_fermi[i] = similar(x)
    # end

    numcg = 7
    _temporary_fermion_forCG = Temporalfields(x; num=numcg)
    
    # _temporary_fermion_forCG = Array{xtype,1}(undef, numcg)
    # for i = 1:numcg
        # _temporary_fermion_forCG[i] = similar(x)
    # end


    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    #println("eps_CG = ",eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

    verbose_level = check_parameters(parameters, "verbose_level", 2)
    verbose_print = Verbose_print(verbose_level)

    method_CG = check_parameters(parameters, "method_CG", "bicg")
    #println("xtype ",xtype)
    #println("D ", typeof(wilsonoperator))

    return D5DW_GeneralizedDomainwall_operator{Dim,T,xtype,typeof(x_wilson)}(
        U,
        wilsonoperator,
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
        bs,
        cs,
    )
end



function (D::D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion})(
    U,
) where {Dim,T,fermion,wilsonfermion}
    return D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion}(
        U,
        D.wilsonoperator(U),
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
        D.bs,
        D.cs,
    )
end

struct Adjoint_D5DW_GeneralizedDomainwall_operator{T} <: Adjoint_Dirac_operator
    parent::T
end






struct GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion} <:
       Dirac_operator{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    D5DW::D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion}
    D5DW_PV::D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion}
    mass::Float64
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    boundarycondition::Vector{Int8}
    bs::Vector{Float64} #coefficient for GeneralizedDomainwall
    cs::Vector{Float64} #coefficient for GeneralizedDomainwall

end

function GeneralizedDomainwall_Dirac_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    @assert haskey(parameters, "mass") "parameters should have the keyword mass"
    mass = parameters["mass"]

    @assert haskey(parameters, "L5") "parameters should have the keyword L5"
    L5 = parameters["L5"]

    bs = check_parameters(parameters, "bs", fill(1.5,L5))
    cs = check_parameters(parameters, "cs", fill(0.5,L5))

    if length(bs) != L5
        error("Length of bs inconsistent!!")
    elseif length(cs) != L5
        error("Length of cs inconsistent!!")
    end


    # if bs == 1 && cs == 1
    #     println_verbose_level1(U[1], "Shamir kernel (standard DW) is used")
    # elseif bs == 2 && cs == 0
    #     println_verbose_level1(U[1], "Borici/Wilson kernel (truncated overlap) is used")
    # elseif bs == 2 && cs == 1
    #     println_verbose_level1(U[1], "scaled Shamir kernel (Generalized DW) is used")
    # end

    D5DW = D5DW_GeneralizedDomainwall_operator(U, x, parameters, mass, bs, cs)
    D5DW_PV = D5DW_GeneralizedDomainwall_operator(U, x, parameters, 1, bs, cs)

    #boundarycondition = check_parameters(parameters,"boundarycondition",[1,1,1,-1])

    if Dim == 4
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])
    elseif Dim == 2
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, -1])
    else
        error("Dim should be 2 or 4!")
    end

    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    #println("eps_CG = ",eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

    verbose_level = check_parameters(parameters, "verbose_level", 2)
    verbose_print = Verbose_print(verbose_level)

    method_CG = check_parameters(parameters, "method_CG", "bicg")

    return GeneralizedDomainwall_Dirac_operator{Dim,eltype(U),typeof(x),typeof(x.w[1])}(
        U,
        D5DW,
        D5DW_PV,
        mass,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        boundarycondition,
        bs,
        cs,
    )
end


function (D::GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion})(
    U,
) where {Dim,T,fermion,wilsonfermion}
    return GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion}(
        U,
        D.D5DW(U),
        D.D5DW_PV(U),
        D.mass,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D.boundarycondition,
        D.bs,
        D.cs,
    )
end


function get_temporaryvectors(A::T) where {T<:GeneralizedDomainwall_Dirac_operator}
    return A.D5DW._temporary_fermi
end

function get_temporaryvectors_forCG(A::T) where {T<:GeneralizedDomainwall_Dirac_operator}
    return A.D5DW._temporary_fermion_forCG
end

struct Adjoint_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion} <:
       Adjoint_Dirac_operator
    parent::GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion}
end

function Base.adjoint(A::D5DW_GeneralizedDomainwall_operator)
    Adjoint_D5DW_GeneralizedDomainwall_operator(A)
end

function Base.adjoint(
    A::GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion},
) where {Dim,T,fermion,wilsonfermion}
    Adjoint_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion}(A)
end



struct DdagD_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion} <: DdagD_operator
    dirac::Domainwall_Dirac_operator{Dim,T,fermion,wilsonfermion}
    function DdagD_GeneralizedDomainwall_operator(
        U::Array{<:AbstractGaugefields{NC,Dim},1},
        x,
        parameters,
    ) where {NC,Dim}
        return new{Dim,eltype(U),typeof(x),typeof(x.w[1])}(
            GeneralizedDomainwall_Dirac_operator(U, x, parameters),
        )
    end

    function DdagD_GeneralizedDomainwall_operator(
        D::GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion},
    ) where {Dim,T,fermion,wilsonfermion}
        return new{Dim,T,fermion,wilsonfermion}(D)
    end
end

struct GeneralizedD5DWdagD5DW_Wilson_operator{T} <: DdagD_operator
    dirac::T
    function GeneralizedD5DWdagD5DW_Wilson_operator(
        U::Array{T,1},
        x,
        parameters,
        mass,
        bs,
        cs,
    ) where {T<:AbstractGaugefields}
        return new{T}(D5DW_GeneralizedDomainwall_operator(U, x, parameters, mass, bs, cs))
    end

    function GeneralizedD5DWdagD5DW_Wilson_operator(
        D::GeneralizedDomainwall_Dirac_operator{Dim,T,fermion,wilsonfermion},
    ) where {Dim,T,fermion,wilsonfermion}
        dtype = typeof(D.D5DW)
        return new{dtype}(D.D5DW)
    end

    function GeneralizedD5DWdagD5DW_Wilson_operator(
        D::D5DW_GeneralizedDomainwall_operator{Dim,T,fermion,wilsonfermion},
    ) where {Dim,T,fermion,wilsonfermion}
        dtype = typeof(D)
        return new{dtype}(D)
    end
end




function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:GeneralizedD5DWdagD5DW_Wilson_operator,
    T3<:AbstractFermionfields,
}
    #temp = A.dirac._temporary_fermi[1]
    temps = A.dirac._temporary_fermi
    temp, it_temp = get_temp(temps)
    mul!(temp, A.dirac, x)
    mul!(y, A.dirac', temp)
    unused!(temps, it_temp)
    return
end



function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:D5DW_GeneralizedDomainwall_operator,
    T3<:AbstractFermionfields,
}
    temps = A._temporary_fermi
    temp3, it_temp3 = get_temp(temps)
    temp4, it_temp4 = get_temp(temps)
    D5DWx!(
        y,
        A.U,
        x,
        A.mass,
        A.wilsonoperator,
        A.L5,
        A.bs,
        A.cs,
        #A._temporary_fermi[3],
        #A._temporary_fermi[4],
        temp3,
        temp4,
    )
    unused!(temps, it_temp3)
    unused!(temps, it_temp4)
    return
end

function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:Adjoint_D5DW_GeneralizedDomainwall_operator,
    T3<:AbstractFermionfields,
}
    temps = A.parent._temporary_fermi
    temp3, it_temp3 = get_temp(temps)
    temp4, it_temp4 = get_temp(temps)

    D5DWdagx!(
        y,
        A.parent.U,
        x,
        A.parent.mass,
        A.parent.wilsonoperator,
        A.parent.L5,
        A.parent.bs,
        A.parent.cs,
        #A.parent._temporary_fermi[3],
        #A.parent._temporary_fermi[4],
        temp3,
        temp4,
    )
    unused!(temps, it_temp3)
    unused!(temps, it_temp4)
end


function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:Adjoint_GeneralizedDomainwall_operator,
    T3<:AbstractFermionfields,
}

    #A = D5DW(m)*D5DW(m=1)^(-1)
    #=
    A^+ = [D5DW(m)*D5DW(m=1)^(-1)]^+
        = D5DW(m=1)^(-1)^+ D5DW(m)^+
        = [D5DW(m=1)^+]^(-1) D5DW(m)^+
    y = A^+*x = [D5DW(m=1)^+]^(-1) D5DW(m)^+*x
    =#
    temps = A.parent.D5DW_PV._temporary_fermi
    temp1, it_temp1 = get_temp(temps)

    #mul!(A.parent.D5DW_PV._temporary_fermi[1], A.parent.D5DW', x)
    mul!(temp1, A.parent.D5DW', x)

    bicg(
        y,
        A.parent.D5DW_PV',
        #A.parent.D5DW_PV._temporary_fermi[1],
        temp1,
        eps=A.parent.eps_CG,
        maxsteps=A.parent.MaxCGstep,
        verbose=A.parent.verbose_print,
    )
    unused!(temps, it_temp1)
    #maxsteps = 10000)
    #verbose = Verbose_3()) 

    return
end

function LinearAlgebra.mul!(
    y::T1,
    A::T2,
    x::T3,
) where {
    T1<:AbstractFermionfields,
    T2<:GeneralizedDomainwall_Dirac_operator,
    T3<:AbstractFermionfields,
}
    #A = D5DW(m)*D5DW(m=1))^(-1)
    #y = A*x = D5DW(m)*D5DW(m=1))^(-1)*x
    #println("x ",x.w[1][1,1,1,1])

    temps = A.D5DW_PV._temporary_fermi
    temp1, it_temp1 = get_temp(temps)

    bicg(
        temp1,
        #A.D5DW_PV._temporary_fermi[1],
        A.D5DW_PV,
        x,
        eps=A.eps_CG,
        maxsteps=A.MaxCGstep,
        verbose=A.verbose_print,
    )
    #mul!(y, A.D5DW, A.D5DW_PV._temporary_fermi[1])
    mul!(y, A.D5DW, temp1)
    unused!(temps, it_temp1)
    #println("y ",y.w[1][1,1,1,1])

    #bicg(A.D5DW._temporary_fermi[1],A.D5DW,y,
    #eps=A.eps_CG,maxsteps = A.MaxCGstep,verbose = A.verbose_print)
    #mul!(A.D5DW._temporary_fermi[2],A.D5DW_PV,A.D5DW._temporary_fermi[1])
    #println("tmp2 ",A.D5DW._temporary_fermi[2].w[1][1,1,1,1])


    #error("Do not use Domainwall_operator directory. Use D5DW_Domainwall_operator M = D5DW(m)*D5DW(-1)^{-1}")
    #D5DWx!(y,A.U,x,A.m,A.wilsonoperator._temporary_fermi) 
    return
end

include("./GeneralizedDomainwallFermion_5d.jl")




function Initialize_GeneralizedDomainwallFermion(
    u::AbstractGaugefields{NC,Dim},
    L5;
    nowing=false,
) where {NC,Dim}
    _, _, NN... = size(u)
    return Initialize_GeneralizedDomainwallFermion(u,L5, NC, NN..., nowing=nowing)
end


function Initialize_GeneralizedDomainwallFermion(u,L5, NC, NN...; nowing=false)
    Dim = length(NN)
    if Dim == 4
        fermion = GeneralizedDomainwallFermion_5D(u,L5,;nowing=nowing)

        #if nowing
        #    fermion = GeneralizedDomainwallFermion_5D(L5, NC, NN..., nowing=nowing)
        #else
        #    fermion = GeneralizedDomainwallFermion_5D_wing(L5, NC, NN...)
        #end
        #fermion = DomainwallFermion_5D_wing(L5,NC,NN...) 
        #fermion = WilsonFermion_4D_wing{NC}(NN...)
        #fermion = WilsonFermion_4D_wing(NC,NN...)
    elseif Dim == 2
        fermion = GeneralizedDomainwallFermion_3D_wing(L5, NC, NN...)
        #fermion = WilsonFermion_2D_wing{NC}(NN...)
    else
        error("Dimension $Dim is not supported")
    end
    return fermion
end

function bicg(
    x,
    A::GeneralizedDomainwall_Dirac_operator,
    b;
    eps=1e-10,
    maxsteps=1000,
    verbose=Verbose_print(2),
) #A*x = b -> x = A^-1*b
    #A = D5DW(m)*D5DW(m=1))^(-1)
    #A^-1 = D5DW(m=1)*DsDW(m)^-1
    #x = A^-1*b = D5DW(m=1)*DsDW(m)^-1*b
    #println("b ",b.w[1][1,1,1,1])

    temps = A.D5DW._temporary_fermi
    temp1, it_temp1 = get_temp(temps)


    bicg(
        temp1,
        #A.D5DW._temporary_fermi[1],
        A.D5DW,
        b;
        eps=eps,
        maxsteps=maxsteps,
        verbose=verbose,
    )

    #mul!(x,A.D5DW,A.D5DW._temporary_fermi[1])
    #println("x ",x.w[1][1,1,1,1])

    #mul!(x, A.D5DW_PV, A.D5DW._temporary_fermi[1])
    mul!(x, A.D5DW_PV, temp1)
    unused!(temps, it_temp1)
end



function bicg(
    x,
    A::Adjoint_GeneralizedDomainwall_operator,
    b;
    eps=1e-10,
    maxsteps=1000,
    verbose=Verbose_print(2),
)  #A*x = b -> x = A^-1*b
    #A = D5DW(m)*D5DW(m=1)^(-1)
    #A' = (D5DW(m=1)^+)^(-1) D5DW(m)^+
    #A'^-1 = D5DW(m)^+^-1 D5DW(m=1)^+
    #x = A'^-1*b =D5DW(m)^+^-1 D5DW(m=1)^+*b
    temps = A.parent.D5DW._temporary_fermi
    temp, it_temp = get_temp(temps)
    mul!(temp, A.parent.D5DW_PV', b)
    #mul!(A.parent.D5DW._temporary_fermi[1], A.parent.D5DW_PV', b)

    bicg(
        x,
        A.parent.D5DW',
        temp;
        #A.parent.D5DW._temporary_fermi[1];
        eps=eps,
        maxsteps=maxsteps,
        verbose=verbose,
    )
    unused!(temps, it_temp)

end

function bicgstab(
    x,
    A::GeneralizedDomainwall_Dirac_operator,
    b;
    eps=1e-10,
    maxsteps=1000,
    verbose=Verbose_print(2),
) #A*x = b -> x = A^-1*b
    #A = D5DW(m)*D5DW(m=1))^(-1)
    #A^-1 = D5DW(m=1)*DsDW(m)^-1
    #x = A^-1*b = D5DW(m=1)*DsDW(m)^-1*b
    temps = A.parent.D5DW._temporary_fermi
    temp, it_temp = get_temp(temps)

    bicgstab(
        #A.D5DW._temporary_fermi[1],
        temp,
        A.D5DW,
        b;
        eps=eps,
        maxsteps=maxsteps,
        verbose=verbose,
    )
    #mul!(x, A.D5DW_PV, A.D5DW._temporary_fermi[1])
    mul!(x, A.D5DW_PV, temp)

    unused!(temps, it_temp)
end



function bicgstab(
    x,
    A::Adjoint_GeneralizedDomainwall_operator,
    b;
    eps=1e-10,
    maxsteps=1000,
    verbose=Verbose_print(2),
)  #A*x = b -> x = A^-1*b
    #A = D5DW(m)*D5DW(m=1)^(-1)
    #A' = (D5DW(m=1)^+)^(-1) D5DW(m)^+
    #A'^-1 = D5DW(m)^+^-1 D5DW(m=1)^+
    #x = A'^-1*b =D5DW(m)^+^-1 D5DW(m=1)^+*b
    temps = A.parent.D5DW._temporary_fermi
    temp, it_temp = get_temp(temps)


    mul!(temp, A.parent.D5DW_PV', b)
    #mul!(A.parent.D5DW._temporary_fermi[1], A.parent.D5DW_PV', b)

    bicgstab(
        x,
        A.parent.D5DW',
        #A.parent.D5DW._temporary_fermi[1];
        temp;
        eps=eps,
        maxsteps=maxsteps,
        verbose=verbose,
    )
    unused!(temps, it_temp)

end

function cg(
    x,
    A::DdagD_GeneralizedDomainwall_operator,
    b;
    eps=1e-10,
    maxsteps=1000,
    verbose=Verbose_print(2),
)
    #=
    A^-1 = ( (D5DW(m)*D5DW(m=1))^(-1))^+ D5DW(m)*D5DW(m=1))^(-1) )^-1
      = ( D5DW(m=1)^+)^(-1) D5DW(m)^+ D5DW(m)*D5DW(m=1))^(-1) )^-1
      = D5DW(m=1) (  D5DW(m)^+ D5DW(m) )^(-1) )^-1 D5DW(m=1)^+
    x = A^-1*b = D5DW(m=1) (  D5DW(m)^+ D5DW(m) )^(-1)  D5DW(m=1)^+*b
    =#
    temps = A.dirac.D5DW_PV._temporary_fermi
    temp, it_temp = get_temp(temps)
    #mul!(A.dirac.D5DW_PV._temporary_fermi[1], A.dirac.D5DW_PV', b) #D5DW(m=1)^+*b
    mul!(temp, A.dirac.D5DW_PV', b) #D5DW(m=1)^+*b

    temps2 = #A.dirac.D5DW._temporary_fermi
    #temp = A.dirac.D5DW_PV._temporary_fermi[1]
    temp2, it_temp2 = get_temp(temps2)
    cg(
        temp2,
        #A.dirac.D5DW._temporary_fermi[1],
        A.DdagD,
        #temp2;
        temp;
        eps=eps,
        maxsteps=maxsteps,
        verbose=verbose,
    ) #(  D5DW(m)^+ D5DW(m) )^(-1)  D5DW(m=1)^+*b
    #mul!(x, A.dirac.D5DW_PV, A.dirac.D5DW._temporary_fermi[1])
    mul!(x, A.dirac.D5DW_PV, temp2)
    unused!(temps, it_temp)
    unused!(temps2, it_temp2)

end