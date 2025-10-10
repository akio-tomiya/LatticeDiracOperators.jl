import Gaugefields.Temporalfields_module: Temporalfields, unused!, get_temp

struct Wilson_Dirac_1storder_operator{Dim,T,fermion} <:
       Dirac_operator{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    μ::Int64
    boundarycondition::Vector{Int8}
    _temporary_fermi::Temporalfields{fermion}#Vector{fermion}
end

function Wilson_Dirac_1storder_operator(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    μ,
    x,
    boundarycondition,
) where {NC,Dim}
    xtype = typeof(x)
    num = 4
    _temporary_fermi = Temporalfields(x; num)
    #_temporary_fermi = Array{xtype,1}(undef, num)
    #for i = 1:num
    ##    _temporary_fermi[i] = similar(x)
    #end
    T = eltype(U)

    return Wilson_Dirac_1storder_operator{Dim,T,xtype}(
        U,
        μ,
        boundarycondition,
        _temporary_fermi,
    )
end

struct Wilson_Dirac_operator_faster{Dim,T,fermion} <:
       Wilson_Dirac_operators{Dim} where {T<:AbstractGaugefields}
    U::Array{T,1}
    D::Vector{Wilson_Dirac_1storder_operator{Dim,T,fermion}}
    κ::Float64 #Hopping parameter
    _temporary_fermi::Temporalfields{fermion}#Vector{fermion}
    factor::Float64
    boundarycondition::Vector{Int8}
    eps_CG::Float64
    MaxCGstep::Int64
    verbose_level::Int8
    method_CG::String
    verbose_print::Verbose_print
    _temporary_fermion_forCG::Temporalfields{fermion}#Vector{fermion}

    γ::Array{ComplexF64,3}
    rplusγ::Array{ComplexF64,3}
    rminusγ::Array{ComplexF64,3}
    r::Float64 #Wilson term
    hopp::Array{ComplexF64,1}
    hopm::Array{ComplexF64,1}

end

#=
function Wilson_Dirac_operator_faster(U::Array{<: AbstractGaugefields{NC,Dim},1},x,κ,boundarycondition;factor=1) where {NC,Dim}
    T = eltype(U)
    xtype = typeof(x)
    D = Vector{Wilson_Dirac_1storder_operator{Dim,T,xtype}}(undef,Dim)
    for μ=1:Dim
        D[μ] = Wilson_Dirac_1storder_operator(U,μ,x,boundarycondition)
    end
    num = 5
    _temporary_fermi = Array{xtype,1}(undef,num)
    for i=1:num
        _temporary_fermi[i] = similar(x)
    end

    numcg = 7
    #numcg = 7
    _temporary_fermion_forCG= Array{xtype,1}(undef,numcg)
    for i=1:numcg
        _temporary_fermion_forCG[i] = similar(x)
    end


    eps_CG = default_eps_CG
    #println("eps_CG = ",eps_CG)
    MaxCGstep = default_MaxCGstep

    verbose_level = 2
    verbose_print = Verbose_print(verbose_level)

    method_CG = "bicg"


    for i=1:num
        _temporary_fermi[i] = similar(x)
    end



    return Wilson_Dirac_operator_faster{Dim,T,xtype}(U,D,κ,_temporary_fermi,factor,
        boundarycondition,
        eps_CG,MaxCGstep,verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG)
end
=#

function Wilson_Dirac_operator_faster(
    U::Array{<:AbstractGaugefields{NC,Dim},1},
    x,
    parameters,
) where {NC,Dim}
    T = eltype(U)
    xtype = typeof(x)
    @assert haskey(parameters, "κ") "parameters should have the keyword κ"

    κ = parameters["κ"]


    if Dim == 4
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, 1, 1, -1])
    elseif Dim == 2
        boundarycondition = check_parameters(parameters, "boundarycondition", [1, -1])
    else
        error("Dim should be 2 or 4!")
    end

    D = Vector{Wilson_Dirac_1storder_operator{Dim,T,xtype}}(undef, Dim)
    for μ = 1:Dim
        D[μ] = Wilson_Dirac_1storder_operator(U, μ, x, boundarycondition)
    end

    num = 5
    _temporary_fermi = Temporalfields(x; num)
    #_temporary_fermi = Array{xtype,1}(undef, num)
    #for i = 1:num
    #    _temporary_fermi[i] = similar(x)
    #end


    factor = check_parameters(parameters, "factor", 1)#1/2κ)

    numcg = check_parameters(parameters, "numtempvec_CG", 12)
    #numcg = 7
    _temporary_fermion_forCG = Temporalfields(x; num=numcg)#Array{xtype,1}(undef, numcg)
    #for i = 1:numcg
    #   _temporary_fermion_forCG[i] = similar(x)
    #end


    eps_CG = check_parameters(parameters, "eps_CG", default_eps_CG)
    #println("eps_CG = ",eps_CG)
    MaxCGstep = check_parameters(parameters, "MaxCGstep", default_MaxCGstep)

    verbose_level = check_parameters(parameters, "verbose_level", 2)
    #verbose_print = Verbose_print(verbose_level)

    verbose_print = Verbose_print(verbose_level, myid=get_myrank(x))

    method_CG = check_parameters(parameters, "method_CG", "bicg")


    r = check_parameters(parameters, "r", 1.0)
    @assert r == 1 "In fast Wilson mode, r should be 1. Now r = $r"

    if Dim == 4
        γ, rplusγ, rminusγ = mk_gamma(r)
        hopp = zeros(ComplexF64, 4)
        hopm = zeros(ComplexF64, 4)
        hopp .= κ
        hopm .= κ
    elseif Dim == 2
        γ, rplusγ, rminusγ = mk_sigma(r)
        hopp = zeros(ComplexF64, 2)
        hopm = zeros(ComplexF64, 2)
        hopp .= κ
        hopm .= κ
    end




    return Wilson_Dirac_operator_faster{Dim,T,xtype}(
        U,
        D,
        κ,
        _temporary_fermi,
        factor,
        boundarycondition,
        eps_CG,
        MaxCGstep,
        verbose_level,
        method_CG,
        verbose_print,
        _temporary_fermion_forCG,
        γ,#::Array{ComplexF64,3}
        rplusγ,#::Array{ComplexF64,3}
        rminusγ,#::Array{ComplexF64,3}
        r,#::Float64 #Wilson term
        hopp,#::Array{ComplexF64,1}
        hopm,#::Array{ComplexF64,1})
    )
end


#=
struct DdagD_Wilson_operator_faster{Dim,T,fermion} <: DdagD_operator 
    dirac::Wilson_Dirac_operator_faster{Dim,T,fermion}
    function DdagD_Wilson_operator_faster(U::Array{<: AbstractGaugefields{NC,Dim},1},x,parameters) where  {NC,Dim}
        return new{Dim,eltype(U),typeof(x)}(Wilson_Dirac_operator_faster(U,x,parameters))
    end

    function DdagD_Wilson_operator_faster(D::Wilson_Dirac_operator{Dim,T,fermion}) where {Dim,T,fermion}
        return new{Dim,T,fermion}(D)
    end
end
=#


function (D::Wilson_Dirac_1storder_operator{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Wilson_Dirac_1storder_operator{Dim,T,fermion}(
        U,
        D.μ,
        D.boundarycondition,
        D._temporary_fermi,
    )
end

function (D::Vector{Wilson_Dirac_1storder_operator{Dim,T,fermion}})(U) where {Dim,T,fermion}
    for μ = 1:Dim
        D[μ] = D[μ](U)
    end
    return D
end

function (D::Wilson_Dirac_operator_faster{Dim,T,fermion})(U) where {Dim,T,fermion}
    return Wilson_Dirac_operator_faster{Dim,T,fermion}(
        U,
        D.D(U),
        D.κ,
        D._temporary_fermi,
        D.factor,
        D.boundarycondition,
        D.eps_CG,
        D.MaxCGstep,
        D.verbose_level,
        D.method_CG,
        D.verbose_print,
        D._temporary_fermion_forCG,
        D.γ,#::Array{ComplexF64,3}
        D.rplusγ,#::Array{ComplexF64,3}
        D.rminusγ,#::Array{ComplexF64,3}
        D.r,#::Float64 #Wilson term
        D.hopp,#::Array{ComplexF64,1}
        D.hopm,#::Array{ComplexF64,1})
    )
end

struct Adjoint_Wilson_operator_faster{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:Wilson_Dirac_operator_faster}
    Adjoint_Wilson_operator_faster{typeof(A)}(A)
end

struct Adjoint_Wilson_Dirac_1storder_operator{T} <: Adjoint_Dirac_operator
    parent::T
end

function Base.adjoint(A::T) where {T<:Wilson_Dirac_1storder_operator}
    Adjoint_Wilson_Dirac_1storder_operator{typeof(A)}(A)
end



"""
ψ_n - κ sum_ν U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function LinearAlgebra.mul!(
    y::T1,
    A::Wilson_Dirac_operator_faster{Dim,T,fermion},
    x::T3,
) where {T1<:AbstractFermionfields,T,Dim,fermion,T3<:AbstractFermionfields}
    #=
    println("---------------")
    @time clear_fermion!(y)
    @time add_fermion!(y, A.factor, x)
    for μ = 1:Dim
        println("----")
        temp1, it_temp1 = get_temp(A._temporary_fermi)
        #mul!(A._temporary_fermi[1], A.D[μ], x)
        @time mul!(temp1, A.D[μ], x)



        #add_fermion!(y, -A.factor * A.κ, A._temporary_fermi[1])
        @time add_fermion!(y, -A.factor * A.κ, temp1)
        unused!(A._temporary_fermi, it_temp1)
        println("-----")
    end
    println("---------------")
    =#


    clear_fermion!(y)
    #println(" x, ", dot(x, x))
    add_fermion!(y, A.factor, x)
    set_wing_fermion!(y)
    set_wing_fermion!(x)
    #println(" y, ", dot(y, y))
    for μ = 1:Dim
        #println(" μ = $μ, yfirst ", dot(y, y))
        temp1, it_temp1 = get_temp(A._temporary_fermi)
        #mul!(A._temporary_fermi[1], A.D[μ], x)
        mul!(temp1, A.D[μ], x)
        #println(" μ = $μ, ", dot(temp1, temp1))

        #error("dd")

        #add_fermion!(y, -A.factor * A.κ, A._temporary_fermi[1])
        add_fermion!(y, -A.factor * A.κ, temp1)
        #println(" μ = $μ, yend ", dot(y, y))
        unused!(A._temporary_fermi, it_temp1)
    end
    set_wing_fermion!(y)



end

function LinearAlgebra.mul!(
    y::T1,
    A::Adjoint_Wilson_operator_faster{Wilson_Dirac_operator_faster{Dim,T,fermion}},
    x::T3,
) where {T1<:AbstractFermionfields,T,Dim,fermion,T3<:AbstractFermionfields}
    clear_fermion!(y)

    add_fermion!(y, A.parent.factor, x)
    #println("y")
    set_wing_fermion!(y)
    #println("x")
    set_wing_fermion!(x)

    for μ = 1:Dim
        temp1, it_temp1 = get_temp(A.parent._temporary_fermi)
        mul!(temp1, A.parent.D[μ]', x)


        #mul!(A.parent._temporary_fermi[1], A.parent.D[μ]', x)
        #add_fermion!(y, -A.parent.factor * A.parent.κ, A.parent._temporary_fermi[1])
        add_fermion!(y, -A.parent.factor * A.parent.κ, temp1)
        unused!(A.parent._temporary_fermi, it_temp1)
    end
    set_wing_fermion!(y)

end

function LinearAlgebra.mul!(
    y::T1,
    A::Wilson_Dirac_1storder_operator{Dim,T,fermion},
    x::T3,
) where {T1<:WilsonFields_4D_MPILattice,T,Dim,fermion,T3<:AbstractFermionfields}

    apply_Dirac_1storder_ν!(y, x, A.U, A.μ, A.boundarycondition, A._temporary_fermi)
    return
end

function LinearAlgebra.mul!(
    y::T1,
    A::Wilson_Dirac_1storder_operator{Dim,T,fermion},
    x::T3,
) where {T1<:AbstractFermionfields,T,Dim,fermion,T3<:AbstractFermionfields}

    if A.μ == 1
        apply_Dirac_1storder_1!(y, x, A.U, A.boundarycondition, A._temporary_fermi)
    elseif A.μ == 2
        apply_Dirac_1storder_2!(y, x, A.U, A.boundarycondition, A._temporary_fermi)
    elseif A.μ == 3
        apply_Dirac_1storder_3!(y, x, A.U, A.boundarycondition, A._temporary_fermi)
    elseif A.μ == 4
        apply_Dirac_1storder_4!(y, x, A.U, A.boundarycondition, A._temporary_fermi)
    else
        error("μ = $(A.μ) is not supported!!")
    end
end

function LinearAlgebra.mul!(
    y::T1,
    A::Adjoint_Wilson_Dirac_1storder_operator{
        Wilson_Dirac_1storder_operator{Dim,T,fermion},
    },
    x::T3,
) where {T1<:WilsonFields_4D_MPILattice,T,Dim,fermion,T3<:AbstractFermionfields}

    apply_Dirac_1storder_ν_dagger!(y, x,
        A.parent.U,
        A.parent.μ,
        A.parent.boundarycondition,
        A.parent._temporary_fermi,)
    return
end

function LinearAlgebra.mul!(
    y::T1,
    A::Adjoint_Wilson_Dirac_1storder_operator{
        Wilson_Dirac_1storder_operator{Dim,T,fermion},
    },
    x::T3,
) where {T1<:AbstractFermionfields,T,Dim,fermion,T3<:AbstractFermionfields}


    if A.parent.μ == 1
        apply_Dirac_1storder_1_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )

    elseif A.parent.μ == 2
        apply_Dirac_1storder_2_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    elseif A.parent.μ == 3
        apply_Dirac_1storder_3_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    elseif A.parent.μ == 4
        apply_Dirac_1storder_4_dagger!(
            y,
            x,
            A.parent.U,
            A.parent.boundarycondition,
            A.parent._temporary_fermi,
        )
    else
        error("μ = $(A.parent.μ) is not supported!!")
    end

    #apply_Dirac_1storder_ν_dagger!(y, x,
    ##    A.parent.U,
    #    A.parent.μ,
    #    A.parent.boundarycondition,
    #    A.parent._temporary_fermi,)
    return
end

const γ, rplusγ, rminusγ = mk_gamma(1)

function mul_1minusγνx!(y, ν, temp1)
    mul_1minusγμx!(y, temp1, ν)
    return

    if ν == 1
        mul_1minusγ1x!(y, temp1)
    elseif ν == 2
        mul_1minusγ2x!(y, temp1)
    elseif ν == 3
        mul_1minusγ3x!(y, temp1)
    elseif ν == 4
        mul_1minusγ4x!(y, temp1)
    end
end

function mul_1plusγνx!(y, ν, temp1)
    mul_1plusγμx!(y, temp1, ν)
    return


    if ν == 1
        mul_1plusγ1x!(y, temp1)
    elseif ν == 2
        mul_1plusγ2x!(y, temp1)
    elseif ν == 3
        mul_1plusγ3x!(y, temp1)
    elseif ν == 4
        mul_1plusγ4x!(y, temp1)
    end
end


"""
U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function apply_Dirac_1storder_ν!(y, x, U, ν, boundarycondition, _temporary_fermi)
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    clear_fermion!(y)

    #=
    Ux_ν!(temp1, U[ν], x, ν; boundarycondition)
    mul_1minusγνx!(y, ν, temp1)

    Ux_afterν!(Udagx, U[ν]', x, -ν; boundarycondition)
    mul_1plusγνx!(temp1, ν, Udagx)
    add_fermion!(y, 1, temp1)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)
    return
    =#


    #=
    println(" x   --")
    if typeof(temp1) <: WilsonFermion_4D_nowing
        display(x.f[:, 1, 1, 1, 1, :])
        display(U[ν].U[:, :, 1, 1, 1, 1])
    else
        #println(typeof(temp1))
        display(x.f.A[:, :, 2, 2, 2, 2])
        display(U[ν].U.A[:, :, 2, 2, 2, 2])
    end
    =#

    mul!(Udagx, U[ν]', x)


    set_wing_fermion!(Udagx)
    #println("Udagx ", dot(Udagx, Udagx))
    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)


    #println("temp1 ", dot(temp1, temp1))
    mul_1minusγνx!(y, ν, temp1)
    #println("y ", dot(y, y))

    #=
    println(" Udagx   --")
    if typeof(temp1) <: WilsonFermion_4D_nowing
        display(Udagx.f[:, 1, 1, 1, 1, :])
    else
        #println(typeof(temp1))
        display(Udagx.f.A[:, :, 2, 2, 2, 2])
    end
    =#

    xminus = shift_fermion(Udagx, -ν)
    set_wing_U!(U[ν])

    Uminus = shift_U(U[ν], -ν)

    #mul!(Udagx, Uminus', xminus)
    #=
    println(" xminus   --")
    if typeof(temp1) <: WilsonFermion_4D_nowing
        display(xminus.parent.f[:, 4, 1, 1, 1, :])
    else
        #println(typeof(temp1))
        display(xminus.f.data.A[:, :, 1, 2, 2, 2])
    end
    =#

    mul!(temp1, Uminus', xminus)

    #=
    println("temp1   --")
    if typeof(temp1) <: WilsonFermion_4D_nowing
        display(temp1.f[:, 1, 1, 1, 1, :])
    else
        #println(typeof(temp1))
        display(temp1.f.A[:, :, 2, 2, 2, 2])
    end
    println("Udagx2 ", dot(temp1, temp1))
    =#

    #mul!(Udagx, Uminus', xminus)
    #mul_1plusγνx!(temp1, ν, Udagx)
    #add_fermion!(y, 1, temp1)

    mul_1plusγνx!(Udagx, ν, temp1)
    #
    #println("Udagx3 ", dot(Udagx, Udagx))
    #println("yy ", dot(y, y))
    add_fermion!(y, 1, Udagx)
    #println("yy ", dot(y, y))

    set_wing_fermion!(y)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    return




end

"""
U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function apply_Dirac_1storder_1!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 1
    #temp1 = _temporary_fermi[1]
    temp1, it_temp1 = get_temp(_temporary_fermi)

    #Udagx = _temporary_fermi[2]
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)


    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)

    mul_1minusγ1x!(y, temp1)

    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1plusγ1x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_2!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 2
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    mul_1minusγ2x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1plusγ2x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_3!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 3
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    mul_1minusγ3x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1plusγ3x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end


function apply_Dirac_1storder_4!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 4
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    mul_1minusγ4x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1plusγ4x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end


#=

"""
U_n[ν](1 - γν)*ψ_{n+ν} + U_{n-ν}[-ν]^+ (1 + γν)*ψ_{n-ν}
"""
function apply_Dirac_1storder_1!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 1
    #temp1 = _temporary_fermi[1]
    temp1, it_temp1 = get_temp(_temporary_fermi)

    #Udagx = _temporary_fermi[2]
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    #mul!(Udagx, U[ν]', x)
    #set_wing_fermion!(Udagx)



    Ux_ν!(temp1, U[ν], x, ν)

    #xplus = shift_fermion(x, ν)
    #mul!(temp1, U[ν], xplus)

    mul_1minusγ1x!(y, temp1)

    #mul!(y,view(rminusγ,:,:,ν),temp1)

    Ux_afterν!(Udagx, U[ν]', x, -ν)
    mul_1plusγ1x!(temp1, Udagx)
    add_fermion!(y, 1, temp1)

    #xminus = shift_fermion(Udagx, -ν)


    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    #mul_1plusγ1x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    #add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_2!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 2
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    #mul!(Udagx, U[ν]', x)
    #set_wing_fermion!(Udagx)

    #xplus = shift_fermion(x, ν)
    #mul!(temp1, U[ν], xplus)

    Ux_ν!(temp1, U[ν], x, ν)
    mul_1minusγ2x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    #xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    #mul_1plusγ2x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    #add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    Ux_afterν!(Udagx, U[ν]', x, -ν)
    mul_1plusγ2x!(temp1, Udagx)
    add_fermion!(y, 1, temp1)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_3!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 3
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    #mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    #xplus = shift_fermion(x, ν)
    #mul!(temp1, U[ν], xplus)
    Ux_ν!(temp1, U[ν], x, ν)
    mul_1minusγ3x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)


    Ux_afterν!(Udagx, U[ν]', x, -ν)
    mul_1plusγ3x!(temp1, Udagx)
    add_fermion!(y, 1, temp1)

    #mul_1plusγ3x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    #add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end


function apply_Dirac_1storder_4!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 4
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    #xplus = shift_fermion(x, ν)
    #mul!(temp1, U[ν], xplus)
    Ux_ν!(temp1, U[ν], x, ν)
    mul_1minusγ4x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1plusγ4x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_ν_dagger!(y, x, U, ν, boundarycondition, _temporary_fermi)
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)

    #=
    println("------")
    @time clear_fermion!(y)
    @time Ux_ν!(temp1, U[ν], x, ν; boundarycondition)
    @time mul_1plusγνx!(y, ν, temp1)
    @time Ux_afterν!(Udagx, U[ν]', x, -ν; boundarycondition)
    @time mul_1minusγνx!(temp1, ν, Udagx)
    @time add_fermion!(y, 1, temp1)
    println("------")
    =#


    #=
    clear_fermion!(y)
    Ux_ν!(temp1, U[ν], x, ν; boundarycondition)
    mul_1plusγνx!(y, ν, temp1)
    Ux_afterν!(Udagx, U[ν]', x, -ν; boundarycondition)
    mul_1minusγνx!(temp1, ν, Udagx)
    add_fermion!(y, 1, temp1)
    =#
    #set_wing_fermion!(x)



    #println("ν  = $ν ----------------------")
    #println("x ", dot(x, x))
    xplus = shift_fermion(x, ν; boundarycondition)
    #mul!(temp1, U[ν], x) #debuh
    mul!(temp1, U[ν], xplus)
    set_wing_fermion!(temp1) #debug

    #println("temp1 ", dot(temp1, temp1))

    mul_1plusγνx!(y, ν, temp1)
    #println("y ", dot(y, y))

    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)
    #println("Udagx ", dot(Udagx, Udagx))


    xminus = shift_fermion(Udagx, -ν)
    mul_1minusγνx!(temp1, ν, xminus)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y)
    #error("d")

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

end

=#

function apply_Dirac_1storder_ν_dagger!(y, x, U, ν, boundarycondition, _temporary_fermi)
    if ν == 1
        apply_Dirac_1storder_1_dagger!(y, x, U, boundarycondition, _temporary_fermi)
    elseif ν == 2
        apply_Dirac_1storder_2_dagger!(y, x, U, boundarycondition, _temporary_fermi)
    elseif ν == 3
        apply_Dirac_1storder_3_dagger!(y, x, U, boundarycondition, _temporary_fermi)
    elseif ν == 4
        apply_Dirac_1storder_4_dagger!(y, x, U, boundarycondition, _temporary_fermi)
    else
        error("μ = $(ν) is not supported!!")
    end
end

function apply_Dirac_1storder_1_dagger!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 1
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)


    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)


    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    #Ux_ν!(temp1, U[ν], x, ν)
    mul_1plusγ1x!(y, temp1)

    #mul!(y,view(rminusγ,:,:,ν),temp1)



    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    xminus = shift_fermion(Udagx, -ν)
    mul_1minusγ1x!(temp1, xminus)

    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)


    #Ux_afterν!(Udagx, U[ν]', x, -ν)
    #set_wing_fermion!(Udagx)
    #mul_1minusγ1x!(temp1, Udagx)
    #add_fermion!(y, 1, temp1)




    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)


    #display(xout)
    #    exit()
    return
end

function apply_Dirac_1storder_2_dagger!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 2
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]
    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    #Ux_ν!(temp1, U[ν], x, ν)
    mul_1plusγ2x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1minusγ2x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    #display(xout)
    #    exit()


    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    return
end

function apply_Dirac_1storder_3_dagger!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 3
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]

    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    #Ux_ν!(temp1, U[ν], x, ν)
    mul_1plusγ3x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1minusγ3x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    #display(xout)
    #    exit()
    return
end


function apply_Dirac_1storder_4_dagger!(y, x, U, boundarycondition, _temporary_fermi)

    ν = 4
    #temp1 = _temporary_fermi[1]
    #Udagx = _temporary_fermi[2]

    temp1, it_temp1 = get_temp(_temporary_fermi)
    Udagx, it_Udagx = get_temp(_temporary_fermi)
    #Ux = _temporary_fermi[3]
    clear_fermion!(y)
    mul!(Udagx, U[ν]', x)
    set_wing_fermion!(Udagx)

    xplus = shift_fermion(x, ν)
    mul!(temp1, U[ν], xplus)
    #Ux_ν!(temp1, U[ν], x, ν)
    mul_1plusγ4x!(y, temp1)
    #mul!(y,view(rminusγ,:,:,ν),temp1)

    xminus = shift_fermion(Udagx, -ν)
    #xminus = shift_fermion(x,-ν)
    #Uminus = shift_U(U[ν],-ν)
    #mul!(temp1,Uminus',xminus)

    mul_1minusγ4x!(temp1, xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),xminus)
    #mul!(temp1,view(rplusγ,:,:,ν),temp1)
    add_fermion!(y, 1, temp1)
    set_wing_fermion!(y, boundarycondition)

    unused!(_temporary_fermi, it_temp1)
    unused!(_temporary_fermi, it_Udagx)

    #display(xout)
    #    exit()
    return
end

function D4x!(
    xout::T1,
    U::Array{G,1},
    x::T2,
    A::T,
    Dim,
) where {T1,T2,G<:AbstractGaugefields,T<:Wilson_Dirac_operator_faster}
    temps = A._temporary_fermi
    temp1, it_temp1 = get_temp(temps)

    clear_fermion!(xout)
    for μ = 1:Dim
        #mul!(A._temporary_fermi[1], A.D[μ], x)
        #add_fermion!(xout, 0.5, A._temporary_fermi[1])
        mul!(temp1, A.D[μ], x)
        add_fermion!(xout, 0.5, temp1)
    end
    set_wing_fermion!(xout)
    unused!(temps, it_temp1)
end

function D4dagx!(
    xout::T1,
    U::Array{G,1},
    x::T2,
    A::T,
    Dim,
) where {T1,T2,G<:AbstractGaugefields,T<:Wilson_Dirac_operator_faster}
    temps = A._temporary_fermi
    temp1, it_temp1 = get_temp(temps)
    clear_fermion!(xout)
    for μ = 1:Dim
        #mul!(A._temporary_fermi[1], A.D[μ]', x)
        #add_fermion!(xout, 0.5, A._temporary_fermi[1])
        mul!(temp1, A.D[μ]', x)
        add_fermion!(xout, 0.5, temp1)
    end
    set_wing_fermion!(xout)
    unused!(temps, it_temp1)
end

