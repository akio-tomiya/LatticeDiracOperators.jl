import Gaugefields:Traceless_antihermitian_add!

struct Wilsonclover_data 
end

abstract type Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} <: FermiAction{Dim,Dirac,fermion,gauge}
end

struct WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm} <: Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} #FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    hascloverterm::Bool
    clover_data::Union{Nothing,Wilsonclover_data}
    _temporary_fermionfields::Vector{fermion}
    _temporary_gaugefields::Vector{gauge}

    function WilsonFermiAction(D::Dirac_operator{Dim},hascovnet,covneuralnet,parameters_action) where Dim
        hascloverterm = check_parameters(parameters_action,"hascloverterm",false)
        if hascloverterm
            error("not implemented yet")
        else
            clover_data = nothing
        end

        num = 6

        x = D._temporary_fermi[1]
        xtype = typeof(x)
        _temporary_fermionfields = Array{xtype,1}(undef,num)
        for i=1:num
            _temporary_fermionfields[i] = similar(x)
        end

        Utemp = D.U[1]
        Utype = typeof(Utemp)
        numU = 2
        _temporary_gaugefields = Array{Utype,1}(undef,numU)
        for i=1:numU
            _temporary_gaugefields[i] = similar(Utemp)
        end


        return new{Dim,typeof(D),xtype,Utype,hascloverterm}(hascovnet,covneuralnet,D,
                            hascloverterm,
                            clover_data,
                        _temporary_fermionfields,
                        _temporary_gaugefields)

    end
end

function evaluate_FermiAction(fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},U,ϕ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge,hascloverterm} 
    W = fermi_action.diracoperator(U)
    η = fermi_action._temporary_fermionfields[1]
    solve_DinvX!(η,W',ϕ)
    Sf = dot(η,η)
    return real(Sf)
end

function calc_UdSfdU!(UdSfdU::Vector{<: AbstractGaugefields},fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} ,U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields) where  {Dim,Dirac,fermion,gauge,hascloverterm} 
    #println("------dd")
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Wilson_operator(W)
    X = fermi_action._temporary_fermionfields[end]
    Y = fermi_action._temporary_fermionfields[2]
    #X = (D^dag D)^(-1) ϕ 
    #
    #println("Xd ",X[1,1,1,1,1,1])
    solve_DinvX!(X,WdagW,ϕ)
    #println("X ",X[1,1,1,1,1,1])
    clear_U!(UdSfdU)

    calc_UdSfdU_fromX!(UdSfdU,Y,fermi_action,U,X) 
    #println("----aa--")
    set_wing_U!(UdSfdU)
end

#function calc_UdSfdU_fromX!(UdSfdU::Vector{<: AbstractGaugefields},Y,fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm} ,U,X;coeff = 1) where {Dim,Dirac,fermion,gauge,hascloverterm}
function calc_UdSfdU_fromX!(UdSfdU::Vector{<: AbstractGaugefields},Y,fermi_action,U,X;coeff = 1) where {Dim}
    W = fermi_action.diracoperator(U)
    mul!(Y,W,X)
    #set_wing_fermion!(Y)

    temp0_f = fermi_action._temporary_fermionfields[3]
    temp1_f = fermi_action._temporary_fermionfields[4]
    temp0_g = fermi_action._temporary_gaugefields[1]

    for μ=1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X,μ)

        #@time mul!(temp0_f,U[μ],X)

        mul!(temp0_f,U[μ],Xplus)
        
        
        # (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp1_f,view(W.rminusγ,:,:,μ),temp0_f)
        
        # κ (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp0_f,W.hopp[μ],temp1_f)

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        mul!(temp0_g,temp0_f,Y') 

        add_U!(UdSfdU[μ],-coeff,temp0_g)

        #!  Construct P2*U_adj(x,mu)
        # Y_{k+μ}^dag U_{k,μ}^dag
        Yplus = shift_fermion(Y,μ)
        mul!(temp0_f,Yplus',U[μ]')

        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp1_f,temp0_f,view(W.rplusγ,:,:,μ))

        # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f,W.hopm[μ],temp1_f)

        # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_g,X,temp0_f) 

        add_U!(UdSfdU[μ],coeff,temp0_g)


    end


end


function calc_p_UdSfdU!(p,fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm} ,U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields,coeff = 1) where  {Dim,Dirac,fermion,gauge,hascloverterm} 
    #println("------dd")
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Wilson_operator(W)
    X = fermi_action._temporary_fermionfields[end]
    Y = fermi_action._temporary_fermionfields[2]
    #X = (D^dag D)^(-1) ϕ 
    #
    #println("Xd ",X[1,1,1,1,1,1])
    solve_DinvX!(X,WdagW,ϕ)
    #println("X ",X[1,1,1,1,1,1])
    #clear_U!(UdSfdU)

    calc_p_UdSfdU_fromX!(p,Y,fermi_action,U,X,coeff= coeff) 
    #println("----aa--")
    #set_wing_U!(UdSfdU)
end

function calc_p_UdSfdU_fromX!(p,Y,fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm} ,U,X;coeff = 1) where {Dim,Dirac,fermion,gauge,hascloverterm}
    W = fermi_action.diracoperator(U)
    mul!(Y,W,X)
    #set_wing_fermion!(Y)

    temp0_f = fermi_action._temporary_fermionfields[3]
    temp1_f = fermi_action._temporary_fermionfields[4]
    temp0_g = fermi_action._temporary_gaugefields[1]

    for μ=1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X,μ)

        #@time mul!(temp0_f,U[μ],X)

        mul!(temp0_f,U[μ],Xplus)
        
        
        # (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp1_f,view(W.rminusγ,:,:,μ),temp0_f)
        
        # κ (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp0_f,W.hopp[μ],temp1_f)

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        mul!(temp0_g,temp0_f,Y') 

        Traceless_antihermitian_add!(p[μ],-coeff,temp0_g)

        #add_U!(UdSfdU[μ],-coeff,temp0_g)

        #!  Construct P2*U_adj(x,mu)
        # Y_{k+μ}^dag U_{k,μ}^dag
        Yplus = shift_fermion(Y,μ)
        mul!(temp0_f,Yplus',U[μ]')

        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp1_f,temp0_f,view(W.rplusγ,:,:,μ))

        # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f,W.hopm[μ],temp1_f)

        # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_g,X,temp0_f) 

        Traceless_antihermitian_add!(p[μ],coeff,temp0_g)

        #add_U!(UdSfdU[μ],coeff,temp0_g)


    end


end


function gauss_sampling_in_action!(η::AbstractFermionfields,U,fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} ) where {Dim,Dirac,fermion,gauge,hascloverterm} 
    #gauss_distribution_fermion!(η)
    gauss_distribution_fermion!(η,rand)
end

using InteractiveUtils

function sample_pseudofermions!(ϕ::AbstractFermionfields,U,fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} ,ξ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge,hascloverterm} 
    W = fermi_action.diracoperator(U)
    mul!(ϕ,W',ξ)
    set_wing_fermion!(ϕ)
end
