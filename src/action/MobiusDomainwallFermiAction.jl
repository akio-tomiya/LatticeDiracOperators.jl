import Gaugefields: Traceless_antihermitian_add!



struct MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge} <:
       FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    #_temporary_fermionfields::Vector{fermion}
    #_temporary_gaugefields::Vector{gauge}
    _temporary_fermionfields::Temporalfields{fermion}#Vector{fermion}
    _temporary_gaugefields::Temporalfields{gauge}#Vector{gauge}

    function MobiusDomainwallFermiAction(
        D::Dirac_operator{Dim},
        hascovnet,
        covneuralnet,
    ) where {Dim}
        num = 10
        temps = get_temporaryvectors(D)
        x = temps[1]
        xtype = typeof(x)
        _temporary_fermionfields = Temporalfields(x; num)
        #_temporary_fermionfields = Array{xtype,1}(undef, num)
        #for i = 1:num
        #    _temporary_fermionfields[i] = similar(x)
        #end

        Utemp = D.U[1]
        Utype = typeof(Utemp)
        numU = 2
        _temporary_gaugefields = Temporalfields(Utemp; num=numU)
        #_temporary_gaugefields = Array{Utype,1}(undef, numU)
        #for i = 1:numU
        #    _temporary_gaugefields[i] = similar(Utemp)
        #end


        return new{Dim,typeof(D),xtype,Utype}(
            hascovnet,
            covneuralnet,
            D,
            _temporary_fermionfields,
            _temporary_gaugefields,
        )

    end
end

function evaluate_FermiAction(
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    temps = fermi_action._temporary_fermionfields
    η, it_η = get_temp(temps)
    #η = fermi_action._temporary_fermionfields[1]
    solve_DinvX!(η, W', ϕ)
    Sf = dot(η, η)
    unused!(temps, it_η)

    return real(Sf)
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    #println("------dd")
    QD5DW = fermi_action.diracoperator.D5DW(U)
    Q = MobiusD5DWdagD5DW_Wilson_operator(QD5DW)
    D5_PV = fermi_action.diracoperator.D5DW_PV(U)

    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[1]
    #temps_dw = fermi_action._temporary_fermionfields[1]

    X0, it_X0 = get_temp(temps)
    Y, it_Y = get_temp(temps)

    #X0 = fermi_action._temporary_fermionfields[6]
    #Y = fermi_action._temporary_fermionfields[5]


    mul!(temps_dw, D5_PV', ϕ) #temps_dw = D5_PV'*ϕ

    solve_DinvX!(X0, Q, temps_dw) #X0 = Q^-1 D5_PV'*ϕ
    #set_wing_fermion!(X)
    set_wing_fermion!(ϕ)

    clear_U!(UdSfdU)
    #println(Y[1][1, 1, 1, 1, 1, 1])
    calc_UdSfdU_fromX!(UdSfdU, Y, ϕ, fermi_action, U, X0)

    #println("----aa--")
    set_wing_U!(UdSfdU)

    unused!(temps, it_temps_dw)
    unused!(temps, it_X0)
    unused!(temps, it_Y)
end

function calc_UdSfdU_fromX!(
    UdSfdU::Vector{<:AbstractGaugefields},
    Y,
    ϕ,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    X0;
    coeff=1,
) where {Dim,Dirac,fermion,gauge}
    return _calc_UdSfdU_fromX_MPILattice!(
        UdSfdU, Y, ϕ, fermi_action, U, X0; coeff)
end

function _calc_UdSfdU_fromX_MPILattice!(
    UdSfdU::Vector{<:AbstractGaugefields},
    Y,
    ϕ,
    fermi_action,
    U,
    X0;
    coeff=1,
)
    Dim = length(U)
    Dim == 4 || throw(ArgumentError(
        "the MPILattice domain-wall force requires four gauge directions"))
    W = fermi_action.diracoperator.D5DW(U)
    temps = fermi_action._temporary_fermionfields

    temps_dw, it_temps_dw = get_temp(temps)
    X, it_X = get_temp(temps)
    Z, it_Z = get_temp(temps)
    fifth_scratch, it_fifth_scratch = get_temp(temps)

    mul!(temps_dw, W, X0)
    clear_fermion!(Y)
    add_fermion!(Y, 1, ϕ, -1, temps_dw)
    set_wing_fermion!(Y)
    unused!(temps, it_temps_dw)

    apply_F!(X, W.L5, W.mass, X0, fifth_scratch)
    apply_δF!(Z, W.L5, 1 - W.mass, X0, fifth_scratch)
    if W isa D5DW_GeneralizedDomainwall_operator_MPILattice
        combine_generalized_domainwall_force!(
            X, X0, Z, W.D.a, W.D.b, W.D.c)
        force_c = one(eltype(W.as))
    else
        force_c = (W.b - W.c) / 2
        add!(force_c, X, (W.c + W.b) / 2, X0)
    end

    gauge_temps = fermi_action._temporary_gaugefields
    temp_gauge, it_temp_gauge = get_temp(gauge_temps)
    temp_fermion, it_temp_fermion = get_temp(temps)
    spin_scratch, it_spin_scratch = get_temp(temps)
    κ = 1 / 2

    for μ in 1:Dim
        Xplus = shift_fermion(X, μ)
        mul!(temp_fermion, U[μ], Xplus)
        mul_1minusγμx!(spin_scratch, temp_fermion, μ)
        mul!(temp_fermion, κ, spin_scratch)
        muladd_U!(
            UdSfdU[μ], coeff, temp_gauge,
            temp_fermion, Y', spin_scratch)

        Yplus = shift_fermion(Y, μ)
        mul!(temp_fermion, Yplus', U[μ]')
        mul_x1plusγμ!(spin_scratch, temp_fermion, μ)
        mul!(temp_fermion, κ, spin_scratch)
        muladd_U!(
            UdSfdU[μ], -coeff, temp_gauge,
            X, temp_fermion, spin_scratch)

        Zplus = shift_fermion(Z, μ)
        mul!(temp_fermion, U[μ], Zplus)
        mul_1minusγμx!(spin_scratch, temp_fermion, μ)
        mul!(temp_fermion, κ, spin_scratch)
        muladd_U!(
            UdSfdU[μ], coeff * force_c, temp_gauge,
            temp_fermion, ϕ', spin_scratch)

        ϕplus = shift_fermion(ϕ, μ)
        mul!(temp_fermion, ϕplus', U[μ]')
        mul_x1plusγμ!(spin_scratch, temp_fermion, μ)
        mul!(temp_fermion, κ, spin_scratch)
        muladd_U!(
            UdSfdU[μ], -coeff * force_c, temp_gauge,
            Z, temp_fermion, spin_scratch)
    end

    unused!(temps, it_temp_fermion)
    unused!(temps, it_spin_scratch)
    unused!(gauge_temps, it_temp_gauge)
    unused!(temps, it_Z)
    unused!(temps, it_X)
    unused!(temps, it_fifth_scratch)
    return nothing
end

function calc_p_UdSfdU!(
    p,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
    coeff=1,
) where {Dim,Dirac,fermion,gauge}
    #println("------dd")
    QD5DW = fermi_action.diracoperator.D5DW(U)
    Q = MobiusD5DWdagD5DW_Wilson_operator(QD5DW)
    D5_PV = fermi_action.diracoperator.D5DW_PV(U)

    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)

    #temps_dw = fermi_action._temporary_fermionfields[1]

    #X = fermi_action._temporary_fermionfields[end]
    #Y = fermi_action._temporary_fermionfields[end-1]
    X, it_X = get_temp(temps)
    Y, it_Y = get_temp(temps)

    mul!(temps_dw, D5_PV', ϕ)

    solve_DinvX!(X, Q, temps_dw)


    #set_wing_fermion!(ϕ)

    calc_p_UdSfdU_fromX!(p, Y, ϕ, fermi_action, U, X, coeff=coeff)
    #println("----aa--")
    #set_wing_U!(UdSfdU)
    unused!(temps, it_temps_dw)
    unused!(temps, it_X)
    unused!(temps, it_Y)
end

function calc_p_UdSfdU_fromX!(
    p,
    Y,
    ϕ,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    X;
    coeff=1,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator.D5DW(U)

    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[1]

    #temps_dw = fermi_action._temporary_fermionfields[1]

    mul!(temps_dw, W, X)
    clear_fermion!(Y)
    add_fermion!(Y, -1, ϕ, 1, temps_dw)
    set_wing_fermion!(Y)

    unused!(temps, it_temps_dw)

    #temp0_g = fermi_action._temporary_gaugefields[1]
    temps_g = fermi_action._temporary_gaugefields
    temp0_g, it_temp0_g = get_temp(temps_g)

    temp0_f5, it_temp0_f5 = get_temp(temps)
    temp1_f5, it_temp1_f5 = get_temp(temps)

    κ = 1 / 2
    Dwilson = W.wilsonoperator
    for i5 = 1:X.L5

        #temp0_f = fermi_action._temporary_fermionfields[1].w[i5] #F_field
        #temp1_f = fermi_action._temporary_fermionfields[2].w[i5] #F_field
        temp0_f = temp0_f5.w[i5]
        temp1_f = temp1_f5.w[i5]

        for μ = 1:Dim
            #!  Construct U(x,mu)*P1
            Xs = X.w[i5]
            Ys = Y.w[i5]


            # U_{k,μ} X_{k+μ}
            Xsplus = shift_fermion(Xs, μ)

            #@time mul!(temp0_f,U[μ],X)
            mul!(temp0_f, U[μ], Xsplus)

            # (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp1_f, view(Dwilson.rminusγ, :, :, μ), temp0_f)

            # κ (r-γ_μ) U_{k,μ} X_{k+μ}

            mul!(temp0_f, κ, temp1_f)



            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            mul!(temp0_g, temp0_f, Ys')

            Traceless_antihermitian_add!(p[μ], -coeff, temp0_g)
            #println(p[μ][1,1,1,1,1])


            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            Ysplus = shift_fermion(Ys, μ)
            mul!(temp0_f, Ysplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp1_f, temp0_f, view(Dwilson.rplusγ, :, :, μ))

            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)

            # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_g, Xs, temp0_f)

            Traceless_antihermitian_add!(p[μ], coeff, temp0_g)
        end
    end

    unused!(temps, it_temp0_f5)
    unused!(temps, it_temp1_f5)
    unused!(temps_g, it_temp0_g)



end



function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
) where {Dim,Dirac,fermion,gauge}
    #gauss_distribution_fermion!(η)
    gauss_distribution_fermion!(η, rand)
end


function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    mul!(ϕ, W', ξ)
    set_wing_fermion!(ϕ)
end
