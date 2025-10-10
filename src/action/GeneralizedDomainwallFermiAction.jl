import Gaugefields: Traceless_antihermitian_add!



struct GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge} <:
       FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    # _temporary_fermionfields::Vector{fermion}
    # _temporary_gaugefields::Vector{gauge}
    _temporary_fermionfields::Temporalfields{fermion}#Vector{fermion}
    _temporary_gaugefields::Temporalfields{gauge}#Vector{gauge}

    function GeneralizedDomainwallFermiAction(
        D::Dirac_operator{Dim},
        hascovnet,
        covneuralnet,
    ) where {Dim}
        num = 10
        temps = get_temporaryvectors(D)
        x = temps[1]
        # x, it_x = get_temp(temps)
        xtype = typeof(x)
        _temporary_fermionfields = Temporalfields(x; num)
        # _temporary_fermionfields = Array{xtype,1}(undef, num)

        # for i = 1:num
        #     _temporary_fermionfields[i] = similar(x)
        # end

        Utemp = D.U[1]
        Utype = typeof(Utemp)
        numU = 2
        # _temporary_gaugefields = Array{Utype,1}(undef, numU)
        _temporary_gaugefields = Temporalfields(Utemp; num=numU)
        # for i = 1:numU
        #     _temporary_gaugefields[i] = similar(Utemp)
        # end

        # unused!(temps, it_x)


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
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    temps = fermi_action._temporary_fermionfields
    # η = fermi_action._temporary_fermionfields[1]
    η, it_η = get_temp(temps)
    solve_DinvX!(η, W', ϕ)
    Sf = dot(η, η)
    unused!(temps, it_η)
    return real(Sf)
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    #println("------dd")
    QD5DW = fermi_action.diracoperator.D5DW(U)
    Q = GeneralizedD5DWdagD5DW_Wilson_operator(QD5DW)
    D5_PV = fermi_action.diracoperator.D5DW_PV(U)

    # temps_dw = fermi_action._temporary_fermionfields[1]
    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[1]

    # X0 = fermi_action._temporary_fermionfields[6]
    # Y = fermi_action._temporary_fermionfields[5]

    X0, it_X0 = get_temp(temps)
    Y, it_Y = get_temp(temps)


    mul!(temps_dw, D5_PV', ϕ) #temps_dw = D5_PV'*ϕ

    solve_DinvX!(X0, Q, temps_dw) #X0 = Q^-1 D5_PV'*ϕ
    #set_wing_fermion!(X)
    set_wing_fermion!(ϕ)

    clear_U!(UdSfdU)

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
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    X0;
    coeff = 1,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator.D5DW(U)
    # temps_dw = fermi_action._temporary_fermionfields[2]
    # X = fermi_action._temporary_fermionfields[9]
    # Z = fermi_action._temporary_fermionfields[10]

    temps = fermi_action._temporary_fermionfields

    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    X, it_X = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    Z, it_Z = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    temp1, it_temp1 = get_temp(temps)#fermi_action._temporary_fermionfields[2]


    # temp1 = fermi_action._temporary_fermionfields[8]


    mul!(temps_dw, W, X0) #D5DW(U)*Q^-1 D5_PV'*ϕ
    clear_fermion!(Y)
    add_fermion!(Y, 1, ϕ, -1, temps_dw) #Y = D5DW(U)*Q^-1 D5_PV'*ϕ - ϕ
    set_wing_fermion!(Y)

    unused!(temps, it_temps_dw)

    bs = W.bs
    cs = W.cs
    L5 = W.L5
    m = W.mass

    apply_F!(X, L5, m, X0, temp1)  #X = F(m)*Q^-1 D5_PV'*ϕ

    for i5 = 1:L5
        add!(cs[i5], X.w[i5], bs[i5], X0.w[i5]) #X = (c-b)/2 * F(m)*Q^-1 D5_PV'*ϕ + (c+b)/2 * Q^-1 D5_PV'*ϕ
    end
    # COMMENT: (b - c)の符号変更

    apply_δF!(Z, L5, 1 - m, X0, temp1) #Z = dF(1-m)*Q^-1 D5_PV'*ϕ

    temps_g = fermi_action._temporary_gaugefields
    temp0_g, it_temp0_g = get_temp(temps_g)# = fermi_action._temporary_gaugefields[1]
    # temp0_g = fermi_action._temporary_gaugefields[1]

    κ = 1 / 2
    Dwilson = W.wilsonoperator

    L5 = fermi_action.diracoperator.D5DW.L5

    if L5 != X.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):X.L5


        for i5 = 1:X.L5
            if i5 <= div(L5, 2) || i5 >= X.L5 - div(L5, 2) + 1
                push!(irange, i5)
            else
                push!(irange_out, i5)
            end

        end

    else
        irange = 1:L5
    end

    temp0_f5, it_temp0_f5 = get_temp(temps)
    temp1_f5, it_temp1_f5 = get_temp(temps)


    #    for i5=1:X.L5
    for i5 in irange

        # temp0_f = fermi_action._temporary_fermionfields[1].w[i5] #F_field
        # temp1_f = fermi_action._temporary_fermionfields[2].w[i5] #F_field

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
            mul!(temp1_f, Dwilson.rminusγ[ :, :, μ], temp0_f)

            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)

            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            mul!(temp0_g, temp0_f, Ys')

            add_U!(UdSfdU[μ], coeff, temp0_g)

            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            Ysplus = shift_fermion(Ys, μ)
            mul!(temp0_f, Ysplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])

            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)

            # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            #println(getvalue(temp0_g,1,1,1,1,1,1))
            #println(temp0_g[1,1,1,1,1,1])
            mul!(temp0_g, Xs, temp0_f)
            #println("temp0_g , ",getvalue(temp0_g,1,1,1,1,1,1))
            #println("temp0_g , ",temp0_g[1,1,1,1,1,1])
            #println("coeff ", coeff)
            #println("before ",getvalue(UdSfdU[μ],1,1,1,1,1,1))
            #println("before ",UdSfdU[μ][1,1,1,1,1,1])

            add_U!(UdSfdU[μ], -coeff, temp0_g)
            #error("h")
            #println("after ",getvalue(UdSfdU[μ],1,1,1,1,1,1))
            #println("after ",UdSfdU[μ][1,1,1,1,1,1])

            Zs = Z.w[i5]
            ϕs = ϕ.w[i5]

            # U_{k,μ} X_{k+μ}
            Zsplus = shift_fermion(Zs, μ)


            #@time mul!(temp0_f,U[μ],X)
            mul!(temp0_f, U[μ], Zsplus)

            # (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp1_f, Dwilson.rminusγ[ :, :, μ], temp0_f)

            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)

            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            mul!(temp0_g, temp0_f, ϕs')

            # add_U!(UdSfdU[μ], -coeff * (c - b) / 2, temp0_g)
            # add_U!(UdSfdU[μ], -coeff * (b - c) / 2, temp0_g)
            add_U!(UdSfdU[μ], coeff * cs[i5], temp0_g)
            # COMMENT: (b - c)の符号変更

            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            ϕsplus = shift_fermion(ϕs, μ)
            mul!(temp0_f, ϕsplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])

            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)

            mul!(temp0_g, Zs, temp0_f)

            # add_U!(UdSfdU[μ], coeff * (c - b) / 2, temp0_g)
            # add_U!(UdSfdU[μ], coeff * (b - c) / 2, temp0_g)
            add_U!(UdSfdU[μ], -coeff * cs[i5], temp0_g)
            # COMMENT: (b - c)の符号変更

        end


    end

    unused!(temps, it_temp0_f5)
    unused!(temps, it_temp1_f5)
    unused!(temps_g, it_temp0_g)
    unused!(temps, it_Z)
    unused!(temps, it_X)
    unused!(temps, it_temp1)


end



function calc_p_UdSfdU!(
    p,
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
    coeff = 1,
) where {Dim,Dirac,fermion,gauge}
    #println("------dd")
    QD5DW = fermi_action.diracoperator.D5DW(U)
    Q = GeneralizedD5DWdagD5DW_Wilson_operator(QD5DW)
    D5_PV = fermi_action.diracoperator.D5DW_PV(U)

    # temps_dw = fermi_action._temporary_fermionfields[1]
    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)

    # X = fermi_action._temporary_fermionfields[end]
    # Y = fermi_action._temporary_fermionfields[end-1]

    X, it_X = get_temp(temps)
    Y, it_Y = get_temp(temps)


    mul!(temps_dw, D5_PV', ϕ)

    solve_DinvX!(X, Q, temps_dw)


    #set_wing_fermion!(ϕ)

    calc_p_UdSfdU_fromX!(p, Y, ϕ, fermi_action, U, X, coeff = coeff)
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
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    U,
    X;
    coeff = 1,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator.D5DW(U)
    # temps_dw = fermi_action._temporary_fermionfields[1]
    temps = fermi_action._temporary_fermionfields
    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[1]
    
    mul!(temps_dw, W, X)
    clear_fermion!(Y)
    add_fermion!(Y, -1, ϕ, 1, temps_dw)
    set_wing_fermion!(Y)

    unused!(temps, it_temps_dw)

    # temp0_g = fermi_action._temporary_gaugefields[1]
    temps_g = fermi_action._temporary_gaugefields
    temp0_g, it_temp0_g = get_temp(temps_g)

    κ = 1 / 2
    Dwilson = W.wilsonoperator

    temp0_f5, it_temp0_f5 = get_temp(temps)
    temp1_f5, it_temp1_f5 = get_temp(temps)
    for i5 = 1:X.L5

        # temp0_f = fermi_action._temporary_fermionfields[1].w[i5] #F_field
        # temp1_f = fermi_action._temporary_fermionfields[2].w[i5] #F_field

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
            mul!(temp1_f, Dwilson.rminusγ[ :, :, μ], temp0_f)

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
            mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])

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
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
) where {Dim,Dirac,fermion,gauge}
    #gauss_distribution_fermion!(η)
    gauss_distribution_fermion!(η, rand)
end

using InteractiveUtils

function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    mul!(ϕ, W', ξ)
    set_wing_fermion!(ϕ)
end
