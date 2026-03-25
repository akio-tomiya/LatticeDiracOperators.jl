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

    # === コンストラクタ2: Renew用（今回追加が必要なもの） ===
    # 全フィールドを引数に取り、そのまま new に渡すコンストラクタです。
    function GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge}(
        hascovnet::Bool,
        covneuralnet::Union{Nothing,CovNeuralnet{Dim}},
        D::Dirac,
        temp_f::Temporalfields{fermion},
        temp_g::Temporalfields{gauge}
    ) where {Dim,Dirac,fermion,gauge}
        return new{Dim,Dirac,fermion,gauge}(
            hascovnet,
            covneuralnet,
            D,
            temp_f,
            temp_g
        )
    end
end

function Renew(fermi_action::GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge}, D::Dirac_operator{Dim},
    ) where {Dim,Dirac,fermion,gauge}
    return GeneralizedDomainwallFermiAction{Dim,Dirac,fermion,gauge}(
        fermi_action.hascovnet,
        fermi_action.covneuralnet,
        D,
        fermi_action._temporary_fermionfields,
        fermi_action._temporary_gaugefields,
    )
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

    # apply_F!(X, L5, m, X0, temp1)  #X = F(m)*Q^-1 D5_PV'*ϕ
    apply_F!(X, L5, m, X0)

    # for i5 = 1:L5
    #     add!(cs[i5], X.w[i5], bs[i5], X0.w[i5]) #X = (c-b)/2 * F(m)*Q^-1 D5_PV'*ϕ + (c+b)/2 * Q^-1 D5_PV'*ϕ
    # end
    add!(cs, X, bs, X0)
    # COMMENT: (b - c)の符号変更

    # apply_δF!(Z, L5, 1 - m, X0, temp1) #Z = dF(1-m)*Q^-1 D5_PV'*ϕ
    apply_δF!(Z, L5, 1 - m, X0)

    temps_g = fermi_action._temporary_gaugefields
    temp0_g, it_temp0_g = get_temp(temps_g)# = fermi_action._temporary_gaugefields[1]
    # temp0_g = fermi_action._temporary_gaugefields[1]

    κ = 1 / 2
    # Dwilson = W.wilsonoperator

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
    # for i5 in irange

        # temp0_f = fermi_action._temporary_fermionfields[1].w[i5] #F_field
        # temp1_f = fermi_action._temporary_fermionfields[2].w[i5] #F_field

        temp0_f = temp0_f5
        temp1_f = temp1_f5

        for μ = 1:Dim
            #!  Construct U(x,mu)*P1
            Xs = X
            Ys = Y

            # U_{k,μ} X_{k+μ}
            Xsplus = shift_fermion(Xs, μ)


            #@time mul!(temp0_f,U[μ],X)
            mul!(temp0_f, U[μ], Xsplus)

            # (r-γ_μ) U_{k,μ} X_{k+μ}
            # mul!(temp1_f, Dwilson.rminusγ[ :, :, μ], temp0_f)
            mul_1minusγμx!(temp1_f, temp0_f, μ)

            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)

            # # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            # mul!(temp0_g, temp0_f, Ys')

            # add_U!(UdSfdU[μ], coeff, temp0_g)
            muladd_U!(UdSfdU[μ], coeff, temp0_g,temp0_f, Ys',temp1_f)

            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            Ysplus = shift_fermion(Ys, μ)
            mul!(temp0_f, Ysplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            # mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])

            # # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            # mul!(temp0_f, κ, temp1_f)

            # # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            # #println(getvalue(temp0_g,1,1,1,1,1,1))
            # #println(temp0_g[1,1,1,1,1,1])
            # mul!(temp0_g, Xs, temp0_f)
            # #println("temp0_g , ",getvalue(temp0_g,1,1,1,1,1,1))
            # #println("temp0_g , ",temp0_g[1,1,1,1,1,1])
            # #println("coeff ", coeff)
            # #println("before ",getvalue(UdSfdU[μ],1,1,1,1,1,1))
            # #println("before ",UdSfdU[μ][1,1,1,1,1,1])

            # add_U!(UdSfdU[μ], -coeff, temp0_g)

            mul_x1plusγμ!(temp1_f, temp0_f, μ)

            mul!(temp0_f, κ, temp1_f)
            
            muladd_U!(UdSfdU[μ], -coeff, temp0_g,Xs, temp0_f,temp1_f)
        
            #error("h")
            #println("after ",getvalue(UdSfdU[μ],1,1,1,1,1,1))
            #println("after ",UdSfdU[μ][1,1,1,1,1,1])

            Zs = Z
            ϕs = ϕ

            # U_{k,μ} X_{k+μ}
            Zsplus = shift_fermion(Zs, μ)
        
            #@time mul!(temp0_f,U[μ],X)
            mul!(temp0_f, U[μ], Zsplus)
            
            #=
            println("8_1 ",dot(temp0_f,temp0_f))
            substitute_fermion!(Zs_2,Zs)
            Zsplus_2 = shift_fermion(Zs_2, μ)
            mul!(temp0_f_2, Umu_2, Zsplus_2)
            println("8_2 ",dot(temp0_f_2,temp0_f_2))
            =#
            

            # (r-γ_μ) U_{k,μ} X_{k+μ}
            #mul!(temp1_f, Dwilson.rminusγ[:, :, μ], temp0_f)
            # (r-γ_μ) U_{k,μ} X_{k+μ}
            #mul!(temp1_f, view(W.rminusγ, :, :, μ), temp0_f)
            
            mul_1minusγμx!(temp1_f, temp0_f, μ)

            #=
            println("9_1 ",dot(temp1_f,temp1_f))
            substitute_fermion!(temp0_f_2, temp0_f)
            mul_1minusγμx!(temp1_f_2, temp0_f_2, μ)
            println("9_2 ",dot(temp1_f_2,temp1_f_2))
            =#
            



            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)
            
            #=
            println("10_1 ",dot(temp0_f,temp0_f))
            substitute_fermion!(temp1_f_2, temp1_f)
            mul!(temp0_f_2, κ, temp1_f_2)
            println("10_2 ",dot(temp0_f_2,temp0_f_2))
            =#
            

            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            #mul!(temp0_g, temp0_f, ϕs')
            #add_U!(UdSfdU[μ], coeff * (b - c) / 2, temp0_g)
            #substitute_U!(tempgtemp, UdSfdU[μ])

            muladd_U!(UdSfdU[μ], cs, coeff, temp0_g,temp0_f, ϕs',temp1_f)
            # muladd_U!(UdSfdU[μ], 0.5, temp0_g,temp0_f, ϕs',temp1_f)

            #=
            println("11_1 ",tr(UdSfdU[μ]))
            substitute_fermion!(temp0_f_2, temp0_f)
            substitute_fermion!(ϕs_2, ϕs)
            muladd_U!(tempgtemp, coeff* (b - c) / 2, temp0_g_2,temp0_f_2, ϕs_2',temp1_f_2)
            println("11_2 ",tr(tempgtemp))
            =#
            


            #println("after2 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #println("after3 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            ϕsplus = shift_fermion(ϕs, μ)
            mul!(temp0_f, ϕsplus', U[μ]')
            
            #=
            println("12_1 ",dot(temp0_f,temp0_f))
            substitute_fermion!(ϕs_2, ϕs)
            ϕsplus_2 = shift_fermion(ϕs_2, μ)
            mul!(temp0_f_2, ϕsplus_2', Umu_2')
            println("12_2 ",dot(temp0_f_2,temp0_f_2))
            =#
            

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            #mul!(temp1_f, temp0_f, Dwilson.rplusγ[:, :, μ])
            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            if Dim == 4
                mul_x1plusγμ!(temp1_f, temp0_f, μ)
                #=
                println("13_1 ",dot(temp1_f,temp1_f))
                substitute_fermion!(temp0_f_2, temp0_f)
                mul_x1plusγμ!(temp1_f_2, temp0_f_2, μ)
                println("13_2 ",dot(temp1_f_2,temp1_f_2))
                =#
                
            else
                mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])
            end

            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)
            #=
            println("14_1 ",dot(temp0_f,temp0_f))
            substitute_fermion!(temp1_f_2, temp1_f)
            mul!(temp0_f_2, κ, temp1_f_2)
            println("14_2 ",dot(temp0_f_2,temp0_f_2))
            =#
            


            muladd_U!(UdSfdU[μ], cs, -coeff, temp0_g,Zs, temp0_f,temp1_f)
            # muladd_U!(UdSfdU[μ], -0.5, temp0_g,Zs, temp0_f,temp1_f)

            #=
            println("15_1 ",tr(UdSfdU[μ]))
            substitute_fermion!(temp0_f_2, temp0_f)
            substitute_fermion!(Zs_2, Zs)
            muladd_U!(tempgtemp, -coeff * (b - c) / 2, temp0_g_2,Zs_2, temp0_f_2,temp1_f_2)
            println("15_2 ",tr(tempgtemp))
            error("end")
            =#
            
            #println("after4 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])

        end


    # end

    unused!(temps, it_temp0_f5)
    unused!(temps, it_temp1_f5)
    unused!(temps_g, it_temp0_g)
    unused!(temps, it_Z)
    unused!(temps, it_X)
    unused!(temps, it_temp1)

    return


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

function calc_mres_and_derivative(fermi_action::GeneralizedDomainwallFermiAction, U, Nr)
    temps = fermi_action._temporary_fermionfields
    p, it_p = get_temp(temps)
    q, it_q = get_temp(temps)
    r, it_r = get_temp(temps)
    s, it_s = get_temp(temps)
    t, it_t = get_temp(temps)
    # t2, it_t2 = get_temp(temps)
    temp3, it_temp3 = get_temp(temps)
    temp4, it_temp4 = get_temp(temps)
    mres = 0.0
    dmdb = 0.0
    dmdc = 0.0
    D = fermi_action.diracoperator.D5DW(U)
    PV = fermi_action.diracoperator.D5DW_PV(U)
    DdagD = GeneralizedD5DWdagD5DW_Wilson_operator(D)
    # as = fermi_action.diracoperator.as
    
    bs = fermi_action.diracoperator.bs 
    cs = fermi_action.diracoperator.cs
    L5 = D.L5
    as = ones(L5)
    factor = 1.0 / Nr
    den1 = zeros(L5)
    num1 = zeros(L5)
    den2 = zeros(L5)
    num2 = zeros(L5)
    den3 = zeros(L5)
    num3 = zeros(L5)
    den4 = zeros(L5)
    num4 = zeros(L5)
    Ns = 0.0
    Ds = 0.0
    dmdbs = zeros(L5)
    dmdcs = zeros(L5)

    for ir = 1:Nr
        clear_fermion!(p)
        Z4_distribution_fermi!(p)


        apply_P!(q, p)

        apply_R!(r, q)

        solve_DinvX!(s, D, r)

        apply_P_edge!(t, s)

        for i5 = 1:L5 
            den1[i5] += real.(dot(t.w[i5], t.w[i5]))
            num1[i5] += real.(dot(q.w[i5], s.w[i5]))
        end
        # # Ns = sum(num1)
        # # Ds = sum(den1)

        # # mres += Ns / Ds * factor

        solve_DinvX!(p,D',q)
        # solve_DinvX!(temp3, DdagD, q)
        # mul!(p, D, temp3)

        solve_DinvX!(q,D',t)
        # solve_DinvX!(temp4, DdagD, t)
        # mul!(q, D, temp4)
        
        # # apply_dDdas!(r, U, s, D.mass, D.wilsonoperator, L5, bs, cs, temp3, temp4)

        # # den2 -= 2.0 * real(dot(t, t2))
        # # num2 -= real(dot(q, p))
        # # for i5 = 1:L5 
        #     # den2[i5] -= 2.0 * real.(dot(q.w[i5], r.w[i5]))
        #     # num2[i5] -= real.(dot(p.w[i5], r.w[i5]))
        # # end

        apply_dDdbs!(r, U, s, D.wilsonoperator, as)

        # # den3 -= 2.0 * real(dot(t, t2))
        # # num3 -= real(dot(q, p))
        for i5 = 1:L5 
            den3[i5] += 2.0 * real.(dot(q.w[i5], r.w[i5]))
            num3[i5] -= real.(dot(p.w[i5], r.w[i5]))
        end

        # # for i5 = 1:L5
        # #     dmdbs[i5] -= (num3[i5] * Ds - Ns * den3[i5]) / Ds^2 * factor
        # # end



        apply_dDdcs!(r, U, s, D.mass, D.wilsonoperator, as, temp3, temp4)

        for i5 = 1:L5 
            den4[i5] += 2.0 * real.(dot(q.w[i5], r.w[i5]))
            num4[i5] -= real.(dot(p.w[i5], r.w[i5]))
        end

        # for i5 = 1:L5
        #     dmdcs[i5] -= (num4[i5] * Ds - Ns * den4[i5]) / Ds^2 * factor
        # end
        # dmdc += (num2 * den1 - num1 * den2) / den1^2

    end

    Ns = sum(num1)
    Ds = sum(den1)
    mres = Ns / Ds * factor# - D.mass 
    # dmdas = zeros(L5)
    # dmdbs = zeros(L5)
    # dmdcs = zeros(L5)
    for i5 = 1:L5
        # dmdas[i5] = (num2[i5] * Ds - Ns * den2[i5]) / Ds^2 * factor
        dmdbs[i5] = (num3[i5] * Ds - Ns * den3[i5]) / Ds^2 * factor
        dmdcs[i5] = (num4[i5] * Ds - Ns * den4[i5]) / Ds^2 * factor
    end

    # mres -= D.mass

    unused!(temps, it_p)
    unused!(temps, it_q)
    unused!(temps, it_r)
    unused!(temps, it_t)
    unused!(temps, it_s)
    unused!(temps, it_temp3)
    unused!(temps, it_temp4)
    # unused!(temps, it_t2)

    return mres, dmdbs, dmdcs
end


function calc_mres_and_derivative_gpu(fermi_action::GeneralizedDomainwallFermiAction, U, Nr)
    temps = fermi_action._temporary_fermionfields
    p, it_p = get_temp(temps)
    q, it_q = get_temp(temps)
    r, it_r = get_temp(temps)
    s, it_s = get_temp(temps)
    t, it_t = get_temp(temps)
    # t2, it_t2 = get_temp(temps)
    temp3, it_temp3 = get_temp(temps)
    temp4, it_temp4 = get_temp(temps)
    mres = 0.0
    dmdb = 0.0
    dmdc = 0.0
    D = fermi_action.diracoperator.D5DW(U)
    DdagD = DdagD_GeneralizedDomainwall_operator_MPILattice(D)
    # PV = fermi_action.diracoperator.D5DW_PV(U)
    # as = fermi_action.diracoperator.as
    bs = fermi_action.diracoperator.bs 
    cs = fermi_action.diracoperator.cs
    L5 = D.L5
    as = ones(L5)
    factor = 1.0 / Nr
    den1 = zeros(L5)
    num1 = zeros(L5)
    den2 = zeros(L5)
    num2 = zeros(L5)
    den3 = zeros(L5)
    num3 = zeros(L5)
    den4 = zeros(L5)
    num4 = zeros(L5)
    wilson_params = D.D.wilson_params
    κ = wilson_params.κ_wilson
    Ns = 0.0
    Ds = 0.0
    mres = 0.0
    dmdbs = zeros(L5)
    dmdcs = zeros(L5)

    for ir = 1:Nr
        clear_fermion!(p)
        Z4_distribution_fermi!(p)

        apply_P!(q, L5, p)

        apply_R!(r, L5, q)

        solve_DinvX!(s, D, r)

        apply_P_edge!(t, L5, s)
        # tt = dot(t, t)
        # println("tt = $tt")
        # Ds = real(dot(t, t))
        # Ns = real(dot(q, s))
        # for i5 = 1:L5 
        #     den1[i5] += real.(dot(t.w[i5], t.w[i5]))
        #     num1[i5] += real.(dot(q.w[i5], s.w[i5]))
        # end
        den1 .+= real.(dot_4dim!(t, t))
        num1 .+= real.(dot_4dim!(q, s))
        # dd1 += real(dot(t,t))
        # dn1 += real(dot(q,s))

        # Ns = sum(num1)
        # Ds = sum(den1)

        # mres += Ns / Ds * factor
        # Ds += real.(dot(t, t))
        # Ns += real.(dot(q, s))
        # mres += num1 / den1 

        solve_DinvX!(p,D',q)
        solve_DinvX!(q,D',t)

        apply_dDdbs!(r, U, κ, s)

        den3 .-= 2.0 .* real.(dot_4dim!(q, r))
        num3 .-= real.(dot_4dim!(p, r))

        # # for i5 = 1:L5
        # #     dmdbs[i5] -= (num3[i5] * Ds - Ns * den3[i5]) / Ds^2 * factor
        # # end

        apply_dDdcs!(r, U, κ, s, D.mass, temp3, temp4)

        # # for i5 = 1:L5 
        # #     den4[i5] -= 2.0 * real.(dot(q.w[i5], r.w[i5]))
        # #     num4[i5] -= real.(dot(p.w[i5], r.w[i5]))
        # # end
        den4 .-= 2.0 .* real.(dot_4dim!(q, r))
        num4 .-= real.(dot_4dim!(p, r))

        # for i5 = 1:L5
        #     dmdcs[i5] -= (num4[i5] * Ds - Ns * den4[i5]) / Ds^2 * factor
        # end

        # dmdc += (num2 * den1 - num1 * den2) / den1^2

    end

    Ns = sum(num1)
    Ds = sum(den1)
    mres = Ns / Ds * factor# - D.mass 
    # mres = dn1 / dd1 * factor

    # dmdas = zeros(L5)
    # dmdbs = zeros(L5)
    # dmdcs = zeros(L5)
    for i5 = 1:L5
        # dmdas[i5] = (num2[i5] * Ds - Ns * den2[i5]) / Ds^2 * factor
        dmdbs[i5] = (num3[i5] * Ds - Ns * den3[i5]) / Ds^2 * factor
        dmdcs[i5] = (num4[i5] * Ds - Ns * den4[i5]) / Ds^2 * factor
    end

    # mres -= D.mass

    unused!(temps, it_p)
    unused!(temps, it_q)
    unused!(temps, it_r)
    unused!(temps, it_t)
    unused!(temps, it_s)
    unused!(temps, it_temp3)
    unused!(temps, it_temp4)
    # unused!(temps, it_t2)

    return mres, dmdbs, dmdcs
end