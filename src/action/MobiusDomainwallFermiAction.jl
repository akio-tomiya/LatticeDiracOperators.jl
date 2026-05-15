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
        # x = temps[1]
        x, it_x = get_temp(temps)
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

        unused!(temps, it_x)


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
    W = fermi_action.diracoperator.D5DW(U)
    temps = fermi_action._temporary_fermionfields

    temps_dw, it_temps_dw = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    X, it_X = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    Z, it_Z = get_temp(temps)#fermi_action._temporary_fermionfields[2]
    temp1, it_temp1 = get_temp(temps)#fermi_action._temporary_fermionfields[2]


    #temps_dw = fermi_action._temporary_fermionfields[2]
    #X = fermi_action._temporary_fermionfields[9]
    #Z = fermi_action._temporary_fermionfields[10]


    #temp1 = fermi_action._temporary_fermionfields[8]


    mul!(temps_dw, W, X0) #D5DW(U)*Q^-1 D5_PV'*ϕ
    clear_fermion!(Y)
    #add_fermion!(Y, -1, ϕ, 1, temps_dw) #Y = D5DW(U)*Q^-1 D5_PV'*ϕ - ϕ
    add_fermion!(Y, 1, ϕ, -1, temps_dw) #Y = D5DW(U)*Q^-1 D5_PV'*ϕ - ϕ
    set_wing_fermion!(Y)

    unused!(temps, it_temps_dw)

    b = W.b
    c = W.c
    L5 = W.L5
    m = W.mass

    apply_F!(X, L5, m, X0, temp1)  #X = F(m)*Q^-1 D5_PV'*ϕ

    #for i5 = 1:L5
        # add!((c - b) / 2, X.w[i5], (c + b) / 2, X0.w[i5]) #X = (c-b)/2 * F(m)*Q^-1 D5_PV'*ϕ + (c+b)/2 * Q^-1 D5_PV'*ϕ
        # b-cに変更
    #    add!((b - c) / 2, X.w[i5], (c + b) / 2, X0.w[i5]) #X = (c-b)/2 * F(m)*Q^-1 D5_PV'*ϕ + (c+b)/2 * Q^-1 D5_PV'*ϕ
    #end
    add!((b - c) / 2, X, (c + b) / 2, X0)

    apply_δF!(Z, L5, 1 - m, X0, temp1) #Z = dF(1-m)*Q^-1 D5_PV'*ϕ

    temps_g = fermi_action._temporary_gaugefields
    temp0_g, it_temp0_g = get_temp(temps_g)# = fermi_action._temporary_gaugefields[1]

    #temp0_g = fermi_action._temporary_gaugefields[1]

    κ = 1 / 2
    #Dwilson = W.wilsonoperator

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

    #debug
    #=
    x5_2 = MobiusDomainwallFermion_5D_MPILattice(U[1], L5)
    Ys_2 = MobiusDomainwallFermion_5D_MPILattice(U[1], L5)
    Xs_2 = MobiusDomainwallFermion_5D_MPILattice(U[1], L5)
    Zs_2 = MobiusDomainwallFermion_5D_MPILattice(U[1], L5)
    ϕs_2 = MobiusDomainwallFermion_5D_MPILattice(U[1], L5)
    
    temp0_f_2 = similar(x5_2)
    temp1_f_2 = similar(x5_2)
    NC=U[1].NC
    NX=U[1].NX
    NY=U[1].NY
    NZ=U[1].NZ
    NT=U[1].NT

    tempgtemp = Gaugefields_4D_MPILattice(NC,NX,NY,NZ,NT)
    temp0_g_2 = similar(tempgtemp)
    Umu_2 = similar(tempgtemp)
    =#
    

    temp0_f = temp0_f5
    temp1_f = temp1_f5
    for μ = 1:Dim
        #!  Construct U(x,mu)*P1
        Xs = X
        Ys = Y
        # U_{k,μ} X_{k+μ}
        Xsplus = shift_fermion(Xs, μ)
        mul!(temp0_f, U[μ], Xsplus)
        

        # (r-γ_μ) U_{k,μ} X_{k+μ}
        mul_1minusγμx!(temp1_f, temp0_f, μ)

        mul!(temp0_f, κ, temp1_f)

        muladd_U!(UdSfdU[μ], coeff, temp0_g,temp0_f, Ys',temp1_f)
        #muladd_U!(UdSfdU[μ], coeff, temp0_g,temp0_f, Ys')

        # Y_{k+μ}^dag U_{k,μ}^dag
        Ysplus = shift_fermion(Ys, μ)
        mul!(temp0_f, Ysplus', U[μ]')

        
        mul_x1plusγμ!(temp1_f, temp0_f, μ)

         # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f, κ, temp1_f)
        
        

        # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        #muladd_U!(UdSfdU[μ], -coeff, temp0_g,Xs, temp0_f)
        muladd_U!(UdSfdU[μ], -coeff, temp0_g,Xs, temp0_f,temp1_f)
        

        Zs = Z
        ϕs = ϕ

        # U_{k,μ} X_{k+μ}
        Zsplus = shift_fermion(Zs, μ)
    
        #@time mul!(temp0_f,U[μ],X)
        mul!(temp0_f, U[μ], Zsplus)
        
        mul_1minusγμx!(temp1_f, temp0_f, μ)
        mul!(temp0_f, κ, temp1_f)
        

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        muladd_U!(UdSfdU[μ], coeff* (b - c) / 2, temp0_g,temp0_f, ϕs',temp1_f)
        #muladd_U!(UdSfdU[μ], coeff* (b - c) / 2, temp0_g,temp0_f, ϕs')


        # Y_{k+μ}^dag U_{k,μ}^dag
        ϕsplus = shift_fermion(ϕs, μ)
        mul!(temp0_f, ϕsplus', U[μ]')

        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        #mul!(temp1_f, temp0_f, Dwilson.rplusγ[:, :, μ])
        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        if Dim == 4
            mul_x1plusγμ!(temp1_f, temp0_f, μ)

        else
            mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])
        end

        # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f, κ, temp1_f)

        #mul!(temp0_g, Zs, temp0_f)
        #add_U!(UdSfdU[μ], -coeff * (b - c) / 2, temp0_g)
        #substitute_U!(tempgtemp, UdSfdU[μ])
        muladd_U!(UdSfdU[μ], -coeff * (b - c) / 2, temp0_g,Zs, temp0_f,temp1_f)
        #muladd_U!(UdSfdU[μ], -coeff * (b - c) / 2, temp0_g,Zs, temp0_f)

    end


    unused!(temps, it_temp0_f5)
    unused!(temps, it_temp1_f5)
    unused!(temps_g, it_temp0_g)
    unused!(temps, it_Z)
    unused!(temps, it_X)
    unused!(temps, it_temp1)


    return 

    #    for i5=1:X.L5
    for i5 in irange

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
            #mul!(temp1_f, Dwilson.rminusγ[:, :, μ], temp0_f)
            mul_1minusγμx!(temp1_f, temp0_f, μ)


            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)

            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            mul!(temp0_g, temp0_f, Ys')
            add_U!(UdSfdU[μ], coeff, temp0_g)

            
            #display(temp1_f[:, :, 1, 1, 1, 1])
            #display(temp0_f[:, :, 1, 1, 1, 1])
            #display(Ys[:, :, 1, 1, 1, 1])

            
            #println("after1 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #error("h")
            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            Ysplus = shift_fermion(Ys, μ)
            mul!(temp0_f, Ysplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            #mul!(temp1_f, temp0_f, Dwilson.rplusγ[ :, :, μ])
            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            if Dwilson.r == 1 && Dim == 4
                mul_x1plusγμ!(temp1_f, temp0_f, μ)
            else
                mul!(temp1_f, temp0_f, view(Dwilson.rplusγ, :, :, μ))
            end



            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)

            # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            #println(getvalue(temp0_g,1,1,1,1,1,1))
            #println(temp0_g[1,1,1,1,1,1])
            mul!(temp0_g, Xs, temp0_f)

            add_U!(UdSfdU[μ], -coeff, temp0_g)
            #println("after2 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])

            #error("h")
            #println("after ",getvalue(UdSfdU[μ],1,1,1,1,1,1))

            Zs = Z.w[i5]
            ϕs = ϕ.w[i5]

            # U_{k,μ} X_{k+μ}
            Zsplus = shift_fermion(Zs, μ)


            #@time mul!(temp0_f,U[μ],X)
            mul!(temp0_f, U[μ], Zsplus)

            # (r-γ_μ) U_{k,μ} X_{k+μ}
            #mul!(temp1_f, Dwilson.rminusγ[:, :, μ], temp0_f)
            # (r-γ_μ) U_{k,μ} X_{k+μ}
            #mul!(temp1_f, view(W.rminusγ, :, :, μ), temp0_f)
            if Dwilson.r == 1 && Dim == 4
                mul_1minusγμx!(temp1_f, temp0_f, μ)
            else
                mul!(temp1_f, view(Dwilson.rminusγ, :, :, μ), temp0_f)
            end


            # κ (r-γ_μ) U_{k,μ} X_{k+μ}
            mul!(temp0_f, κ, temp1_f)

            # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
            mul!(temp0_g, temp0_f, ϕs')
            #println("temp0_g ", temp0_g[1, 1, 1, 1, 1, 1])
            #println(tr(temp0_g))
            #println("before3 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #println(tr(UdSfdU[μ]))
            add_U!(UdSfdU[μ], coeff * (b - c) / 2, temp0_g)
            #println("after2 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #println("after3 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])
            #!  Construct P2*U_adj(x,mu)
            # Y_{k+μ}^dag U_{k,μ}^dag
            ϕsplus = shift_fermion(ϕs, μ)
            mul!(temp0_f, ϕsplus', U[μ]')

            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            #mul!(temp1_f, temp0_f, Dwilson.rplusγ[:, :, μ])
            # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            if Dwilson.r == 1 && Dim == 4
                mul_x1plusγμ!(temp1_f, temp0_f, μ)
            else
                mul!(temp1_f, temp0_f, view(Dwilson.rplusγ, :, :, μ))
            end

            # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
            mul!(temp0_f, κ, temp1_f)

            mul!(temp0_g, Zs, temp0_f)

            add_U!(UdSfdU[μ], -coeff * (b - c) / 2, temp0_g)
            #println("after4 ", UdSfdU[μ][1, 1, 1, 1, 1, 1])

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

function calc_mres_and_derivative(fermi_action, U, Nr)
    temps = fermi_action._temporary_fermionfields
    p, it_p = get_temp(temps)
    q, it_q = get_temp(temps)
    r, it_r = get_temp(temps)
    s, it_s = get_temp(temps)
    t, it_t = get_temp(temps)
    temp3, it_temp3 = get_temp(temps)
    temp4, it_temp4 = get_temp(temps)
    mres = 0.0
    dmdb = 0.0
    dmdc = 0.0
    D = fermi_action.diracoperator.D5DW(U)
    DdagD = DdagD_MobiusDomainwall_operator_MPILattice(D)
    L5 = D.L5
    factor = 1.0 / Nr
    wilson_params = D.D.wilson_params
    κ = wilson_params.κ_wilson
    den1 = 0.0
    num1 = 0.0
    den2 = 0.0
    num2 = 0.0
    den3 = 0.0
    num3 = 0.0

    for ir = 1:Nr
        clear_fermion!(p)
        Z4_distribution_fermi!(p)

        apply_P!(q, L5, p)

        apply_R!(r, L5, q)

        solve_DinvX!(s, D, r)

        apply_P_edge!(t, L5, s)

        den1 += real(dot(t, t))
        num1 += real(dot(q, s))

        apply_dDdb!(r, U, κ, s, D.mass, temp3, temp4)
        solve_DinvX!(p, D, r)
        apply_P_edge!(t, L5, p)

        den2 += 2.0 * real(dot(t, t))
        num2 -= real(dot(q, p))


        apply_dDdc!(r, U, κ, s, D.mass, temp3, temp4)
        solve_DinvX!(p, D, r)
        apply_P_edge!(t, L5, p)

        den3 += 2.0 * real(dot(t, t))
        num3 -= real(dot(q, p))


    end

    mres = num1 / den1 * factor
    dmdb = (num2 * den1 - num1 * den2) / den1^2 * factor
    dmdc = (num3 * den1 - num1 * den3) / den1^2 * factor
    # mres *= factor 
    # dmdb *= factor
    # dmdc *= factor

    unused!(temps, it_p)
    unused!(temps, it_q)
    unused!(temps, it_r)
    unused!(temps, it_t)
    unused!(temps, it_s)
    unused!(temps, it_temp3)
    unused!(temps, it_temp4)

    return real(mres), real(dmdb), real(dmdc)
end

function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::MobiusDomainwallFermiAction{Dim,Dirac,fermion,gauge},
) where {Dim,Dirac,fermion,gauge}
    #gauss_distribution_fermion!(η)
    gauss_distribution_fermion!(η, rand)
end

using InteractiveUtils

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
