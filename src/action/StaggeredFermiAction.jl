import ..Rhmc:
    RHMC, get_order, get_α, get_β, get_α0, get_β_inverse, get_α_inverse, get_α0_inverse

struct StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf} <:
       FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    Nf::Int64
    rhmc_info_for_action::Union{Nothing,RHMC}
    rhmc_info_for_MD::Union{Nothing,RHMC}
    _temporary_fermionfields::Temporalfields{fermion}#Vector{fermion}
    _temporary_gaugefields::Temporalfields{gauge}#Vector{gauge}


    function StaggeredFermiAction(
        D::Dirac_operator{Dim},
        hascovnet,
        covneuralnet,
        parameters_action,
    ) where {Dim}
        @assert haskey(parameters_action, "Nf") "parameters for action should have the keyword Nf"
        Nf_in::Int = parameters_action["Nf"]
        if Dim == 2 #Number of zero modes is different. 
            Nf = 2 * Nf_in
        elseif Dim == 4
            Nf = Nf_in
        end


        if Nf == 4 || Nf == 8 # 8 flavors if phi (Mdag M)^{-1} phi
            rhmc_info_for_action = nothing
            rhmc_info_for_MD = nothing
        else
            #for action: r_action
            #Nf = 8 -> alpha = 1 -> power x^{1/2} 
            #Nf = 2 -> alpha = 1/4 -> power x^1/8 
            #Nf = 1 -> alpha = 1/8  -> power x^1/16 
            order = Nf // 16

            rhmc_info_for_action = RHMC(order, n=15)

            #for MD: r_MD
            #Nf = 8 -> alpha = 1 -> power x^{1} 
            #Nf = 2 -> alpha = 1/4 -> power x^1/4 
            #Nf = 1 -> alpha = 1/8  -> power x^1/8 
            order = Nf // 8
            #rhmcorder = 8 ÷ Nf
            rhmc_info_for_MD = RHMC(order, n=10)

            N_action = get_order(rhmc_info_for_action)
            N_MD = get_order(rhmc_info_for_MD)
        end

        if Nf == 4
            num = 1
        elseif Nf == 8
            num = 1
        else
            num = maximum((N_action, N_MD)) + 1
        end
        num += 4

        #x = D._temporary_fermi[1]
        x, it_x = get_temp(D._temporary_fermi)
        xtype = typeof(x)
        _temporary_fermionfields = Temporalfields(x; num)
        unused!(D._temporary_fermi, it_x)
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
        #nd


        return new{Dim,typeof(D),xtype,Utype,Nf}(
            hascovnet,
            covneuralnet,
            D,
            Nf,
            rhmc_info_for_action,
            rhmc_info_for_MD,
            _temporary_fermionfields,
            _temporary_gaugefields,
        )
    end
end

function evaluate_FermiAction(
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge,Nf}
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Staggered_operator(W)

    rhmc = fermi_action.rhmc_info_for_action

    N = get_order(rhmc)

    x, it_x = get_temp(fermi_action._temporary_fermionfields)
    #x = fermi_action._temporary_fermionfields[end-N]
    #vec_x = fermi_action._temporary_fermionfields[end-N+1:end]
    vec_x, its_vec_x = get_temp(fermi_action._temporary_fermionfields, N)#[end-N+1:end]
    for j = 1:N
        clear_fermion!(vec_x[j])
    end

    vec_β = get_β_inverse(rhmc)
    vec_α = get_α_inverse(rhmc)
    α0 = get_α0_inverse(rhmc)
    shiftedcg(
        vec_x,
        vec_β,
        x,
        WdagW,
        ϕ,
        eps=W.eps_CG,
        maxsteps=W.MaxCGstep,
        verbose=W.verbose_print,
    )
    clear_fermion!(x)
    add_fermion!(x, α0, ϕ)
    for j = 1:N
        αk = vec_α[j]
        add_fermion!(x, αk, vec_x[j])
    end

    Sf = dot(x, x)

    unused!(fermi_action._temporary_fermionfields, it_x)
    unused!(fermi_action._temporary_fermionfields, its_vec_x)
    return real(Sf)
end

function evaluate_FermiAction(
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,8},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    #η = fermi_action._temporary_fermionfields[1]
    η, it_η = get_temp(fermi_action._temporary_fermionfields)#[1]
    solve_DinvX!(η, W', ϕ)
    Sf = dot(η, η)
    unused!(fermi_action._temporary_fermionfields, it_η)
    return real(Sf)
end

function evaluate_FermiAction(
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,4},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    η, it_η = get_temp(fermi_action._temporary_fermionfields)
    #η = fermi_action._temporary_fermionfields[1]
    solve_DinvX!(η, W', ϕ)
    Sf = dot(η, η)
    unused!(fermi_action._temporary_fermionfields, it_η)
    return real(Sf)
end


function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
) where {Dim,Dirac,fermion,gauge,Nf}
    gauss_distribution_fermion!(η)
end

function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,4},
) where {Dim,Dirac,fermion,gauge}
    evensite = false
    W = fermi_action.diracoperator(U)
    temp, it_temp = get_temp(fermi_action._temporary_fermionfields)
    #temp = fermi_action._temporary_fermionfields[1]
    gauss_distribution_fermion!(η)
    mul!(temp, W', η)
    clear_fermion!(temp, evensite)
    solve_DinvX!(η, W', temp)

    unused!(fermi_action._temporary_fermionfields, it_temp)
end

function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,8},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    mul!(ϕ, W', ξ)
    set_wing_fermion!(ϕ)
end

function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,4},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    mul!(ϕ, W', ξ)
    set_wing_fermion!(ϕ)
end


function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,Nf,fermion,gauge}
    W = fermi_action.diracoperator(U)

    if Nf == 4
        mul!(ϕ, W', ξ)
        set_wing_fermion!(ϕ)
    elseif Nf == 8
        mul!(ϕ, W', ξ)
        set_wing_fermion!(ϕ)
    else

        WdagW = DdagD_Staggered_operator(W)

        rhmc = fermi_action.rhmc_info_for_action

        N = get_order(rhmc)

        x = ϕ #fermi_action._temporary_fermionfields[1]
        vec_x, its_vec_x = get_temp(fermi_action._temporary_fermionfields, N)
        #vec_x = fermi_action._temporary_fermionfields[end-N+1:end]
        for j = 1:N
            clear_fermion!(vec_x[j])
        end

        vec_β = get_β(rhmc)
        vec_α = get_α(rhmc)
        α0 = get_α0(rhmc)
        #set_wing_fermion!(ξ)

        shiftedcg(
            vec_x,
            vec_β,
            x,
            WdagW,
            ξ,
            eps=W.eps_CG,
            maxsteps=W.MaxCGstep,
            verbose=W.verbose_print,
        )
        clear_fermion!(ϕ)
        add_fermion!(ϕ, α0, ξ)
        for j = 1:N
            αk = vec_α[j]
            add_fermion!(ϕ, αk, vec_x[j])
        end
        set_wing_fermion!(ϕ)

        unused!(fermi_action._temporary_fermionfields, its_vec_x)
    end

    #error("sample_pseudofermions!(ϕ,fermi_action) is not implemented in type ϕ:$(typeof(ϕ)), fermi_action:$(typeof(fermi_action))")
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,Nf,gauge}
    @assert Nf != 4 && Nf != 8
    clear_U!(UdSfdU)

    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Staggered_operator(W)

    rhmc = fermi_action.rhmc_info_for_MD
    N = get_order(rhmc)
    x, it_x = get_temp(fermi_action._temporary_fermionfields)
    #x = fermi_action._temporary_fermionfields[end-N]
    vec_x, its_vec_x = get_temp(fermi_action._temporary_fermionfields, N)
    #vec_x = fermi_action._temporary_fermionfields[end-N+1:end]
    #Y = fermi_action._temporary_fermionfields[2]
    Y, it_Y = get_temp(fermi_action._temporary_fermionfields)
    for j = 1:N
        clear_fermion!(vec_x[j])
    end
    vec_β = get_β_inverse(rhmc)
    vec_α = get_α_inverse(rhmc)
    #α0 = get_α0_inverse(rhmc)

    shiftedcg(
        vec_x,
        vec_β,
        x,
        WdagW,
        ϕ,
        eps=W.eps_CG,
        maxsteps=W.MaxCGstep,
        verbose=W.verbose_print,
    )

    for j = 1:N
        set_wing_fermion!(vec_x[j])

        calc_UdSfdU_fromX!(UdSfdU, Y, fermi_action, U, vec_x[j], coeff=vec_α[j])
    end

    set_wing_U!(UdSfdU)

    unused!(fermi_action._temporary_fermionfields, it_x)
    unused!(fermi_action._temporary_fermionfields, it_Y)
    unused!(fermi_action._temporary_fermionfields, its_vec_x)

    #error("calc_UdSfdU!(UdSfdU,fermi_action,U) is not implemented in type dSfdU:$(typeof(UdSfdU)), fermi_action:$(typeof(fermi_action)), ϕ:$(typeof(ϕ))")
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,4},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    #function cald_UdSfdU!(UdSfdU::Vector{<: AbstractGaugefields},fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,4},U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields) where {Dim,Dirac,fermion,gauge}
    calc_UdSfdU_Nf_4_8!(UdSfdU, fermi_action, U, ϕ)
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,8},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    calc_UdSfdU_Nf_4_8!(UdSfdU, fermi_action, U, ϕ)
end

function calc_UdSfdU_Nf_4_8!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge,Nf}
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Staggered_operator(W)
    X, it_X = get_temp(fermi_action._temporary_fermionfields)
    Y, it_Y = get_temp(fermi_action._temporary_fermionfields)
    #X = fermi_action._temporary_fermionfields[1]
    #Y = fermi_action._temporary_fermionfields[2]
    #X = (D^dag D)^(-1) ϕ 
    #
    solve_DinvX!(X, WdagW, ϕ)
    clear_U!(UdSfdU)

    calc_UdSfdU_fromX!(UdSfdU, Y, fermi_action, U, X)
    set_wing_U!(UdSfdU)
    unused!(fermi_action._temporary_fermionfields, it_X)
    unused!(fermi_action._temporary_fermionfields, it_Y)
end

function calc_UdSfdU_fromX!(
    UdSfdU::Vector{<:AbstractGaugefields},
    Y,
    fermi_action::StaggeredFermiAction{Dim,Dirac,fermion,gauge,Nf},
    U,
    X;
    coeff=1,
) where {Dim,Dirac,fermion,gauge,Nf}
    W = fermi_action.diracoperator(U)
    mul!(Y, W, X)

    #temp0_f = fermi_action._temporary_fermionfields[3]
    #temp0_g = fermi_action._temporary_gaugefields[1]
    temp0_f, it_temp0_f = get_temp(fermi_action._temporary_fermionfields)
    temp0_g, it_temp0_g = get_temp(fermi_action._temporary_gaugefields)

    #println(coeff)
    for μ = 1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X, μ)

        Us = staggered_U(U[μ], μ)
        mul!(temp0_f, Us, Xplus)
        #println(temp0_f[1,1,1,1,1,1])

        #U_{k,μ} X_{k+μ}) ⊗ Y_k
        mul!(temp0_g, temp0_f, Y')
        add_U!(UdSfdU[μ], coeff * 0.5, temp0_g)
        #println(temp2_g[1,1,1,1,1,1])
        #mul!(dSfdU[μ],U[μ]',temp2_g) #additional term

        #!  Construct P2*U_adj(x,mu)
        # Y_{k+μ}^dag U_{k,μ}^dag
        Yplus = shift_fermion(Y, μ)
        mul!(temp0_f, Yplus', Us')

        #X_k ⊗ Y_{k+μ}^dag U_{k,μ}^dag
        mul!(temp0_g, X, temp0_f)
        add_U!(UdSfdU[μ], coeff * 0.5, temp0_g)

        #mul!(temp3_g,U[μ]',temp2_g)
        #add_U!(dSfdU[μ],temp3_g)
    end
    unused!(fermi_action._temporary_fermionfields, it_temp0_f)
    unused!(fermi_action._temporary_gaugefields, it_temp0_g)

end
