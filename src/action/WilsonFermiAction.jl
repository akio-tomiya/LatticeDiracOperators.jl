import Gaugefields: Traceless_antihermitian_add!, Generator
import Gaugefields.Temporalfields_module: Temporalfields, unused!, get_temp

#include("clover_data.jl")

abstract type Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} <:
              FermiAction{Dim,Dirac,fermion,gauge} end

struct WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm} <:
       Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} #FermiAction{Dim,Dirac,fermion,gauge}
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    hascloverterm::Bool
    # clover_data::Union{Nothing,Wilsonclover_data}
    _temporary_fermionfields::Temporalfields{fermion}# Vector{fermion}
    _temporary_gaugefields::Temporalfields{gauge}# Vector{gauge}
    SUNgenerator::Union{Nothing,Generator}

    function WilsonFermiAction(
        D::Dirac_operator{Dim},
        hascovnet,
        covneuralnet,
        parameters_action,
    ) where {Dim}
        hascloverterm = has_cloverterm(D)
        #println(hascloverterm)
        #hascloverterm = check_parameters(parameters_action, "hascloverterm", false)
        numU = 2
        num = 6
        NC = D._temporary_fermi[1].NC
        numbasis = NC^2 - 1
        if hascloverterm
            numU += 9 + 2numbasis
            num += 4 #+ 2numbasis
            SUNgenerator = Generator(NC)
        else
            SUNgenerator = nothing
        end
        #if hascloverterm
        #NC,_,NN... = size(D.U[1])
        #NV = prod(NN)
        #clover_data = Wilsonclover_data(D,NV)            
        #numU += 2
        #    error("not implemented yet.")
        #else
        #    clover_data = nothing

        #end



        #x = D._temporary_fermi[1]
        x, it_x = get_temp(D._temporary_fermi)

        xtype = typeof(x)
        _temporary_fermionfields = Temporalfields(x; num)# Array{xtype,1}(undef, num)
        #for i = 1:num
        #    _temporary_fermionfields[i] = similar(x)
        #end

        Utemp = D.U[1]
        Utype = typeof(Utemp)

        _temporary_gaugefields = Temporalfields(Utemp; num=numU)#Array{Utype,1}(undef, numU)
        #for i = 1:numU
        #    _temporary_gaugefields[i] = similar(Utemp)
        #end


        return new{Dim,typeof(D),xtype,Utype,hascloverterm}(
            hascovnet,
            covneuralnet,
            D,
            hascloverterm,
            #clover_data,
            _temporary_fermionfields,
            _temporary_gaugefields,
            SUNgenerator
        )

    end
end

function evaluate_FermiAction(
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
    U,
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    η, it_eta = get_temp(fermi_action._temporary_fermionfields)#fermi_action._temporary_fermionfields[1]
    solve_DinvX!(η, W', ϕ)
    Sf = dot(η, η)
    unused!(fermi_action._temporary_fermionfields, it_eta)
    return real(Sf)
end

function calc_UdSfdU!(
    UdSfdU::Vector{<:AbstractGaugefields},
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}

    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Wilson_operator(W)
    X, it_X = get_temp(fermi_action._temporary_fermionfields)
    Y, it_Y = get_temp(fermi_action._temporary_fermionfields)
    #X = fermi_action._temporary_fermionfields[end]
    #Y = fermi_action._temporary_fermionfields[2]
    #X = (D^dag D)^(-1) ϕ 
    #
    #println("Xd ",X[1,1,1,1,1,1])
    #ϕ.f .= 1
    #println("solve DinvX")
    solve_DinvX!(X, WdagW, ϕ)
    #error("ee")
    #println("X ",X[1,1,1,1,1,1])
    clear_U!(UdSfdU)

    calc_UdSfdU_fromX!(UdSfdU, Y, fermi_action, U, X)
    #println("----aa--")
    set_wing_U!(UdSfdU)
    unused!(fermi_action._temporary_fermionfields, it_X)
    unused!(fermi_action._temporary_fermionfields, it_Y)
end

function calc_UdSfdU_fromX!(
    UdSfdU::Vector{<:AbstractGaugefields},
    Y,
    fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm},
    U,
    X;
    coeff=1,
) where {Dim,Dirac,fermion,gauge,hascloverterm}
    W = fermi_action.diracoperator(U)
    mul!(Y, W, X)
    #set_wing_fermion!(Y)

    temp0_f, it_temp0_f = get_temp(fermi_action._temporary_fermionfields)
    temp1_f, it_temp1_f = get_temp(fermi_action._temporary_fermionfields)
    #temp0_f = fermi_action._temporary_fermionfields[3]
    #temp1_f = fermi_action._temporary_fermionfields[4]
    temp0_g, it_temp0_g = get_temp(fermi_action._temporary_gaugefields)
    #temp0_g = fermi_action._temporary_gaugefields[1]

    if hascloverterm
        D = fermi_action.diracoperator
        hop = D.κ
        Clover_coefficient = D.cloverterm.cSW
        coe = im * 0.125 * hop * Clover_coefficient
    end


    for μ = 1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X, μ)

        #@time mul!(temp0_f,U[μ],X)

        mul!(temp0_f, U[μ], Xplus)


        # (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp1_f, view(W.rminusγ, :, :, μ), temp0_f)

        # κ (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp0_f, W.hopp[μ], temp1_f)

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        mul!(temp0_g, temp0_f, Y')

        add_U!(UdSfdU[μ], -coeff, temp0_g)

        #!  Construct P2*U_adj(x,mu)
        # Y_{k+μ}^dag U_{k,μ}^dag
        Yplus = shift_fermion(Y, μ)
        mul!(temp0_f, Yplus', U[μ]')

        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp1_f, temp0_f, view(W.rplusγ, :, :, μ))

        # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f, W.hopm[μ], temp1_f)

        # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_g, X, temp0_f)

        add_U!(UdSfdU[μ], coeff, temp0_g)

        if hascloverterm
            clear_U!(temp0_g)
            dSclover!(temp0_g, μ, X, Y, U, fermi_action)
            #println(sum(abs.(temp0_g.U)))
            add_U!(UdSfdU[μ], -coeff * 0, temp0_g)
            #add!(p[μ],-τ*mdparams.Δτ,c)
        end

    end

    unused!(fermi_action._temporary_fermionfields, it_temp0_f)
    unused!(fermi_action._temporary_fermionfields, it_temp1_f)
    unused!(fermi_action._temporary_gaugefields, it_temp0_g)

end




function calc_p_UdSfdU!(
    p,
    fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm},
    U::Vector{<:AbstractGaugefields},
    ϕ::AbstractFermionfields,
    coeff=1,
) where {Dim,Dirac,fermion,gauge,hascloverterm}
    #println("------dd")
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Wilson_operator(W)
    X = fermi_action._temporary_fermionfields[end]
    Y = fermi_action._temporary_fermionfields[2]
    #X = (D^dag D)^(-1) ϕ 
    #
    #println("Xd ",X[1,1,1,1,1,1])
    solve_DinvX!(X, WdagW, ϕ)
    #println("X ",X[1,1,1,1,1,1])
    #clear_U!(UdSfdU)

    calc_p_UdSfdU_fromX!(p, Y, fermi_action, U, X, coeff=coeff)
    #println("----aa--")
    #set_wing_U!(UdSfdU)
end

function calc_p_UdSfdU_fromX!(
    p,
    Y,
    fermi_action::WilsonFermiAction{Dim,Dirac,fermion,gauge,hascloverterm},
    U,
    X;
    coeff=1,
) where {Dim,Dirac,fermion,gauge,hascloverterm}
    W = fermi_action.diracoperator(U)
    mul!(Y, W, X)
    #set_wing_fermion!(Y)

    temp0_f = fermi_action._temporary_fermionfields[3]
    temp1_f = fermi_action._temporary_fermionfields[4]
    temp0_g = fermi_action._temporary_gaugefields[1]

    for μ = 1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X, μ)

        #@time mul!(temp0_f,U[μ],X)

        mul!(temp0_f, U[μ], Xplus)


        # (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp1_f, view(W.rminusγ, :, :, μ), temp0_f)

        # κ (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp0_f, W.hopp[μ], temp1_f)

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        mul!(temp0_g, temp0_f, Y')

        Traceless_antihermitian_add!(p[μ], -coeff, temp0_g)

        #add_U!(UdSfdU[μ],-coeff,temp0_g)

        #!  Construct P2*U_adj(x,mu)
        # Y_{k+μ}^dag U_{k,μ}^dag
        Yplus = shift_fermion(Y, μ)
        mul!(temp0_f, Yplus', U[μ]')

        # Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp1_f, temp0_f, view(W.rplusγ, :, :, μ))

        # κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_f, W.hopm[μ], temp1_f)

        # X_k ⊗ κ Y_{k+μ}^dag U_{k,μ}^dag*(r+γ_μ)
        mul!(temp0_g, X, temp0_f)

        Traceless_antihermitian_add!(p[μ], coeff, temp0_g)

        #add_U!(UdSfdU[μ],coeff,temp0_g)


    end


end


function gauss_sampling_in_action!(
    η::AbstractFermionfields,
    U,
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
) where {Dim,Dirac,fermion,gauge}
    #gauss_distribution_fermion!(η)
    #gauss_distribution_fermion!(η, rand)
    gauss_distribution_fermion!(η)
end

function gauss_sampling_in_action!(
    η::WilsonFermion_4D_accelerator,
    U,
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
) where {Dim,Dirac,fermion,gauge}
    #gauss_distribution_fermion!(η)
    #gauss_distribution_fermion!(η, rand)
    gauss_distribution_fermion!(η)
end





using InteractiveUtils

function sample_pseudofermions!(
    ϕ::AbstractFermionfields,
    U,
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
    ξ::AbstractFermionfields,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)

    mul!(ϕ, W', ξ)

    set_wing_fermion!(ϕ)
end

function sample_pseudofermions!(
    ϕ::WilsonFermion_4D_accelerator,
    U,
    fermi_action::Wilsontype_FermiAction{Dim,Dirac,fermion,gauge},
    ξ::WilsonFermion_4D_accelerator,
) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    ik = findfirst(x -> isnan(x), ξ.f)
    if ik != nothing
        display(ξ.f[ik])
        error("ddξ")
    end

    ik = findfirst(x -> isnan(x), W.U[1].U)
    if ik != nothing
        println(ik)
        display(W.U[ik])
        error("ddW")
    end

    mul!(ϕ, W', ξ)
    ik = findfirst(x -> isnan(x), ϕ.f)
    if ik != nothing
        println(ik)
        display(ϕ.f[ik])
        error("ddϕ")
    end

    set_wing_fermion!(ϕ)
end


function UdScloverdU(z, μ, X, Y, U, fermi_action::WilsonFermiAction{Dim,Dirac,
    fermion,gauge,hascloverterm}) where {Dim,Dirac,fermion,gauge,hascloverterm}
    @assert Dim == 4 "only Dim =4 is supported!"
    D = fermi_action.diracoperator
    cloverterm = D.cloverterm

end

"""
Calculate   dS_clover/dA_mu(x)
"""
function dSclover!(z, μ, X, Y, U, fermi_action::WilsonFermiAction{Dim,Dirac,
    fermion,gauge,hascloverterm}) where {Dim,Dirac,fermion,gauge,hascloverterm}
    @assert Dim == 4 "only Dim =4 is supported!"
    D = fermi_action.diracoperator
    cloverterm = D.cloverterm
    clear_U!(z)


    NX = X.NX
    NY = X.NY
    NZ = X.NZ
    NT = X.NT
    NC = X.NC
    NV = NX * NY * NZ * NT
    numbasis = NC^2 - 1
    temps = fermi_action._temporary_gaugefields
    #numbasis = ifelse(NC==3,8,3)


    work1 = temps[2]
    work2 = temps[3]
    work3 = temps[4]
    work4 = temps[5]

    gtmp1 = temps[6]
    gtmp2 = temps[7]
    gtmp3 = temps[8]
    gtmp4 = temps[9]

    dF1 = temps[8+1:8+numbasis]
    #dF1 = temps[9+1]
    #dF1 = temps[9:16]
    dF2 = temps[8+numbasis+1:8+2numbasis]
    #dF2 = temps[9+2]
    #dF2 = temps[17:24]




    #if D.cloverterm.internal_flags[1] ==  false
    #    fprep3!(NX,NY,NZ,NT,D.cloverterm.inn_table)
    #    D.cloverterm.internal_flags[1] = true
    #end

    clear_U!(z)
    veta = fermi_action._temporary_fermionfields[end-5] #cloverterm._ftmp_vectors[1]
    vxi = fermi_action._temporary_fermionfields[end-4] #cloverterm._ftmp_vectors[2]
    ftmp1 = fermi_action._temporary_fermionfields[end-3] #cloverterm._ftmp_vectors[3]
    ftmp2 = fermi_action._temporary_fermionfields[end-2] #cloverterm._ftmp_vectors[4]
    v1 = D._temporary_fermi[end-1] #cloverterm._ftmp_vectors[5]
    v2 = D._temporary_fermi[end] #cloverterm._ftmp_vectors[6]
    #is1 = cloverterm._is1
    #is2 = cloverterm._is2
    #inn = cloverterm.inn_table

    substitute_fermion!(veta, Y)
    substitute_fermion!(vxi, X)

    #=
    for ν=1:4
        #for ν=μ:4
        if ν==μ
            continue
        end



        substitute_U!(work1,U[μ])
        #substitute!(work1,U[μ])
        U2 = shift_U(U[ν], μ)
        #gauge_shift!(work2,μ,U[ν])
        U3 = shift_U(U[μ],ν)
        #gauge_shift!(work3,ν,U[μ])
        substitute_U!(work4,U[ν])

    #=
                                        w3
                                    .-----------+
            Case 1                  |           |
                nu                 |           |
                    |              w4 |           | w2
                    |                 |           |
                    +----> mu         o-----------. 
                                    x    w1 
    =#
        mul!(gtmp1,work1,U2) #U1U2
        mul!(gtmp2,work4,U3) #U4U3

        mul!(UV,gtmp1,gtmp2') #U1U2 U3'U4'

        # Y^dag  U V sigma X #loop1
        #sigma_{munu}Y
        apply_σμν!(ftmp1,μ,ν,X) 
        #UV sigma X
        mul!(ftmp2,UV,ftmp1)
        mul!(gtmp1,ftmp2,Y')
        add_U!(z,gtmp1)

        #loop1'

        Xnuminus = shift_fermion(X,-ν)
        Ynuminus = shift_fermion(Y,-ν)
        U1 = shift_U(U[ν], -ν)
        U2 = work1
        shift = zeros(Int64,4)
        shift[μ] = 1
        shift[ν] = -1 
        U3 = shift_U(U[ν], Tuple(shift))
        U4 = shift_U(U[μ], -ν)

        apply_σμν!(ftmp1,μ,ν,Ynuminus)
        mul!(gtmp1,U4,U3) #U4 U3
        mul!(gtmp2,U2,gtmp1') #U2 U3' U4'
        mul!(ftmp2,gtmp2,ftmp1) #U2 U3' U4' sigma Yminus

        mul!(ftmp1,U1',Xnuminus) #A^dag X
        mul!(gtmp1,ftmp2,ftmp1') #(A^dag X)^dag
        add_U!(z,gtmp1)


    end


    return
    =#
    #=
    for ialpha = 1:4
        for ic=1:NC
            is = 0
            for it=1:NT
                for iz=1:NZ
                    for iy=1:NY
                        for ix=1:NX
                            is += 1
                            veta[ic,is,ialpha] = Y[ic,ix,iy,iz,it,ialpha]
                            vxi[ic,is,ialpha] = X[ic,ix,iy,iz,it,ialpha]
                        end
                    end
                end
            end
        end
    end
    =#



    for ν = 1:4
        #for ν=μ:4
        if ν == μ
            continue
        end

        #=
         .....................  
            Case 1 and 3        
         .....................  
        =#
        iflag = 1
        cal_dFμν!(dF1, dF2,
            U, fermi_action,
            work1, work2, work3, work4,
            gtmp1, gtmp2, gtmp3, gtmp4,
            μ, ν, iflag)

        # ... Case 1
        #println("z = ",sum(abs.(z.U)))
        for ia = 1:numbasis
            VxSigxV!(veta, vxi, dF1, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end
        #println("z1 = ",z*z)
        #println(sum(abs.(z.U)))

        #=
                                       w3
                                  .<----------o
          Case 3                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         .-----------. 

        =#
        # -> nu,mu shift!
        #=
         for is=1:NV
                is1[is] = inn[is,μ,1]
            end

            for is=1:NV
                is2[is] = inn[is1[is],ν,1]
            end
        =#
        shift = zeros(Int64, 4)
        shift[μ] = 1
        shift[ν] = 1
        veta_shift = shift_fermion(veta, Tuple(shift))
        vxi_shift = shift_fermion(vxi, Tuple(shift))

        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF2, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end


        #println("z2 = ",z*z)

        #=
         .....................  
            Case 2 and 4        
         .....................  
        =#

        iflag = 2

        cal_dFμν!(dF1, dF2,
            U, fermi_action,
            work1, work2, work3, work4,
            gtmp1, gtmp2, gtmp3, gtmp4,
            μ, ν, iflag)
        #=
                                                w3
                                          .-----------+
                Case 2                    |           |
                    nu                    |           |
                        |             w4  |           | w2
                        |                 |           |
                        +----> mu         .---------->o 
                                            x    w1 
        =#
        # ... Case 2
        # mu shift
        #=
         for is=1:NV
                is1[is] = inn[is,μ,1]
            end
        =#

        veta_shift = shift_fermion(veta, μ)
        vxi_shift = shift_fermion(vxi, μ)

        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF1, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end

        # ... Case 4
        #=
        for is=1:NV
                is1[is] = inn[is,ν,1]
            end
        =#
        veta_shift = shift_fermion(veta, ν)
        vxi_shift = shift_fermion(vxi, ν)

        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF2, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end

        #println("z4 = ",z*z)

        #=
         .....................  
            Case 4' and 2'      
         .....................  
        =#

        iflag = 3

        cal_dFμν!(dF1, dF2,
            U, fermi_action,
            work1, work2, work3, work4,
            gtmp1, gtmp2, gtmp3, gtmp4,
            μ, ν, iflag)

        # ... Case 4'
        for ia = 1:numbasis
            VxSigxV!(veta, vxi, dF1, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end
        #println("z5 = ",z*z)

        # ... Case 2' 
        shift = zeros(Int64, 4)
        shift[μ] = 1
        shift[ν] = -1
        veta_shift = shift_fermion(veta, Tuple(shift))
        vxi_shift = shift_fermion(vxi, Tuple(shift))

        #=
        for is=1:NV
            is1[is] = inn[is,μ,1]
        end

        for is=1:NV
            is2[is] = inn[is1[is],ν,2]
        end
        =#



        #=
        for ic=1:NC
            for ialpha=1:4
                for is=1:NV
                    v1[ic,is,ialpha] = veta[ic,is2[is],ialpha]
                    v2[ic,is,ialpha] = vxi[ic,is2[is],ialpha]
                end
            end
        end
        =#
        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF2, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end
        #println("z6 = ",z*z)

        #=
         .....................  
            Case 3' and 1'     
         .....................  
        =#
        iflag = 4

        cal_dFμν!(dF1, dF2,
            U, fermi_action,
            work1, work2, work3, work4,
            gtmp1, gtmp2, gtmp3, gtmp4,
            μ, ν, iflag)




        # ... Case 3'
        veta_shift = shift_fermion(veta, μ)
        vxi_shift = shift_fermion(vxi, μ)

        #for is=1:NV
        #     is1[is] = inn[is,μ,1]
        # end

        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF1, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end


        # ... Case 1' 
        veta_shift = shift_fermion(veta, -ν)
        vxi_shift = shift_fermion(vxi, -ν)
        #=
        for is=1:NV
            is1[is] = inn[is,ν,2]
        end
        =#
        for ia = 1:numbasis
            VxSigxV!(veta_shift, vxi_shift, dF2, z, ftmp1, ftmp2, μ, ν, gtmp1, ia)
        end


    end


end

function VxSigxV!(v1, v2, u, z, tmp1, tmp2, μ, ν, gtmp1, ia)

    apply_σμν!(tmp2, μ, ν, v2)
    mul!(tmp1, u[ia], tmp2)
    #println("VxSigxV! ",sum(abs.(tmp1.f))," z $(sum(abs.(z.U)))")
    mul!(gtmp1, v1', tmp1)
    add_U!(z, gtmp1)

end

function cal_dFμν_old!(dFμν1, dFμν2, U::Array{<:AbstractGaugefields{NC},1}, fermi_action, work1, work2, work3, work4, temp1, temp2, temp3, temp4, μ, ν, iflag) where {NC}
    #function calc_dFμν!(dFμν1,dFμν2,U::Array{<:AbstractGaugefields{NC},1},fermi_action,μ,ν,iflag,work1,work4)
    D = fermi_action.diracoperator
    hop = D.κ
    Clover_coefficient = D.cloverterm.cSW
    coe = im * 0.125 * hop * Clover_coefficient
    #println("coe ",coe)
    #=
    ...  A) the upper staple  ....................
                               w3
       nu                 .-----------+
                          |           | 
       /|             w4  |           | w2
        |                 |           |
        |                 |           |
        +----> mu         .-----------.

    =#
    if iflag == 1 || iflag == 2
        substitute_U!(work1, U[μ])
        #substitute!(work1,U[μ])
        U2 = shift_U(U[ν], μ)
        #gauge_shift!(work2,μ,U[ν])
        U3 = shift_U(U[μ], ν)
        #gauge_shift!(work3,ν,U[μ])
        substitute_U!(work4, U[ν])
        #substitute!(work4,U[ν])
    end

    # ...  iflag = 1 (Case 1 and 3)
    if iflag == 1
        #=
                                       w3
                                  .-----------+
          Case 1                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         o-----------. 
                                  x    w1 

                                       w3
                                  .<----------o
          Case 3                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         .-----------. 

        =#
        mul!(temp1, work1, U2) #U1U2
        mul!(temp2, work4, U3) #U4U3

        for ia = 1:numbasis
            lambdamul(temp3, temp1, ia, fparam.SUNgenerator)
            mul!(temp4, temp3, temp2')
            #mul!(temp4,im)
            Antihermitian!(temp4, factor=im)
            #cimaglnk!(temp4)
            substitute!(dFμν1[ia], coe, temp4)

            mul!(temp4, temp2', temp3)
            Antihermitian!(temp4, factor=im)
            #mul!(temp4,im)
            #cimaglnk!(temp4)
            substitute!(dFμν2[ia], coe, temp4)

        end


        #mul!(temp4,temp1,temp2') #U1U2 U3'U4'
        #mul!(dFμν1,coe,temp4)


        #mul!(temp4,temp2',temp1) #U3' U4' U1 U2
        #mul!(dFμν2,coe,temp4)
    end

    # ...  iflag = 2 (Case 2 and 4)
    if iflag == 2
        #=
                                        w3
                                  .-----------+
        Case 2                    |           |
            nu                    |           |
                |             w4  |           | w2
                |                 |           |
                +----> mu         .---------->o 
                                    x    w1 

                                        w3
                                  o-----------+
            Case 4                |           |
                nu                |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         .-----------. 
                                    x    w1 
        =#
        mul!(temp1, U2, U3') #U2 U3'

        for ia = 1:numbasis
            lambdamul(temp2, work1, ia, fparam.SUNgenerator) # temp2=(lambda_a/2)*work1
            mul!(temp3, work4', temp2)
            mul!(temp4, temp1, temp3)
            Antihermitian!(temp4, factor=im)
            #mul!(temp4,im)
            cimaglnk!(temp4)
            substitute!(dFμν1[ia], coe, temp4)

            mul!(temp4, temp3, temp1)
            Antihermitian!(temp4, factor=im)
            #mul!(temp4,im)
            cimaglnk!(temp4)
            substitute!(dFμν2[ia], coe, temp4)
        end


        #=
        mul!(temp3,work4',work1) #U4' U1
        mul!(temp4,temp1,temp3) #U2 U3' U4' U1
        mul!(dFμν1,coe,temp4)

        mul!(temp4,temp3,temp1) #U4' U1 U2 U3' 
        mul!(dFμν2,coe,temp4)
        =#
    end


    #=
        ...  B) the lower staple  ....................

           nu
           /|
            |
            |
            |                 x    w1
            +----> mu         .-----------+
                              |           |
                          w4  |           | w2
                              |           |
                              |           |
                              .-----------.

    =#
    if iflag == 3 || iflag == 4
        substitute_U!(work1, U[μ])
        #substitute!(work1,U[μ])
        shift = zeros(Int64, 4)
        shift[μ] = 1
        shift[ν] = -1
        U2 = shift_U(U[ν], Tuple(shift))
        #idir2= (μ,-ν)
        #gauge_shift!(work2,idir2,U[ν])
        U3 = shift_U(U[μ], -ν)
        U4 = shift_U(U[ν], -ν)
        #gauge_shift!(work3,-ν,U[μ])
        #gauge_shift!(work4,-ν,U[ν])
        #println(work2[1,1,1])
        #exit()
    end

    #   ...  iflag = 3 (Case 4' and 2')
    if iflag == 3
        #=
          Case 4'
               nu
                |
                |                 x    w1
                +----> mu         o<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------.
                                       w3
          Case 2'
               nu
                |
                |                 x    w1
                +----> mu         .<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------o
                                       w3
        =#

        mul!(temp1, U4', U3) #U4' U3

        mul!(temp3, U2, work1') # U2 U1'
        mul!(temp4, temp1, temp3) #U4' U3 U2 U1'
        mul!(dFμν1, -coe, temp4)

        mul!(temp4, temp3, temp1) #U2 U1' U4' U3
        mul!(dFμν2, -coe, temp4)


    end

    # ...  iflag = 4 (Case 3' and 1')
    if iflag == 4
        #=
          Case 3' 
               nu
                |
                |                 x    w1
                +----> mu         .<----------o 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------.
                                       w3
          Case 1' 
               nu
                |
                |                 x    w1
                +----> mu         .<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  o-----------.

        =#

        mul!(temp1, U3, U2) #U3 U2

        mul!(temp3, U4, work1)  #U4 U1
        mul!(temp4, temp3', temp1) #U1' U4' U3 U2
        mul!(dFμν1, -coe, temp4)

        mul!(temp4, temp1, temp3') #U3 U2 U1' U4'
        mul!(dFμν2, -coe, temp4)

    end



end


function cal_dFμν!(dFμν1, dFμν2, U::Array{<:AbstractGaugefields{NC},1}, fermi_action, work1, work2, work3, work4, temp1, temp2, temp3, temp4, μ, ν, iflag) where {NC}
    D = fermi_action.diracoperator
    hop = D.κ
    Clover_coefficient = D.cloverterm.cSW
    coe = im * 0.125 * hop * Clover_coefficient
    #numbasis = ifelse(NC == 3,8,3)
    numbasis = NC^2 - 1


    #=
        ...  A) the upper staple  ....................
                                   w3
           nu                 .-----------+
                              |           | 
           /|             w4  |           | w2
            |                 |           |
            |                 |           |
            +----> mu         .-----------.

    =#
    if iflag == 1 || iflag == 2
        substitute_U!(work1, U[μ])
        #substitute!(work1,U[μ])
        U2 = shift_U(U[ν], μ)
        #gauge_shift!(work2,μ,U[ν])
        U3 = shift_U(U[μ], ν)
        #gauge_shift!(work3,ν,U[μ])
        substitute_U!(work4, U[ν])
        #substitute!(work4,U[ν])
    end

    # ...  iflag = 1 (Case 1 and 3)
    if iflag == 1
        #=
                                       w3
                                  .-----------+
          Case 1                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         o-----------. 
                                  x    w1 

                                       w3
                                  .<----------o
          Case 3                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         .-----------. 

        =#
        mul!(temp1, work1, U2)
        mul!(temp2, work4, U3)



        for ia = 1:numbasis
            Gaugefields.lambda_k_mul!(temp3, temp1, ia, fermi_action.SUNgenerator)
            #lambdamul(temp3,temp1,ia,fermi_action.SUNgenerator)
            #mul!(temp4,temp3,temp2')
            mul!(temp4, temp3, temp2')
            mul!(temp3, im, temp4)
            #mul!(temp4,im)
            Antihermitian!(dFμν1[ia], temp3, factor=coe)
            #cimaglnk!(temp4)
            #substitute!(dFμν1[ia],coe,temp4)

            mul!(temp4, temp2', temp3)
            mul!(temp3, im, temp4)
            #mul!(temp4,im)
            #cimaglnk!(temp4)
            Antihermitian!(dFμν2[ia], temp3, factor=coe)
            #substitute!(dFμν2[ia],coe,temp4)

        end


    end
    # ...  iflag = 2 (Case 2 and 4)
    if iflag == 2
        #=
                                        w3
                                  .-----------+
          Case 2                  |           |
               nu                 |           |
                |             w4  |           | w2
                |                 |           |
                +----> mu         .---------->o 
                                  x    w1 

                                        w3
                                  o-----------+
          Case 4                  |           |
               nu                 |           |
                |              w4 |           | w2
                |                 |           |
                +----> mu         .-----------. 
                                  x    w1 
        =#
        mul!(temp1, U2, U3')

        for ia = 1:numbasis
            #lambdamul(temp2,work1,ia,fparam.SUNgenerator) # temp2=(lambda_a/2)*work1
            Gaugefields.lambda_k_mul!(temp2, work1, ia, fermi_action.SUNgenerator)
            mul!(temp3, work4', temp2)
            mul!(temp4, temp1, temp3)
            mul!(temp2, im, temp4)
            #mul!(temp4,im)
            Antihermitian!(dFμν1[ia], temp2, factor=coe)
            #cimaglnk!(temp4)
            #substitute!(dFμν1[ia],coe,temp4)

            mul!(temp4, temp3, temp1)
            mul!(temp2, im, temp4)
            #mul!(temp4,im)
            Antihermitian!(dFμν2[ia], temp2, factor=coe)
            #cimaglnk!(temp4)
            #substitute!(dFμν2[ia],coe,temp4)

        end

    end

    #=
        ...  B) the lower staple  ....................

           nu
           /|
            |
            |
            |                 x    w1
            +----> mu         .-----------+
                              |           |
                          w4  |           | w2
                              |           |
                              |           |
                              .-----------.

    =#
    if iflag == 3 || iflag == 4
        substitute_U!(work1, U[μ])
        #substitute!(work1,U[μ])
        shift = zeros(Int64, 4)
        shift[μ] = 1
        shift[ν] = -1
        U2 = shift_U(U[ν], Tuple(shift))
        #idir2= (μ,-ν)
        #gauge_shift!(work2,idir2,U[ν])
        U3 = shift_U(U[μ], -ν)
        U4 = shift_U(U[ν], -ν)
        #gauge_shift!(work3,-ν,U[μ])
        #gauge_shift!(work4,-ν,U[ν])
        #println(work2[1,1,1])
        #exit()
    end
    #   ...  iflag = 3 (Case 4' and 2')
    if iflag == 3
        #=
          Case 4'
               nu
                |
                |                 x    w1
                +----> mu         o<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------.
                                       w3
          Case 2'
               nu
                |
                |                 x    w1
                +----> mu         .<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------o
                                       w3
        =#

        mul!(temp1, U4', U3)

        for ia = 1:numbasis
            Gaugefields.lambda_k_mul!(temp2, work1, ia, fermi_action.SUNgenerator)
            #lambdamul(temp2,work1,ia,fparam.SUNgenerator) # temp2=(lambda_a/2)*work1
            mul!(temp3, U2, temp2')

            mul!(temp4, temp1, temp3)
            mul!(temp2, -im, temp4)
            Antihermitian!(dFμν1[ia], temp2, factor=coe)

            #mul!(temp4,-im)
            #cimaglnk!(temp4)
            #substitute!(dFμν1[ia],coe,temp4)

            mul!(temp4, temp3, temp1)
            mul!(temp2, -im, temp4)
            Antihermitian!(dFμν2[ia], temp2, factor=coe)
            #mul!(temp4,-im)
            #cimaglnk!(temp4)
            #substitute!(dFμν2[ia],coe,temp4)
        end
    end

    # ...  iflag = 4 (Case 3' and 1')
    if iflag == 4
        #=
          Case 3' 
               nu
                |
                |                 x    w1
                +----> mu         .<----------o 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  .-----------.
                                       w3
          Case 1' 
               nu
                |
                |                 x    w1
                +----> mu         .<----------. 
                                  |           |
                              w4  |           | w2
                                  |           |
                                  |           |
                                  o-----------.

        =#

        mul!(temp1, U3, U2)

        for ia = 1:numbasis
            Gaugefields.lambda_k_mul!(temp2, work1, ia, fermi_action.SUNgenerator)
            #lambdamul(temp2,work1,ia,fparam.SUNgenerator) # temp2=(lambda_a/2)*work1
            mul!(temp3, U4, temp2)

            mul!(temp4, temp3', temp1)
            mul!(temp2, -im, temp4)
            #mul!(temp4,-im)
            #cimaglnk!(temp4)
            #println(temp4.g)
            #exit()
            Antihermitian!(dFμν1[ia], temp2, factor=coe)

            #substitute!(dFμν1[ia],coe,temp4)

            mul!(temp4, temp1, temp3')
            mul!(temp2, -im, temp4)
            #mul!(temp4,-im)
            Antihermitian!(dFμν2[ia], temp2, factor=coe)
            #cimaglnk!(temp4)

            #substitute!(dFμν2[ia],coe,temp4)
        end
    end
    return



end
