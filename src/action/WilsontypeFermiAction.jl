struct Wilson_GeneralDirac_FermiAction{Dim,Dirac,fermion,gauge} <: Wilsontype_FermiAction{Dim,Dirac,fermion,gauge} 
    hascovnet::Bool
    covneuralnet::Union{Nothing,CovNeuralnet{Dim}}
    diracoperator::Dirac
    _temporary_fermionfields::Vector{fermion}
    _temporary_gaugefields::Vector{gauge}

    function Wilson_GeneralDirac_FermiAction(D::Dirac_operator{Dim},hascovnet,covneuralnet,parameters_action) where Dim

        num = 8

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


        return new{Dim,typeof(D),xtype,Utype}(hascovnet,covneuralnet,D,
                        _temporary_fermionfields,
                        _temporary_gaugefields)

    end
end

function calc_dSfdU!(dSfdU::Vector{<: AbstractGaugefields},fermi_action::Wilson_GeneralDirac_FermiAction{Dim,Dirac,fermion,gauge} ,U::Vector{<: AbstractGaugefields},ϕ::AbstractFermionfields) where  {Dim,Dirac,fermion,gauge,hascloverterm} 
    #println("------dd")
    W = fermi_action.diracoperator(U)
    WdagW = DdagD_Wilson_GeneralDirac_operator(W)
    X = fermi_action._temporary_fermionfields[5]
    Y = fermi_action._temporary_fermionfields[6]
    #X = (D^dag D)^(-1) ϕ 
    #
    #println("Xd ",X[1,1,1,1,1,1])
    solve_DinvX!(X,WdagW,ϕ)
    #println("X ",X[1,1,1,1,1,1])
    clear_U!(dSfdU)

    calc_dSfdU_fromX!(dSfdU,Y,fermi_action,U,X) 
    #println("----aa--")
    set_wing_U!(dSfdU)
end




function calc_dSfdU_fromX!(dSfdU::Vector{<: AbstractGaugefields},Y,fermi_action::Wilson_GeneralDirac_FermiAction{Dim,Dirac,fermion,gauge} ,U,X;coeff = 1) where {Dim,Dirac,fermion,gauge}
    W = fermi_action.diracoperator(U)
    mul!(Y,W,X)
    #sset_wing_fermion!(Y)

    temp0_f = fermi_action._temporary_fermionfields[3]
    temp1_f = fermi_action._temporary_fermionfields[4]
    temp0_g = fermi_action._temporary_gaugefields[end]
    #temp1_g = fermi_action._temporary_gaugefields[2]


    numterms = get_numterms(W)

    
    #=
    
    for μ=1:Dim
        #!  Construct U(x,mu)*P1

        # U_{k,μ} X_{k+μ}
        Xplus = shift_fermion(X,μ)
        #@time mul!(temp0_f,U[μ],X)
        diracterm = W[1]
        coeff_i = get_coefficient(diracterm)

        mul!(temp1_f,rminusγ1[:,:,μ],Xplus) #Xplus Y^dag
    
        # κ (r-γ_μ) U_{k,μ} X_{k+μ}
        mul!(temp0_f,coeff_i,temp1_f)

        # κ ((r-γ_μ) U_{k,μ} X_{k+μ}) ⊗ Y_k
        cross!(temp0_g,Y',temp0_f)
        add_U!(dSfdU[μ],-coeff,temp0_g)

        Yplus = shift_fermion(Y,μ) #X   Yplus^dag U_{k,μ}^dag * rplusγ1[:,:,μ]
        mul!(temp1_f,rplusγ1[:,:,μ],Yplus)
        mul!(temp0_f,coeff_i,temp1_f)

        cross!(temp0_g,X',temp0_f)
        add_U!(dSfdU[μ],-coeff,temp0_g)
        #add_U!(dSfdU[μ],-coeff,temp0_g')

        #= memo
        -Ydag otimes rminusγ1*Xplus
        -Xdag otimes rplusγ1*Yplus

        =#
    end
    return
    

    =#
    

    diracterm = W[1]
    coeff_i = get_coefficient(diracterm)
    #clear_U!(dSfdU)
    #return 
    
    #=

    for μ=1:Dim
        Xplus = shift_fermion(X,μ)
        #println(rplusγ1[:,:,μ])
        mul!(temp0_f,rminusγ1[:,:,μ],Xplus)
        #println(Xplus[1,1,1,1,1,1])
        #println(temp0_f[1,1,1,1,1,1])
        cross!(temp0_g,Y',temp0_f)

        #mul!(temp0_g,Y',temp0_f) 
       # println(temp0_g[:,:,1,1,1,1])

        add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
        println("c 1, ", temp0_g[1,1,1,1,1,1])

        Yplus = shift_fermion(Y,μ)
        mul!(temp0_f,rplusγ1[:,:,μ],Yplus)
        cross!(temp0_g,X',temp0_f)
        #mul!(temp0_g,X',temp0_f) 
        add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
        println("c 2, ",temp0_g[1,1,1,1,1,1])
        
    end
    =#

    #return

    #error("dd")

    
    

    

    for i=1:numterms
        diracterm = W[i]
        position  = collect(get_fermionposition(diracterm))
        Sdag = get_spinormatrix_dagger(diracterm)
        S = get_spinormatrix(diracterm)
        #γ5S = get_spinormatrix_γ5M(diracterm)
        coeff_i = get_coefficient(diracterm)
        #println("i = $i ",coeff_i)
        #if coeff_i < 1e-15
        #    continue
        #end
        for μ=1:Dim
            derivatives = get_derivatives(diracterm,μ)
            linkinfo_derivatives_leftdag = get_linkinfo_derivatives_leftdag(diracterm,μ)
            linkinfo_derivatives_right = get_linkinfo_derivatives_right(diracterm,μ)
            numderivatives = length(derivatives)

            for inum=1:numderivatives
                dFdμ_i = derivatives[inum]
                dFdμ_i_position_left = collect(dFdμ_i.position) 
                #dFdμ_i_position_left = -collect(dFdμ_i.position) 
                dFdμ_i_position_right = dFdμ_i_position_left .+ position 

                #println(Tuple(dFdμ_i_position_left))
                #println(Tuple(dFdμ_i_position_right))
                #if coeff_i > 1e-15
                Xshifted = shift_fermion(X,Tuple(dFdμ_i_position_right))
                Yshifted = shift_fermion(Y,Tuple(dFdμ_i_position_left))

                #if coeff_i > 1e-15
                gaugefield_fermion_mul!(temp1_f,linkinfo_derivatives_right[inum],U,Xshifted,fermi_action._temporary_fermionfields)
                #end
                mul!(temp1_f,S)

                #if coeff_i > 1e-15
                gaugefield_fermion_mul!(temp0_f,linkinfo_derivatives_leftdag[inum],U,Yshifted,fermi_action._temporary_fermionfields)
                #end
                cross!(temp0_g,temp0_f',temp1_f) 
                #println("1, ",coeff*coeff_i*temp0_g[:,:,1,1,1,1])
                #mul!(temp0_g,temp1_f,temp0_f') 

                #if coeff_i > 1e-15
                add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
                #end
                #println("1, μ = $μ ", dSfdU[μ][1,1,1,1,1,1])
                

                #if coeff_i > 1e-15
                
                Xshifted = shift_fermion(X,Tuple(dFdμ_i_position_left))
                Yshifted = shift_fermion(Y,Tuple(dFdμ_i_position_right))

                gaugefield_fermion_mul!(temp1_f,linkinfo_derivatives_right[inum],U,Yshifted,fermi_action._temporary_fermionfields)
                mul!(temp1_f,Sdag)

                gaugefield_fermion_mul!(temp0_f,linkinfo_derivatives_leftdag[inum],U,Xshifted,fermi_action._temporary_fermionfields)
                cross!(temp0_g,temp0_f',temp1_f) 
                
                #println("1, ",coeff*coeff_i*temp0_g[:,:,1,1,1,1])
                #mul!(temp0_g,temp1_f,temp0_f') 

                add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
                #println("2, μ = $μ ", dSfdU[μ][1,1,0,1,1,1])
                #end

            end
            
            #=

            derivatives = get_derivatives_dag(diracterm,μ)
            linkinfo_derivatives_leftdag = get_linkinfo_derivatives_dag_leftdag(diracterm,μ)
            linkinfo_derivatives_right = get_linkinfo_derivatives_dag_right(diracterm,μ)
            numderivatives = length(derivatives)

            for inum=1:numderivatives
                dFdμ_i = derivatives[inum]
                dFdμ_i_position_left = collect(dFdμ_i.position) 
                dFdμ_i_position_right = dFdμ_i_position_left .+ position 

                Xshifted = shift_fermion(X,Tuple(dFdμ_i_position_left))
                Yshifted = shift_fermion(Y,Tuple(dFdμ_i_position_right))

                gaugefield_fermion_mul!(temp1_f,linkinfo_derivatives_right[inum],U,Yshifted,fermi_action._temporary_fermionfields)
                mul!(temp1_f,S')

                gaugefield_fermion_mul!(temp0_f,linkinfo_derivatives_leftdag[inum],U,Xshifted,fermi_action._temporary_fermionfields)
                cross!(temp0_g,temp0_f',temp1_f) 
                #mul!(temp0_g,temp1_f,temp0_f') 

                add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
                println("2, ", temp0_g[1,1,1,1,1,1])

                #println("2, ",coeff*coeff_i*temp0_g[:,:,1,1,1,1])
                
            end
            =#
        end
        

        #show(diracterm)

        #D_positions = get_positions(diracterm)
        #D_directions = get_directions(diracterm)
        #D_isdagvectors = get_isdagvectors(diracterm)

    end
    

    return

    #=
        for μ=1:Dim
            derivatives = get_derivatives(diracterm,μ)
            linkinfo_derivatives_leftdag = get_linkinfo_derivatives_leftdag(diracterm,μ)
            linkinfo_derivatives_right = get_linkinfo_derivatives_right(diracterm,μ)
            numderivatives = length(derivatives)
            if numderivatives != 0
                println("i =$i μ = $μ, num = $numderivatives")
                show( derivatives)
            end
            

            for inum=1:numderivatives
                dFdμ_i = derivatives[inum]
                dFdμ_i_left = get_leftlinks(dFdμ_i)
                dFdμ_i_right = get_rightlinks(dFdμ_i)
                dFdμ_i_position_left = collect(dFdμ_i.position) 
                dFdμ_i_position_right = dFdμ_i_position_left .+ position 
                #show(dFdμ_i_left )
                #show(dFdμ_i_right )
                #println(dFdμ_i_position_right )
                #println(dFdμ_i_position_left )
                println(Tuple(dFdμ_i_position_left))
                println(Tuple(dFdμ_i_position_right))
                Xshifted = shift_fermion(X,Tuple(dFdμ_i_position_right))
                Yshifted = shift_fermion(Y,Tuple(dFdμ_i_position_left))

                gaugefield_fermion_mul!(temp1_f,linkinfo_derivatives_right[inum],U,Xshifted,fermi_action._temporary_fermionfields)
                mul!(temp1_f,S)

                gaugefield_fermion_mul!(temp0_f,linkinfo_derivatives_leftdag[inum],U,Yshifted,fermi_action._temporary_fermionfields)
                mul!(temp0_g,temp0_f',temp1_f) 
                #mul!(temp0_g,temp1_f,temp0_f') 

                add_U!(dSfdU[μ],coeff*coeff_i,temp0_g)
            end

            derivatives_dag = get_derivatives_dag(diracterm,μ)

            
            linkinfo_derivatives_dag_leftdag = get_linkinfo_derivatives_dag_leftdag(diracterm,μ)
            linkinfo_derivatives_dag_right = get_linkinfo_derivatives_dag_right(diracterm,μ)
            numderivatives = length(derivatives_dag)
            for inum=1:numderivatives
                println("position = ",derivatives_dag[inum].position)
            end


            if numderivatives != 0
                println("dag i =$i μ = $μ, num = $numderivatives")
                show( derivatives_dag)
            end

            for inum=1:numderivatives
                dFdμ_i = derivatives_dag[inum]
                dFdμ_i_left = get_leftlinks(dFdμ_i)
                dFdμ_i_right = get_rightlinks(dFdμ_i)
                dFdμ_i_position_left = collect(dFdμ_i.position) 
                dFdμ_i_position_right = dFdμ_i_position_left .+ position 
                println(Tuple(dFdμ_i_position_left))
                println(Tuple(dFdμ_i_position_right))
                Xshifted = shift_fermion(X,Tuple(-dFdμ_i_position_left))
                Yshifted = shift_fermion(Y,Tuple(-dFdμ_i_position_right))

                gaugefield_fermion_mul!(temp1_f,linkinfo_derivatives_dag_right[inum],U,Yshifted,fermi_action._temporary_fermionfields)

                mul!(temp1_f,S')

                gaugefield_fermion_mul!(temp0_f,linkinfo_derivatives_dag_leftdag[inum],U,Xshifted,fermi_action._temporary_fermionfields)
                mul!(temp0_g,temp0_f',temp1_f) 
                #mul!(temp0_g,temp1_f,temp0_f') 

                add_U!(dSfdU[μ],-coeff*coeff_i,temp0_g)
            end
        end
        
    end

    error("dte")

    =#
    return 


    

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

