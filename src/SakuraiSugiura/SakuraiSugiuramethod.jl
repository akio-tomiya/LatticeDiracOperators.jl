
    function eigensystem(A,x,radius,origin,Z2randomize!,randomize!;L0=10,Nq=64,M=10,κ=2,δ=1e-14,MaxCGstep=3000,epsCG=1e-16,α=1)
        println("Estimating number of eigenvalues...")
        println("------------------------------------------------------------")
        xtype = typeof(x)
        V = Vector{xtype}(undef,L0)
        for i=1:L0
            V[i] = zero(x)
            Z2randomize!(V[i])
        end

        Sk = Vector{xtype}(undef,L0*M)
        for i=1:L0*M
            Sk[i] = zero(x)
        end

        Y = Vector{xtype}(undef,Nq)
        for j=1:Nq
            Y[j] = zero(x)
        end

        evaluate_Eq22!(Sk,Y,V,A,x,radius,origin,α,L0,Nq,M,MaxCGstep,epsCG)

        S0 = Vector{xtype}(undef,L0)
        for j=1:L0
            S0[j] = zero(x)
            axpy!(1,Sk[j],S0[j])
        end

        mst = evaluate_Eq26(S0,V,L0)

        println("done.")
        println("------------------------------------------------------------")
        println("Estimated number of eigenvalues: ",mst)
        L = Int64(round(κ*mst/M,RoundNearestTiesAway))
        println("L is ",L)
    
        println("Calculating eigenvalues and eigenvectors...")
        println("------------------------------------------------------------")   

        V = Vector{xtype}(undef,L)
        for i=1:L
            V[i] = zero(x)
            randomize!(V[i])
        end

        

    end

    function evaluate_Eq22!(Sk,Y,V,A,x,radius,origin,α,L,Nq,M,MaxCGstep,epsCG)
        vec_z = zeros(ComplexF64,Nq)
        vec_w = zeros(ComplexF64,Nq)
        for j in 1:Nq
            θj = (2π/Nq)*(j-1/2)
            vec_z[j] = origin + radius*(cos(θj)+im*α*sin(θj))
            vec_w[j] = α*cos(θj)+im*sin(θj)
        end

        for i in 1:L
            println("i = ",i,"/",L)
            #(zj I - H) Yj = V
            #(-zj I + H) (-Yj) = V
            Y = shiftedbicg_2003(-vec_z,A,V[i],maxsteps = MaxCGstep,eps=epsCG,verboselevel = 2)

            for k in 0:M-1
                ik = k*L+i
                for j in 1:Nq
                    coeff = radius*vec_w[j]*vec_z[j]^k/Nq
                    axpy!(-coeff,Y[j],Sk[ik])
                    #vec_Sk[:,ik] += ρ*vec_w[j]*vec_z[j]^k*vec_y[:,j]/Nq
                end
            end

        end
    end

    function evaluate_Eq26(S0,V,L0)
        m = 0.0im
        for i in 1:L0
            m += dot(V[i],S0[i])
            #m += hatV[:,i]'*vec_s0[:,i]
        end
        ms = Int64(round(real(m)/L0))
        if ms < 0
            ms = 1
        end
        return ms
    end


function eigensystem_old(A,x,N,ρ,γ;α=0.1,L0=10,Nq=64,M=10,κ=2,δ=1e-14,numr = 0,cg=true,MaxCGstep=3000,epsCG=1e-16)
        println("Estimating number of eigenvalues...")
        println("------------------------------------------------------------")

        V = Array{typeof(x),1}(undef,L0)
        for i=1:L0
            temp = rand([-1,1],N)
            V[i] = similar(x)
            for j=1:N
                V[i][j] = temp[j]
            end
            #=
            if typeof(x) <: Abstractfermion
                V[i] = similar(x)
                Z2_distribution_fermion!(V[i])
            else
                V[i] = rand([-1,1],N)
            end
            =#
        end

        Sk = Array{typeof(x),1}(undef,L0*M)
        for i=1:L0*M
            Sk[i] = similar(x)
        end

        Y = Array{typeof(x),1}(undef,Nq)
        for j=1:Nq
            Y[j] = similar(x)
        end

        #V = rand([-1,1],N,L0)
        calculate_Eq22!(Sk,Y,V,A,x,N,ρ,γ,α,L0,Nq,M,cg,MaxCGstep,epsCG)

        S0 = Array{typeof(x),1}(undef,L0)
        for i=1:L0
            S0[i] = similar(x)
            if typeof(x) <: Abstractfermion
                add!(0,S0[i],1,Sk[i]) #S0[i] = S[i]
            else
                S0[i] .= Sk[i]
            end
        end

        mst = calc_msinEq26(S0,V,L0)

        println("done.")
        println("------------------------------------------------------------")
        println("Estimated number of eigenvalues: ",mst)
        L = Int64(round(κ*mst/M,RoundNearestTiesAway))
        println("L is ",L)

        println("Calculating eigenvalues and eigenvectors...")
        println("------------------------------------------------------------")    


        V = Array{typeof(x),1}(undef,L)
        for i=1:L
            temp = rand(N).*2 .- 1
            V[i] = similar(x)
            for j=1:N
                V[i][j] = temp[j]
            end

            #=
            if typeof(x) <: Abstractfermion
                V[i] = similar(x)
                uniform_distribution_fermion!(V[i])
            else
                V[i] = rand(N).*2 .- 1
            end
            =#

        end

        Sk = Array{typeof(x),1}(undef,L*M)
        for i=1:L*M
            Sk[i] = similar(x)
        end


        ε,vec_x,ms = calc_SS(Sk,Y,V,A,x,N,ρ,γ,α,L,Nq,M,δ,cg,MaxCGstep,epsCG)
        println(ε)
        Ax = similar(x)
        r = similar(x)

        is = 0
        eigenvalues = ComplexF64[]
        residuals = ComplexF64[]
        eigenvectors = []

        for i in 1:ms
            mul!(Ax,A,vec_x[i])
            add!(0,r,-ε[i],vec_x[i])
            add!(r,1,Ax)

            resiup = sqrt(real(dot(r,r)))
            println(resiup)

            
            #if resiup < 0.1 # && γ-ρ <= ε[i] <= ρ+γ
                is += 1
                push!(eigenvalues,ε[i])
            #    push!(residuals,resi)
                push!(eigenvectors,vec_x[i])
            #end
            

            #=
            residown1 = sqrt(dot(Ax,Ax))
            residown2 = sqrt(dot(vec_x[i],vec_x[i]))
            resi = real(resiup/(residown1+abs(ε[i])*residown2))
            if resi < 0.1 # && γ-ρ <= ε[i] <= ρ+γ
                is += 1
                push!(eigenvalues,ε[i])
                push!(residuals,resi)
                push!(eigenvectors,vec_x[i])
            end
            =#


        end

        #println("-----------------------------------------")   
        println("done.")
        println("------------------------------------------------------------")
        return eigenvalues,residuals,eigenvectors,is
        
    end

    function calc_msinEq26(S0,V,L0)
        m = 0.0im
        for i in 1:L0
            m += dot(V[i],S0[i])
        end
        #m = real(m)/L0
        #println(m)
        ms = Int64(round(real(m)/L0))
        if ms < 0
            ms = 1
        end

        return ms
    end

    function calc_SS(Sk,Y,mV,A,x,N,ρ,γ,α,L,Nq,M,δ,cg,MaxCGstep,epsCG)
        calculate_Eq22!(Sk,Y,mV,A,x,N,ρ,γ,α,L,Nq,M,cg,MaxCGstep,epsCG)

        Skmatrix = zeros(ComplexF64,N,L*M)
        if typeof(x) <: Abstractfermion
            fermions2vectors!(Skmatrix,Sk)
        else
            for i=1:L*M
                for j=1:N
                    Skmatrix[j,i] = Sk[i][j]
                end
            end
        end

        
        (U,Σ,W ) = svd(Skmatrix)
        j = 0
        ratio = 1
        while ratio > δ
            j += 1
            ratio = Σ[j]/Σ[1]
            if j == length(Σ)
                ratio = 0.0
            end
        end

        ms = j
        Q = Array{typeof(x),1}(undef,ms)
        for i=1:ms
            Q[i] = similar(x)
            for j=1:N
                Q[i][j]=U[j,i]
            end
            if typeof(x) <: Abstractfermion
                set_wing_fermion!(Q[i])
            end
        end

        temp = Array{typeof(x),1}(undef,ms)
        for i=1:ms
            temp[i] = similar(x)
            mul!(temp[i],A,Q[i])
        end

        mat_Ht =zeros(ComplexF64,ms,ms)
        for i=1:ms
            for j=1:ms
                mat_Ht[j,i] = dot(Q[j],temp[i])
            end
        end

        #mat_Ht = (mat_Ht'+mat_Ht)/2

        ε,vec_w = eigen(mat_Ht)
        vec_x = Array{typeof(x),1}(undef,ms)
        for i=1:ms
            vec_x[i] = similar(x)
            if typeof(x) <: Abstractfermion
            else
                vec_x[i] .= 0
            end

            for j=1:N
                for k=1:ms
                    vec_x[i][j] += Q[k][j]*vec_w[k,i]
                end
            end

            if typeof(x) <: Abstractfermion
                set_wing_fermion!(vec_x[i])
            end
        end
        #vec_x = mat_Q*vec_w  

        return ε,vec_x,ms
    end

    function calculate_Eq22!(Sk,Y,mV,A,x,N,ρ,γ,α,L,Nq,M,cg,MaxCGstep,epsCG) #mV = -V
        if typeof(x) <: Abstractfermion
            for i in 1:L
                for k = 0:M-1
                    ik = k*L+i
                    clear_fermion!(Sk[ik])
                end
            end
        else
            for i in 1:L
                for k = 0:M-1
                    ik = k*L+i
                    Sk[ik] .= 0
                end
            end
        end


        z = zeros(ComplexF64,Nq)
        w = zero(z)

        for j = 1:Nq
            θ = (2π/Nq)*(j-1/2)
            z[j] = γ + ρ*(cos(θ)+im*α*sin(θ))
            w[j] = α*cos(θ) + im*sin(θ)
        end

        #(zI - A) Y = V
        #(A - z I) (-Y) = V
        for i in 1:L
            #shiftedcg(Y,-z,x,A,mV[i])
            Y = shiftedbicg_2003(-z,A,mV[i],maxsteps = MaxCGstep,eps=epsCG,verboselevel = 1)
            #shiftedbicg(Y,-z,x,A,mV[i],maxsteps = MaxCGstep,eps=epsCG)
            for k = 0:M-1
                ik = k*L+i
                for j = 1:Nq
                    coef = ρ*w[j]*z[j]^k/Nq
                    add!(Sk[ik],-coef,Y[j])
                end
            end
        end

    end

    
