
    function estimate_numberofeigenvalues(A,x,radius,origin,Z2randomize!,randomize!;L0=10,Nq=64,M=20,κ=2,δ=1e-14,MaxCGstep=3000,epsCG=1e-16,α=1)
        γ = origin
        ρ = radius

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

        evaluate_Eq22!(Sk,V,A,x,radius,origin,α,L0,Nq,M,MaxCGstep,epsCG)

        S0 = Vector{xtype}(undef,L0)
        for j=1:L0
            S0[j] = zero(x)
            axpy!(1,Sk[j],S0[j])
        end

        mst = evaluate_Eq26(S0,V,L0)

        println("done.")
        println("------------------------------------------------------------")
        println("Estimated number of eigenvalues: ",mst)

        return mst
    end

    function eigensystem(A,x,radius,origin,Z2randomize!,randomize!;L0=10,Nq=64,M=10,κ=2,δ=1e-14,MaxCGstep=3000,epsCG=1e-16,α=1)
        mst = estimate_numberofeigenvalues(A,x,radius,origin,Z2randomize!,randomize!,L0=L0,Nq=Nq,M=M,κ=κ,δ=δ,MaxCGstep=MaxCGstep,epsCG=epsCG,α=α)
        xtype = typeof(x)
        Nrh = Int64(round(κ*mst/M,RoundNearestTiesAway))
        if Nrh < 2
            Nrh = 2
        end
        Nint = Nq
        Nmm = 2*M
        N = length(x)

        z = zeros(ComplexF64,Nint)
        w = zero(z)

        for j = 1:Nint
            θ = (2π/Nint)*(j-1/2)
            z[j] = origin + radius*(cos(θ)+im*α*sin(θ))
            w[j] = α*cos(θ) + im*sin(θ)
        end

        V = Vector{xtype}(undef,Nrh)
        for i=1:Nrh
            V[i] = zero(x)
            #for j=1:N
            #    V[i][j] = rand()
            #end
            randomize!(V[i])
        end

        Sk = Vector{xtype}(undef,Nrh*Nmm*2)
        for i=1:Nrh*Nmm*2
            Sk[i] = zero(x)
        end

        #S = zeros(ComplexF64,N,Nrh*Nmm*2)

        for i=1:Nrh
            println("i = ",i,"/",Nrh)
            #(zj I - H) Yj = V
            #(-zj I + H) (-Yj) = V
            Y = shiftedbicg_2003(-z,A,V[i],maxsteps = MaxCGstep,eps=epsCG,verboselevel = 2)

            for k in 0:Nmm*2-1
                ik = k*Nrh+i
                for j in 1:Nint
                    coeff = radius*w[j]*z[j]^k/Nint
                    axpy!(-coeff,Y[j],Sk[ik])
                    #vec_Sk[:,ik] += ρ*vec_w[j]*vec_z[j]^k*vec_y[:,j]/Nq
                end
            end
        end

        μ = zeros(ComplexF64,Nrh,Nrh,2*Nmm)
        S = zeros(ComplexF64,N,Nrh*Nmm*2)
        for i=1:Nrh*Nmm*2
            S[:,i] = convert_to_normalvector(Sk[i])
        end

        Vmatrix = zeros(ComplexF64,N,Nrh)
        for i=1:Nrh
            Vmatrix[:,i] = convert_to_normalvector(V[i])
        end

        μ = zeros(ComplexF64,Nrh,Nrh,2*Nmm)
        for k=0:Nmm*2-1
            μ[:,:,k+1] = Vmatrix'*S[:,1+k*Nrh:Nrh+k*Nrh]
        end
        Tv = zeros(ComplexF64,Nmm*Nrh,Nmm*Nrh)
        Tt = zeros(ComplexF64,Nmm*Nrh,Nmm*Nrh)
        
        for i=1:Nmm
            for j=1:Nmm
                Tv[1+(i-1)*Nrh:Nrh+(i-1)*Nrh,1+(j-1)*Nrh:Nrh+(j-1)*Nrh] = μ[:,:,i+j-1+1] 
                Tt[1+(i-1)*Nrh:Nrh+(i-1)*Nrh,1+(j-1)*Nrh:Nrh+(j-1)*Nrh] = μ[:,:,i+j-2+1]
            end
        end

        F = svd(Tt)
        U = F.U
        Σ = F.S
        W = F.V

        eps = 1e-8

        δ = 1e-10
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
        Σinv = zeros(ms,ms)
        for i=1:ms
            Σinv[i,i] = 1/Σ[i]
        end

        H = U[:,1:ms]'*Tv*W[:,1:ms]*Σinv
        ε,v = eigen(H)
        println(ε)

        vec_xvec = zeros(ComplexF64,N,ms)
        for is=1:ms
            for i=1:Nmm
                a = Σinv*v[:,is] 
                b = W[:,1:ms]*a
                c = S[:,1:Nrh*Nmm]*b
                vec_xvec[:,is] += c
            end
        end

        vec_x = Array{typeof(x),1}(undef,ms)
        for i=1:ms
            vec_x[i] = zero(x)
            for j=1:N
                vec_x[i][j] = vec_xvec[j,i]
            end
        end

        Ax = similar(x)
        r = similar(x)

        is = 0
        eigenvalues = ComplexF64[]
        residuals = ComplexF64[]
        eigenvectors = []

        for i in 1:ms
            rr = real( ε[i]-origin)^2+imag( ε[i]-origin)^2
            if rr > radius^2 
                continue
            end


            mul!(Ax,A,vec_x[i])
            add!(0,r,-ε[i],vec_x[i])
            add!(r,1,Ax)

            residual = sqrt(real(dot(r,r)))/sqrt(real(dot(vec_x[i],vec_x[i])))
            #println(residual)
            println("$(ε[i]) residual: $residual")
            
            if residual  < eps
                println("$(ε[i]) residual: $residual")
                is += 1
                push!(eigenvalues,ε[i])
                push!(residuals,residual)
                push!(eigenvectors,vec_x[i])
            end        

        end

        #println("-----------------------------------------")   
        println("done.")
        println("------------------------------------------------------------")
        return eigenvalues,residuals,eigenvectors,is



    end

    function eigensystem_old2(A,x,radius,origin,Z2randomize!,randomize!;L0=10,Nq=64,M=20,κ=2,δ=1e-14,MaxCGstep=3000,epsCG=1e-16,α=1)

        γ = origin
        ρ = radius

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

        evaluate_Eq22!(Sk,V,A,x,radius,origin,α,L0,Nq,M,MaxCGstep,epsCG)

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
        if L < 2
            L = 2
        end
        println("L is ",L)
    
        println("Calculating eigenvalues and eigenvectors...")
        println("------------------------------------------------------------")   

        V = Vector{xtype}(undef,L)
        for i=1:L
            V[i] = zero(x)
            randomize!(V[i])
        end


        ε,vec_x,ms = evaluate_SS(Sk,V,A,x,radius,origin,α,L,Nq,M,MaxCGstep,epsCG,δ)
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

            
            if resiup < 0.001  && γ-ρ <= real(ε[i]) <= ρ+γ
                is += 1
                push!(eigenvalues,ε[i])
                push!(residuals,resiup)
                push!(eigenvectors,vec_x[i])
            end
            

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

    function evaluate_Eq22!(Sk,V,A,x,radius,origin,α,L,Nq,M,MaxCGstep,epsCG)
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

    function evaluate_SS(Sk,V,A,x,radius,origin,α,L,Nq,M,MaxCGstep,epsCG,δ)
        xvec = convert_to_normalvector(x)
        N = length(xvec)
        
        evaluate_Eq22!(Sk,V,A,x,radius,origin,α,L,Nq,M,MaxCGstep,epsCG)
        Skmatrix = zeros(ComplexF64,N,L*M)
        for i=1:L*M
            Skmatrix[:,i] = convert_to_normalvector(Sk[i])
        end
        Vmatrix = zeros(ComplexF64,N,L)
        for i=1:L
            Vmatrix[:,i] = convert_to_normalvector(V[i])
        end
        μk = zeros(ComplexF64,L,L,M)
        for k=0:M-1
            μk[:,:,k+1] = Vmatrix'*Skmatrix[:,1+k*L:L+k*L]
            #μk[:,:,i] = Vmatrix'*Skmatrix[:,1+(i-1)*L:L+(i-1)*L]
        end
        M2 = div(M,2)
        T1matrix = zeros(ComplexF64,L*M2,L*M2)
        T2matrix = zeros(ComplexF64,L*M2,L*M2)
        for i=1:M2
            for j=1:M2
                T1matrix[1+(i-1)*L:L+(i-1)*L,1+(j-1)*L:L+(j-1)*L] = μk[:,:,i+j-1+1]
                T2matrix[1+(i-1)*L:L+(i-1)*L,1+(j-1)*L:L+(j-1)*L] = μk[:,:,i+j-2+1]
            end
        end
        F =  svd(T2matrix)
        U = F.U
        Σ = F.S
        W = F.V
        #(U,Σ,W ) = svd(T2matrix)
        println( Σ)
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
        println("ms = $ms")
        Σmatrixinv = zeros(ComplexF64,ms,ms)
        for i=1:ms
            Σmatrixinv[i,i] = 1/Σ[i]
        end
        H = U[:,1:ms]'*T1matrix*W[:,1:ms]*Σmatrixinv

        ε,vec_w = eigen(H)
        vec_xvec = zeros(ComplexF64,N,ms)
        for is=1:ms
            for i=1:M2
                a = Σmatrixinv*vec_w[:,is] 
                b = W[:,1:ms]*a
                c = Skmatrix[:,1:L*M2]*b
                vec_xvec[:,is] += c
                #vec_xvec[:,is] += Skmatrix[:,1+(i-1)*L:L+(i-1)*L]*W[:,1:ms]*Σmatrixinv*vec_w[:,is] 
            end
        end

        vec_x = Array{typeof(x),1}(undef,ms)
        for i=1:ms
            vec_x[i] = zero(x)
            for j=1:N
                vec_x[i][j] = vec_xvec[j,i]
            end
        end

        

        return ε,vec_x,ms



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
        if L < 2
            L = 2
        end
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

            
            if resiup < 0.001  && γ-ρ <= real(ε[i]) <= ρ+γ
                is += 1
                push!(eigenvalues,ε[i])
                push!(residuals,resiup)
                push!(eigenvectors,vec_x[i])
            end
            

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

    
