module SSmodule
    using LinearAlgebra
    function eigs_SS(A,x)
    end

    function shiftedbicg_Frommer2003(σ,A,b;maxsteps=3000,eps = 1e-15)
        N = length(σ)
        btype = typeof(b)
        q = zero(b)
        x = zero(b)
        r = deepcopy(b)
        rt = deepcopy(b)
        u = zero(b)
        ut = zero(b)
        vec_u = Vector{btype}(undef,N)
        vec_x = Vector{btype}(undef,N)
        for j=1:N
            vec_u[j] = zero(b)
            vec_x[j] = zero(b)
        end
        ρkold = ones(ComplexF64,N)
        ρk = ones(ComplexF64,N)
        πkold = ones(ComplexF64,N)
        πk = ones(ComplexF64,N)
        πknew = ones(ComplexF64,N)
        ρold = 1
        αold = 1
        residual = 0

        for i=1:maxsteps
            ρ = dot(rt,r)
            β = -ρ/ρold    
            axpby!(1,r,-β,u)
            axpby!(1,rt,-β',ut)
            mul!(q,A,u)
            qrt = dot(rt,q)
            α = ρ/qrt
            axpy!(α,u,x)


            for k=1:N
                πknew[k] = (1+α*σ[k])*πk[k]+ (α*β/αold)*(πkold[k]- πk[k])
                βk = (πkold[k]/πk[k])^2*β
                αk =  (πk[k]/πknew[k])*α
                axpby!(1/πk[k],r,-βk,vec_u[k])
                axpy!(αk,vec_u[k],vec_x[k])
            end


            axpy!(-α,q,r)
            mul!(q,A',ut)
            axpy!(-α',q,rt)
            αold = α
            ρold  = ρ

            residual = dot(r,r)
            println(dot(r,r))

            if abs(residual) < eps
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
                return vec_x
            end

            for k=1:N
                πkold[k] = πk[k]
                πk[k] = πknew[k]
            end

        end

        error("bicg is not converged. The residual is $residual")


    end

    function shiftedbicg_Frommer2003_seed(σ,A,b;maxsteps=3000,eps = 1e-15,zseedin = 0im)
        zseed = zseedin
        N = length(σ)
        btype = typeof(b)
        q = zero(b)
        x = zero(b)
        r = deepcopy(b)
        rt = deepcopy(b)
        u = zero(b)
        ut = zero(b)
        vec_u = Vector{btype}(undef,N)
        vec_x = Vector{btype}(undef,N)
        for j=1:N
            vec_u[j] = zero(b)
            vec_x[j] = zero(b)
        end
        ρkold = ones(ComplexF64,N)
        ρk = ones(ComplexF64,N)
        πkold = ones(ComplexF64,N)
        πk = ones(ComplexF64,N)
        πknew = ones(ComplexF64,N)
        ρold = 1
        αold = 1
        residual = 0

        for i=1:maxsteps
            ρ = dot(rt,r)
            β = -ρ/ρold    
            axpby!(1,r,-β,u)
            axpby!(1,rt,-β',ut)
            mul!(q,A,u)
            axpby!(zseed,u,1,q)

            qrt = dot(rt,q)
            α = ρ/qrt
            axpy!(α,u,x)


            for k=1:N
                πknew[k] = (1+α*(σ[k]-zseed))*πk[k]+ (α*β/αold)*(πkold[k]- πk[k])
                βk = (πkold[k]/πk[k])^2*β
                αk =  (πk[k]/πknew[k])*α
                axpby!(1/πk[k],r,-βk,vec_u[k])
                axpy!(αk,vec_u[k],vec_x[k])
            end


            axpy!(-α,q,r)
            mul!(q,A',ut)
            axpby!(zseed',ut,1,q)
            axpy!(-α',q,rt)
            
            residual = dot(r,r)
            println(dot(r,r),"\t",zseed)

            for k=1:N
                πkold[k] = πk[k]
                πk[k] = πknew[k]
            end

            if false

            πkseed = 1
            πkseedold = 1 
            kseed = 0 
            for k=1:N
                if abs(πk[k]) < abs(πkseed)
                    πkseed =πk[k]
                    πkseedold = πkold[k] 
                    zseed = σ[k]
                    kseed = k
                end
            end



            
            println(kseed)
            if kseed != 0
                axpby!(0,rt,1/πkseed,r)
                axpby!(0,r,1/πkseed,rt)
                axpby!(0,r,1/πkseed,u)
                axpby!(0,r,1/πkseed,ut)
                axpby!(1,vec_x[kseed],0,x)
                ρ = ρ/(πkseedold*πkseedold)
                α = (πkseedold/πkseed)*α

                for k=1:N
                    πkold[k] = πkold[k]/πkseedold
                    πk[k] = πk[k]/πkseed
                end
            end
            

            end


            αold = α
            ρold  = ρ




            if abs(residual) < eps
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
                return vec_x
            end

            

        end

        error("bicg is not converged. The residual is $residual")


    end

    function shiftedbicg_Frommer2003_G_seed(σ,A,b;maxsteps=3000,eps = 1e-15,zseedin = 0im)
        zseed = zseedin
        N = length(σ)
        btype = typeof(b)
        q = zero(b)
        x = zero(b)
        r = deepcopy(b)
        rt = deepcopy(b)
        u = zero(b)
        ut = zero(b)
        vec_u = Vector{btype}(undef,N)
        vec_x = Vector{btype}(undef,N)
        for j=1:N
            vec_u[j] = zero(b)
            vec_x[j] = zero(b)
        end
        ρkold = ones(ComplexF64,N)
        ρk = ones(ComplexF64,N)
        πkold = ones(ComplexF64,N)
        πk = ones(ComplexF64,N)
        πknew = ones(ComplexF64,N)
        ρold = 1
        αold = 1
        residual = 0

        for i=1:maxsteps
            ρ = dot(rt,r)
            β = -ρ/ρold    
            axpby!(1,r,-β,u)
            axpby!(1,rt,-β',ut)
            mul!(q,A,u)
            axpby!(zseed,u,-1,q)

            qrt = dot(rt,q)
            α = ρ/qrt
            axpy!(α,u,x)


            for k=1:N
                πknew[k] = (1+α*(σ[k]-zseed))*πk[k]+ (α*β/αold)*(πkold[k]- πk[k])
                βk = (πkold[k]/πk[k])^2*β
                αk =  (πk[k]/πknew[k])*α
                axpby!(1/πk[k],r,-βk,vec_u[k])
                axpy!(αk,vec_u[k],vec_x[k])
            end


            axpy!(-α,q,r)
            mul!(q,A',ut)
            axpby!(zseed',ut,-1,q)
            axpy!(-α',q,rt)

            #ρMAX = maximum(abs.(πknew))^2            
            residual = abs(dot(r,r))#*ρMAX)

            

            #println(residual,"\t",zseed,"\t")#,dot(r,r),"\t",ρMAX)

            for k=1:N
                πkold[k] = πk[k]
                πk[k] = πknew[k]
            end

            if false

            πkseed = 1
            πkseedold = 1 
            kseed = 0 
            for k=1:N
                if abs(πk[k]) < abs(πkseed)
                    πkseed =πk[k]
                    πkseedold = πkold[k] 
                    zseed = σ[k]
                    kseed = k
                end
            end



            
            println(kseed)
            if kseed != 0
                axpby!(0,rt,1/πkseed,r)
                axpby!(0,r,1/πkseed',rt)
                #axpby!(0,r,1/πkseedold,u)
                #axpby!(0,r,1/πkseedold',ut)
                axpby!(1,vec_x[kseed],0,x)
                ρ = ρ/(πkseedold'*πkseedold)
                α = (πkseedold/πkseed)*α

                for k=1:N
                    πkold[k] = πkold[k]/πkseedold
                    πk[k] = πk[k]/πkseed
                end
            end
            

            end


            αold = α
            ρold  = ρ




            if abs(residual) < eps
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
                return vec_x
            end

            

        end

        error("bicg is not converged. The residual is $residual")


    end


    function shiftedbicgstab_inSS(σjin,A,b;maxsteps=3000,eps = 1e-15,zseed= 0im) #(sigma_i I + A)*x = b
        σj = deepcopy(σjin) .- zseed
        N = length(σj)
        btype = typeof(b)
        r = deepcopy(b)
        temp1 = similar(b)#temps[2]
        mul!(temp1,A,x)
        axpy!(-1,temp1,r) #add!(r,-1,temp1)
    
        rs = deepcopy(r)#temps[3]
        #substitute_fermion!(rs,r)
        p = deepcopy(r)#temps[4]
        #substitute_fermion!(p,r)
        Ap = similar(b)#temps[5]
        s = similar(b)#temps[6]
        t = similar(b)#temps[7]


        #r = deepcopy(b)
        #p = deepcopy(b)
        #q = similar(b)
        #s = similar(b)
        x = zero(b)

        vec_r = Vector{btype}(undef,N)
        vec_p = Vector{btype}(undef,N)
        vec_x = Vector{btype}(undef,N)
        for j=1:N
            vec_r[j] = deepcopy(b)
            vec_p[j] = deepcopy(b)
            vec_x[j] = zero(b)
        end


        αm = 1.0
        βm = 0.0
    
        ρm = ones(ComplexF64,N)
        ρ0 = ones(ComplexF64,N)
        ρp = ones(ComplexF64,N)
        τj = deepcopy(ρ0)
        #τjm = deepcopy(ρ0)
        residual = 0

        mul!(p,A',r) #p = (A + zseed*I)'*r
        axpy!(zseed',r,p) 
        c1 = dot(p,p)

        for i=1:maxsteps
            c1 = dot(rs,r)
            mul!(Ap,A,p)
            c2 = dot(rs,Ap)
            α = c1/c2
            #s = r - α*A*p
            #axpby!(1,r,β,p) #add!(β,p,1,r)
            # s = r - α Ap
            axpby!(1,r,0,s) #add!(0,s,1,r)
            axpy!(-α,Ap,s) #add!(s,-α,Ap)
            mul!(t,A,s)
            d1 = dot(t,s)
            d2 = dot(t,t)
            ζk = d1/d2
    
            #r = (1-ω A)s
            axpby!(1,s,0,r)#add!(0,r,1,s)
            axpy!(-ζk,t,r)  #add!(r,-ω,t)
    
            #x = x + ωs+ αp
            axpy!(ζk,s,x) #add!(x,ω,s)
            axpy!(α,p,x) #add!(x,α,p)
    
            β = (dot(rs,r)/c1)*(α/ζk)
    
            #p = r + β*(1-ωA)*p
            axpby!(1,r,β,p) #add!(β,p,1,r)
            axpy!(-ζk*β,Ap,p) #add!(p,-ω*β,Ap)
    
            αk = alpha
            βk = beta
            println(dot(r,r))

            for j=1:N
                ρkj = ρ0[j]
                if abs(ρkj) < eps
                    continue
                end
                ρkmj =ρm[j]
                ρp[j] = ρkj*ρkmj*αm/(ρkmj*αm*(1.0+αk*σj[j])+αk*βm*(ρkmj-ρkj))
                αkj = (ρp[j]/ρkj)*αk
                ζkj = ζk/(1+ζk*σj[j])

                #! ...  vec_x[j]   = vec_x[j]   + αkj * vec_p[j] +ζkj*τj*ρp[j]*s
                axpy!(αkj,vec_p[j],vec_x[j])
                axpy!(ζkj*τj[j]*ρp[j],s,vec_x[j])

                τjm = τj[j]  
                τj[j] = (1/(1+ζk*σj[j]))*τj[j] 
                βkj = (ρp[j]/ρkj)^2*βk
                # vec_p[j] = τj[j]*ρp[j]*r + β*(vec_p[j] + (ζkj/αkj)*τjm*(ρp[j]*s-ρkj*r) )
                # vec_p[j] = β*vec_p[j] + β*(τj[j]*ρp[j]-(ζkj/αkj)*τjm*ρkj)*r + β*(ζkj/αkj)*τjm*ρp[j]*s
                axpby!(βkj*(ζkj/αkj)*τjm*ρp[j],s,βkj,vec_p[j])
                axpy!(βkj*(τj[j]*ρp[j]-(ζkj/αkj)*τjm*ρkj),r,vec_p[j])
                
                #add!(vec_x[j],αkj,vec_p[j])
                #βkj = (ρp[j]/ρkj)^2*βk
                #add!(βkj,vec_p[j],ρp[j],r) 
                axpby!(ρp[j],s,βkj,vec_p[j]) #p = beta*p + s
                #add!(βkj,vec_p[j],ρp[j],s) 
    
            end

            ρm .= ρ0
            ρ0 .= ρp
            αm = αk
            βm = βk


            ρMAX = maximum(abs.(ρp))^2
            residual = abs(rr*ρMAX)
    
            if abs(residual) < eps
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
                return x,vec_x
            end

        end
        error("bicg is not converged. The residual is $residual")

    end

    function shiftedbicgstab(x,A,b;eps=1e-10,maxsteps = 1000) #Ax=b
        #println_verbose_level3(verbose,"--------------------------------------")
        #println_verbose_level3(verbose,"bicg-stab method")
    
        #=
        r = deepcopy(b)
        temp1 = similar(x)
        mul!(temp1,A,x)
        add!(r,-1,temp1)
    
        rs = deepcopy(r)
        p = deepcopy(r)
        Ap = similar(r)
        s = similar(r)
        t = similar(r)
        =#
        
    
        
        #temps = get_temporaryvectors_forCG(A)
        #r = temps[1]
        #substitute_fermion!(r,b)
        r = deepcopy(b)
        temp1 = similar(b)#temps[2]
        mul!(temp1,A,x)
        axpy!(-1,temp1,r) #add!(r,-1,temp1)
    
        rs = deepcopy(r)#temps[3]
        #substitute_fermion!(rs,r)
        p = deepcopy(r)#temps[4]
        #substitute_fermion!(p,r)
        Ap = similar(b)#temps[5]
        s = similar(b)#temps[6]
        t = similar(b)#temps[7]
        
        
    
    
        rnorm = real(dot(r,r))
    
        if rnorm < eps
            return
        end
        
        
        for i=1:maxsteps
            c1 = dot(rs,r)
            mul!(Ap,A,p)
            c2 = dot(rs,Ap)
            α = c1/c2
            #s = r - α*A*p
            #axpby!(1,r,β,p) #add!(β,p,1,r)
            # s = r - α Ap
            axpby!(1,r,0,s) #add!(0,s,1,r)
            axpy!(-α,Ap,s) #add!(s,-α,Ap)
            mul!(t,A,s)
            d1 = dot(t,s)
            d2 = dot(t,t)
            ω = d1/d2
    
            #r = (1-ω A)s
            axpby!(1,s,0,r)#add!(0,r,1,s)
            axpy!(-ω,t,r)  #add!(r,-ω,t)
    
            #x = x + ωs+ αp
            axpy!(ω,s,x) #add!(x,ω,s)
            axpy!(α,p,x) #add!(x,α,p)
    
            β = (dot(rs,r)/c1)*(α/ω)
    
            #p = r + β*(1-ωA)*p
            axpby!(1,r,β,p) #add!(β,p,1,r)
            axpy!(-ω*β,Ap,p) #add!(p,-ω*β,Ap)



            
            rnorm = real(r ⋅ r) 
            println("$i-th eps: $rnorm")

            
    
            if rnorm < eps
                println("Converged at $i-th step. eps: $rnorm")
                #println_verbose_level3(verbose,"--------------------------------------")
                return
            end
    
    
    
        end
    
    
    
        error("""
        The BICGstab is not converged! with maxsteps = $(maxsteps)
        residual is $rnorm
        maxsteps should be larger.""")
    
    
    end

    function shiftedbicgstab_inSS(x,A,b;eps=1e-10,maxsteps = 1000) #Ax=b


        r = deepcopy(b)
        temp1 = similar(x)
        mul!(temp1,A,x)
        axpy!(-1,temp1,r) 
        #add!(r,-1,temp1)

        rnorm = real(dot(r,r))
        println(rnorm)
    
        if rnorm < eps
            return
        end
        rs = deepcopy(r)
        p = deepcopy(r)
        s = similar(r)

        s = similar(r)
        Ap = similar(x)
        As = similar(x)
        β = 0.0im    
        ω = 0.0im
    
        for i=0:maxsteps
            if i != 0
                #add!(β,p,1,r)
                #add!(p,-ω*β,Ap)
                axpby!(1,r,β,p) #add!(β,p,1,r)
                axpy!(-ω*β,Ap,p) #add!(p,-ω*β,Ap)
            end

            c1 = dot(rs,r)
            mul!(Ap,A,p)
            c2 = dot(rs,Ap)
            α = c1/c2
            #add!(0,s,1,r)
            axpby!(1,r,0,s)
            #add!(s,-α,Ap)
            axpy!(-α,Ap,s) 
            mul!(As,A,s)
            s1 = dot(As,s)
            s2 = dot(As,As)
    
            ω = s1/s2
            axpy!(α,p,x) #add!(x,α,p)
            
            axpy!(ω,s,x) #add!(x,ω,s)
            
            axpby!(1,s,0,r) #add!(0,r,1,s)
            
            axpy!(-ω,As,r) #add!(r,-ω,As)
            
    
            rnorm = real(dot(r,r)) 
            println("$i-th eps: $rnorm")
    
            if rnorm < eps
                println("Converged at $i-th step. eps: $rnorm")
                #println_verbose_level3(verbose,"--------------------------------------")
                return 
            end
    
            β = (α/ω)*dot(rs,r)/c1

            #=
            axpby!(1,r,β,p) #add!(β,p,1,r)
            axpy!(-ω*β,Ap,p) #add!(p,-ω*β,Ap)
    =#
        end
    
    
     
        
        error("""
        The BICG is not converged! with maxsteps = $(maxsteps)
        residual is $rnorm
        maxsteps should be larger.""")
    
    
    end


    function shiftedbicg_inSS(σjin,A,b;maxsteps=3000,eps = 1e-15,zseedin = 0) #(sigma_i I + A)*x = b
        zseed = zseedin
        σj = deepcopy(σjin) #.- zseed
        N = length(σj)
        btype = typeof(b)
        r = deepcopy(b)
        p = deepcopy(b)
        q = similar(b)
        s = similar(b)
        x = zero(b)

        vec_r = Vector{btype}(undef,N)
        vec_p = Vector{btype}(undef,N)
        vec_x = Vector{btype}(undef,N)
        for j=1:N
            vec_r[j] = deepcopy(b)
            vec_p[j] = deepcopy(b)
            vec_x[j] = zero(b)
        end


        αm = 1.0
        βm = 0.0
    
        ρm = ones(ComplexF64,N)
        ρ0 = ones(ComplexF64,N)
        ρp = ones(ComplexF64,N)
        residual = 0

        #=
        mul!(p,A',r) #p = (A + zseed*I)'*r
        axpy!(zseed',r,p) 
        =#
        c1 = dot(p,p)

        for i=1:maxsteps
            mul!(q,A,p) #q = (A + zseed*I)*p
            axpy!(zseed,p,q) 

            c2 = dot(q,q)
            alpha = c1 / c2
            rr = dot(r,r)
            #! ...  x   = x   + alpha * p  
            axpy!(alpha,p,x)
            #axpy!(a, X, Y) #Overwrite Y with X*a + Y, where a is a scalar. Return Y.
            #axpby!(a, X, b, Y) #Overwrite Y with X*a + Y*b

            #...  res = res - alpha * q 
            axpy!(-alpha,q,r)

            mul!(s,A',r)#s = (A + zseed*I)'*r
            axpy!(zseed',r,s) 

            c3 = dot(s,s)

            beta = c3 / c1
            c1 = c3
            #axpby!(a, X, b, Y) #Y = X*a + Y*b
            axpby!(1,s,beta,p) #p = beta*p + s

    
            αk = alpha
            βk = beta
            println(dot(r,r))
            ρmin = 1
            jmin = 0
            ρkjmin = 1

            for j=1:N
                ρkj = ρ0[j]
                if abs(ρkj) < eps
                    continue
                end
                ρkmj =ρm[j]
                ρp[j] = ρkj*ρkmj*αm/(ρkmj*αm*(1.0+αk*(σj[j]-zseed))+αk*βm*(ρkmj-ρkj))
                if abs(ρmin) > abs(ρp[j])
                    ρmin = ρp[j]
                    jmin = j
                    ρkjmin = ρkj 
                end
                #println(ρp[j])
                αkj = (ρp[j]/ρkj)*αk
                #println("$αk alpha = ",αkj)

                #! ...  vec_x[j]   = vec_x[j]   + αkj * vec_p[j]
                axpy!(αkj,vec_p[j],vec_x[j])
                #println("dot ",dot(x,vec_x[j]))
                #add!(vec_x[j],αkj,vec_p[j])
                βkj = (ρp[j]/ρkj)^2*βk
                #add!(βkj,vec_p[j],ρp[j],r) 
                #println("ρp[j] ",ρp[j])
                axpby!(ρp[j],s,βkj,vec_p[j]) #p = beta*p + s
                #add!(βkj,vec_p[j],ρp[j],s) 
                #println("dot p ",dot(p,vec_p[j]))
    
            end

            ρm .= ρ0
            ρ0 .= ρp
            αm = αk
            βm = βk


            ρMAX = maximum(abs.(ρp))^2

            if jmin != 0
                zseed = σj[jmin]
                axpby!(0,s,1/ρmin,p) 
                axpby!(1,vec_x[jmin],0,x)
                axpby!(0,p,1/ρmin,s)  
                c1 /= ρmin
            end

            residual = abs(rr*ρMAX)


            #println(abs.(ρp))
    
            if abs(residual) < eps
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
                return x,vec_x
            end


        end
        error("bicg is not converged. The residual is $residual")

    end



end