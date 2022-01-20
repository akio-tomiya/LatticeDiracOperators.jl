module CGs

function shiftedcg(vec_x,vec_β,x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"shifted cg method")
    N = length(vec_β)
    temp1 = similar(b)
    r = deepcopy(b)
    p = deepcopy(b)
    q = similar(b)

    vec_r = Array{typeof(r),1}(undef,N)
    vec_p = Array{typeof(p),1}(undef,N)
    for j=1:N
        vec_r[j] = deepcopy(b)
        vec_p[j] = deepcopy(b)
    end

    αm = 1.0
    βm = 0.0

    ρm = ones(ComplexF64,N)
    ρ0 = ones(ComplexF64,N)
    ρp = ones(ComplexF64,N)
    residual = 0


    for i=1:maxsteps
        mul!(q,A,p)

        pAp = p ⋅ q

        rr = dot(r,r)
        αk = rr / pAp


        #! ...  x   = x   + alpha * p   
        add!(x,αk,p)

        #...  r = r - alpha * q 
        add!(r,-αk,q)

        βk = dot(r,r)/ rr
        add!(βk,p,1,r) #p = beta*p + r

        for j=1:N
            ρkj = ρ0[j]
            if abs(ρkj) < eps
                continue
            end
            ρkmj =ρm[j]
            ρp[j] = ρkj*ρkmj*αm/(ρkmj*αm*(1.0+αk*vec_β[j])+αk*βm*(ρkmj-ρkj))
            αkj = (ρp[j]/ρkj)*αk

            add!(vec_x[j],αkj,vec_p[j])
            βkj = (ρp[j]/ρkj)^2*βk
            add!(βkj,vec_p[j],ρp[j],r) 

        end

        ρm[:] = ρ0[:]
        ρ0[:] = ρp[:]
        αm = αk
        βm = βk


        ρMAX = maximum(abs.(ρp))^2
        residual = abs(rr*ρMAX)
        println_verbose_level3(verbose,"$i-th eps: $residual")

        if abs(residual) < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $residual")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end


    end

    
    error("""
    The shifted CG is not converged! with maxsteps = $(maxsteps)
    residual is $residual
    maxsteps should be larger.""")


end
end