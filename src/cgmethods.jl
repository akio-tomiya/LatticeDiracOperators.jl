

using LinearAlgebra
#import Gaugefields:Verbose_level,Verbose_3,Verbose_2,Verbose_1,println_verbose_level3
using InteractiveUtils
import Gaugefields.Verboseprint_mpi:Verbose_print,println_verbose_level1,println_verbose_level2,println_verbose_level3

#export bicg,bicgstag,shiftedcg,bicgstab_evenodd,reducedshiftedcg,cg


function add!(b,Y,a,X) #b*Y + a*X -> Y
    LinearAlgebra.axpby!(a,X,b,Y) #X*a + Y*b -> Y
end

function add!(Y,a,X) #Y + a*X -> Y
    LinearAlgebra.axpby!(a,X,1,Y) #X*a + Y -> Y
end


function add!(b,Y,a,X,iseven::Bool) #b*Y + a*X -> Y
    LinearAlgebra.axpby!(a,X,b,Y,iseven) #X*a + Y*b -> Y
end

function add!(Y,a,X,iseven::Bool) #Y + a*X -> Y
    LinearAlgebra.axpby!(a,X,1,Y,iseven) #X*a + Y -> Y
end



function bicg(x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    #println_verbose_level3()
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg method")
    #=
    res = deepcopy(b)
    temp1 = similar(x)
    p = similar(x)
    q = similar(x)
    s = similar(x)

    mul!(temp1,A,x)
    add!(res,-1,temp1)
    =#

    
    temps = get_temporaryvectors_forCG(A)
    res = temps[1]
    substitute_fermion!(res,b)
    temp1 = temps[2]
    mul!(temp1,A,x)
    add!(res,-1,temp1)
    p = temps[4]
    q = temps[5]
    s = temps[6]
    


    rnorm = real(res⋅res)
    if rnorm < eps
        return
    end
    #println(rnorm)

    mul!(p,A',res)
    c1 = p ⋅ p

    for i=1:maxsteps
        mul!(q,A,p)
        #! ...  c2 = < q | q >
        c2 = q ⋅ q
        
        alpha = c1 / c2
        #! ...  x   = x   + alpha * p  
        add!(x,alpha,p)
        #...  res = res - alpha * q 
        add!(res,-alpha,q)
        rnorm = real(res ⋅ res) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            #println("Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end

        mul!(s,A',res)

        #c3 = s * s
        c3 = s ⋅ s

        beta = c3 / c1
        c1 = c3

        add!(beta,p,1,s) #p = beta*p + s

    end

    
    error("""
    The BICG is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function bicgstab(x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg-stab method")

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
    

    
    temps = get_temporaryvectors_forCG(A)
    r = temps[1]
    substitute_fermion!(r,b)
    temp1 = temps[2]
    mul!(temp1,A,x)
    add!(r,-1,temp1)

    rs = temps[3]
    substitute_fermion!(rs,r)
    p = temps[4]
    substitute_fermion!(p,r)
    Ap = temps[5]
    s = temps[6]
    t = temps[7]
    
    


    rnorm = real(r⋅r)

    if rnorm < eps
        return
    end
    
    
    for i=1:maxsteps
        c1 = dot(rs,r)
        mul!(Ap,A,p)
        c2 = dot(rs,Ap)
        α = c1/c2
        #s = r - α*A*p
        add!(0,s,1,r)
        add!(s,-α,Ap)
        mul!(t,A,s)
        d1 = dot(t,s)
        d2 = dot(t,t)
        ω = d1/d2

        #r = (1-ω A)s
        add!(0,r,1,s)
        add!(r,-ω,t)

        #x = x + ωs+ αp
        add!(x,ω,s)
        add!(x,α,p)

        β = (dot(rs,r)/c1)*(α/ω)

        #p = r + β*(1-ωA)*p
        add!(β,p,1,r)
        add!(p,-ω*β,Ap)
        
        rnorm = real(r ⋅ r) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end



    end



    error("""
    The BICGstab is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function bicgstab_3(x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg-stab method")
    r = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    add!(r,-1,temp1)

    rnorm = real(r⋅r)

    if rnorm < eps
        return
    end

    #function test()

    rs = deepcopy(r)
    #rs = deepcopy(b)
    p = deepcopy(r)
    s = similar(r)
    t = similar(r)
    α = 0
    Ap = similar(p)
    As = similar(p)
    ω = 0

    for i=0:maxsteps
        mul!(Ap,A,p)
        c1 = dot(r,rs)
        α = c1/dot(Ap,rs)
        #s = r - alpha*v
        add!(0,s,1,r)
        add!(s,-α,Ap)

        mul!(As,A,s)
        ω = dot(As,s)/dot(As,As)
        add!(x,α,p)
        add!(x,ω,s)

        add!(0,r,1,s) # r = a*r + b*s (a = 0,b = 1)
        add!(r,-ω,t)

        β = (dot(r,rs)/c1)*α/ω

        #p = r + β*(p - ωm*vm)
        add!(β,p,1,r)
        add!(p,-ω*β,Ap)


        rnorm = real(r ⋅ r) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end
    end
    #end


    #= other
    rs = deepcopy(r)
    ρm = 1
    α = 1
    ωm = 1
    vm = similar(x)
    p = similar(x)
    s = similar(x)
    t = similar(x)

    println("rr ",dot(r,r))
    println("rsrs ",dot(rs,rs))
    println("rsr ",dot(rs,r))

    for i=1:maxsteps
        ρ = dot(rs,r)
        println(ρ    )
        β =(ρ/ρm)*(α/ωm)   
        #p = r + β*(p - ωm*vm)
        add!(β,p,1,r)
        add!(p,-ωm*β,vm)

        mul!(vm,A,p)
        α = ρ/dot(rs,vm)

        #s = r - alpha*v
        add!(0,s,1,r)
        add!(s,-α,vm)

        mul!(t,A,s)
        ωm = dot(t,s)/dot(r,r)

        add!(x,α,p)
        add!(x,ωm,s)

        add!(0,r,1,s) # r = a*r + b*s (a = 0,b = 1)
        add!(r,-ωm,t)

        rnorm = real(r ⋅ r) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end
    end

    =#

    #= text book
    p = deepcopy(r)
    s = similar(r)
    v = similar(x)
    t = similar(x)
    β = 0.0im    
    ω = 0.0im
    ρ2 = 0.0im
    α = 0.0im

    for i=0:maxsteps
        ρ = dot(rs,r)
        if i != 0
            β = α*ρ/(ω*ρ2)

            add!(β,p,1,r)
            add!(p,-ω*β,v) #p = r + beta*(p -omega*v)
        end
        mul!(v,A,p)
        c2 = dot(rs,v) 
        α = ρ/c2
        
        #s = r - α*v
        add!(0,s,1,r)
        add!(s,-α,v)

        rnorm = real(dot(s,s))
        if rnorm < eps
            add!(x,α,p)
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            
            return
        end
        mul!(t,A,s)

        ω = dot(t,s)/dot(t,t)
        #r = s - ω*t
        add!(0,r,1,s) # r = a*r + b*s (a = 0,b = 1)
        add!(s,-ω,t)

        add!(x,α,p)
        add!(x,ω,s)
        ρ2 = ρ

    end


 
    =#
    error("""
    The BICGstab is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function bicgstab_YN(x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg-stab method")
    r = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    add!(r,-1,temp1)

    rnorm = real(r⋅r)

    if rnorm < eps
        return
    end
    rs = deepcopy(r)
    p = deepcopy(r)
    s = similar(r)
    Ap = similar(x)
    As = similar(x)
    β = 0.0im    
    ω = 0.0im

    for i=0:maxsteps
        if i != 0
            add!(β,p,1,r)
            add!(p,-ω*β,Ap)
        end
        c1 = dot(rs,r)
        mul!(Ap,A,p)
        c2 = dot(rs,Ap)
        α = c1/c2
        add!(0,s,1,r)
        add!(s,-α,Ap)
        mul!(As,A,s)
        s1 = dot(As,s)
        s2 = dot(As,As)

        ω = s1/s2
        add!(x,α,p)
        add!(x,ω,s)
        add!(0,r,1,s)
        add!(r,-ω,As)

        rnorm = real(r ⋅ r) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end

        β = (α/ω)*dot(rs,r)/c1

    end


 
    
    error("""
    The BICG is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function bicgstab_evenodd(x,A,b,iseven;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg-stab even-odd method")
    r = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    #println(dot(temp1,temp1,iseven))
    add!(r,-1,temp1,iseven)

    rnorm = real(dot(r,r,iseven))


    if rnorm < eps
        return
    end

    rs = deepcopy(r)
    p = deepcopy(r)
    Ap = similar(r)
    s = similar(r)
    t = similar(r)
    
    
    for i=1:maxsteps
        c1 = dot(rs,r,iseven)
        mul!(Ap,A,p)
        c2 = dot(rs,Ap,iseven)
        α = c1/c2
        #s = r - α*A*p
        add!(0,s,1,r,iseven)
        add!(s,-α,Ap,iseven)
        mul!(t,A,s)
        d1 = dot(t,s,iseven)
        d2 = dot(t,t,iseven)
        ω = d1/d2

        #r = (1-ω A)s
        add!(0,r,1,s,iseven)
        add!(r,-ω,t,iseven)

        #x = x + ωs+ αp
        add!(x,ω,s,iseven)
        add!(x,α,p,iseven)

        β = (dot(rs,r,iseven)/c1)*(α/ω)

        #p = r + β*(1-ωA)*p
        add!(β,p,1,r,iseven)
        add!(p,-ω*β,Ap,iseven)
        
        rnorm = real(dot(r,r,iseven)) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end



    end



    error("""
    The BICGstab is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function bicgstab_evenodd_YN(x,A,b,iseven;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"bicg-stab even-odd method")

    r = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    println(dot(temp1,temp1,iseven))
    add!(r,-1,temp1,iseven)
    rs = deepcopy(r)
    p = deepcopy(r)
    s = similar(r)
    Ap = similar(x)
    As = similar(x)
    β = 0.0im    
    ω = 0.0im




    rnorm = real(dot(r,r,iseven))
    if rnorm < eps
        return
    end




    for i=0:maxsteps

        #=
        if i != 0
            add!(β,p,1,r,iseven)
            add!(p,-ω*β,Ap,iseven)
        end
        =#
        c1 = dot(rs,r,iseven)
        mul!(Ap,A,p)
        c2 = dot(rs,Ap,iseven)
        α = c1/c2
        add!(0,s,1,r,iseven)
        add!(s,-α,Ap,iseven)
        mul!(As,A,s)
        s1 = dot(As,s,iseven)
        s2 = dot(As,As,iseven)

        ω = s1/s2
        add!(x,α,p,iseven)
        add!(x,ω,s,iseven)
        add!(0,r,1,s,iseven)
        add!(r,-ω,As,iseven)

        rnorm = real(dot(r,r,iseven)) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")

        if rnorm < eps
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end

        β = (α/ω)*dot(rs,r,iseven)/c1

        #if i != 0
            add!(β,p,1,r,iseven)
            add!(p,-ω*β,Ap,iseven)
        #end
    end


 
    
    error("""
    The BICG is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function cg(x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    temps = get_temporaryvectors_forCG(A)

    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"cg method")

    #=
    res = deepcopy(b)
    temp1 = similar(x)
    mul!(temp1,A,x)
    add!(res,-1,temp1)
    q = similar(x)
    p = deepcopy(res)
    =#

        
    res = temps[1]
    substitute_fermion!(res,b)
    temp1 = temps[2]
    mul!(temp1,A,x)
    add!(res,-1,temp1)
    q = temps[3]
    p = temps[4]
    substitute_fermion!(p,res)

    
    
    
    


    #p = deepcopy(res)

    rnorm = real(res⋅res)
    #println(rnorm)

    if rnorm < eps
        return
    end

    c1 = p ⋅ p

    for i=1:maxsteps
        mul!(q,A,p)
        c2 = dot(p,q)
        #c2 = p ⋅ q
        
        α = c1 / c2
        #! ...  x   = x   + alpha * p  
        #println("add2")
        add!(x,α,p)
        #...  res = res - alpha * q 
        #println("add1")
        add!(res,-α,q)
        c3 = res ⋅ res
        rnorm = real(c3) 
        println_verbose_level3(verbose,"$i-th eps: $rnorm")
        
        

        if rnorm < eps
            #println("$i eps: $eps rnorm $rnorm")
            println_verbose_level3(verbose,"Converged at $i-th step. eps: $rnorm")
            println_verbose_level3(verbose,"--------------------------------------")
            return
        end

        β = c3 / c1
        c1 = c3

        #println("add3")
        add!(β,p,1,res) #p = beta*p + s

    end

    
    error("""
    The CG is not converged! with maxsteps = $(maxsteps)
    residual is $rnorm
    maxsteps should be larger.""")


end

function Base.:*(x::Array{T,1},y::Array{T,1}) where T <: Number
    return x'*y
end



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

function shiftedbicg(vec_x,vec_β,x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"shifted cg method")
    N = length(vec_β)
    temp1 = similar(b)
    r = deepcopy(b)
    p = deepcopy(b)
    q = similar(b)
    s = similar(b)


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

    mul!(p,A',r)
    c1 = p ⋅ p


    for i=1:maxsteps
        mul!(q,A,p)
        c2 = q ⋅ q
        alpha = c1 / c2
        rr = dot(r,r)
        #! ...  x   = x   + alpha * p  
        add!(x,alpha,p)
        #...  res = res - alpha * q 
        add!(r,-alpha,q)
        
 
        mul!(s,A',r)

        #c3 = s * s
        c3 = s ⋅ s

        beta = c3 / c1
        c1 = c3

        add!(beta,p,1,s) #p = beta*p + s

        αk = alpha
        βk = beta


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
            #add!(βkj,vec_p[j],ρp[j],r) 
            add!(βkj,vec_p[j],ρp[j],s) 

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


function reducedshiftedcg(leftvec,vec_β,x,A,b;eps=1e-10,maxsteps = 1000,verbose = Verbose_print(2)) #Ax=b
    println_verbose_level3(verbose,"--------------------------------------")
    println_verbose_level3(verbose,"shifted cg method")
    N = length(vec_β)
    temp1 = similar(b)
    r = deepcopy(b)
    p = deepcopy(b)
    q = similar(b)

    #=
    vec_r = Array{typeof(r),1}(undef,N)
    vec_p = Array{typeof(p),1}(undef,N)
    for j=1:N
        vec_r[j] = deepcopy(b)
        vec_p[j] = deepcopy(b)
    end
    =#

    Σ = leftvec ⋅ b

    θ = zeros(ComplexF64,N)
    Π = ones(ComplexF64,N) .* Σ


    αm = 1.0
    βm = 0.0

    ρm = ones(ComplexF64,N)
    ρ0 = ones(ComplexF64,N)
    ρp = ones(ComplexF64,N)
    residual = 0


    for i=1:maxsteps
        mul!(q,A,p)

        pAp = p ⋅ q
        rr = r*r
        αk = rr / pAp

        #! ...  x   = x   + alpha * p   
        add!(x,αk,p)

        #...  r = r - alpha * q 
        add!(r,-αk,q)

        βk = (r*r)/ rr
        add!(βk,p,1,r) #p = beta*p + r

        Σ = leftvec ⋅ r

        for j=1:N
            ρkj = ρ0[j]
            if abs(ρkj) < eps
                continue
            end
            ρkmj =ρm[j]
            ρp[j] = ρkj*ρkmj*αm/(ρkmj*αm*(1.0+αk*vec_β[j])+αk*βm*(ρkmj-ρkj))
            αkj = (ρp[j]/ρkj)*αk
            θ[j] += αkj*Π[j]

            #add!(vec_x[j],αkj,vec_p[j])
            βkj = (ρp[j]/ρkj)^2*βk
            Π[j] = βkj*Π[j] + ρp[j]*Σ

            #mul!(Π,ρp[j],Σ,1,βkj)
            #Π[:] = 
            #add!(βkj,vec_p[j],ρp[j],r) 

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
            return θ
        end


    end

    
    error("""
    The shifted CG is not converged! with maxsteps = $(maxsteps)
    residual is $residual
    maxsteps should be larger.""")


end

function shiftedbicg_2003(σ,A,b;maxsteps=3000,eps = 1e-15,verboselevel=2)
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
        if verboselevel == 3
            println("$i-th step: ",residual)
        end

        if abs(residual) < eps
            if verboselevel >= 2
                println("Converged at $i-th step. eps: $residual")
                println("--------------------------------------")
            end
            return vec_x
        end

        for k=1:N
            πkold[k] = πk[k]
            πk[k] = πknew[k]
        end

    end

    error("bicg is not converged. The residual is $residual")


end

