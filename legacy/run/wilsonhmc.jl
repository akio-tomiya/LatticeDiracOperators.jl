#include("../src/LatticeDiracOperators.jl")

using Gaugefields
using LinearAlgebra
using InteractiveUtils
using Random
using LatticeDiracOperators
import Gaugefields:Initialize_4DGaugefields
import LatticeDiracOperators.SSmodule:shiftedbicg_inSS,shiftedbicgstab_inSS,shiftedbicgstab,
            shiftedbicg_Frommer2003,shiftedbicg_Frommer2003_seed,shiftedbicg_Frommer2003_G_seed

function MDtest!(gauge_action,U,Dim,fermi_action,η,ξ)
    p = initialize_TA_Gaugefields(U) #This is a traceless-antihermitian gauge fields. This has NC^2-1 real coefficients. 
    Uold = similar(U)
    substitute_U!(Uold,U)
    MDsteps = 10
    temp1 = similar(U[1])
    temp2 = similar(U[1])
    comb = 6
    factor = 1/(comb*U[1].NV*U[1].NC)
    numaccepted = 0
    Random.seed!(123)

    numtrj = 6
    for itrj = 1:numtrj
       #@code_warntype MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        #error("cc")
        @time accepted = MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
        numaccepted += ifelse(accepted,1,0)

        plaq_t = calculate_Plaquette(U,temp1,temp2)*factor
        println("$itrj plaq_t = $plaq_t")
        println("acceptance ratio ",numaccepted/itrj)
    end
end

function calc_action(gauge_action,U,p)
    NC = U[1].NC
    Sg = -evaluate_GaugeAction(gauge_action,U)/NC #evaluate_GaugeAction(gauge_action,U) = tr(evaluate_GaugeAction_untraced(gauge_action,U))
    Sp = p*p/2
    S = Sp + Sg
    return real(S)
end


function MDstep!(gauge_action,U,p,MDsteps,Dim,Uold,fermi_action,η,ξ)
    Δτ = 1/MDsteps
    NC,_,NN... = size(U[1])
    
    
    for μ=1:Dim
        pwork = gauss_distribution(prod(NN)*(NC^2-1))
        substitute_U!(p[μ],pwork)
    end
    
    #println(p[1][1,1,1,1,1,1])

    #gauss_distribution!(p)
    
    substitute_U!(Uold,U)
    gauss_sampling_in_action!(ξ,U,fermi_action)
    sample_pseudofermions!(η,U,fermi_action,ξ)
    Sfold = real(dot(ξ,ξ))
    println("Sfold = $Sfold")

    #@code_warntype calc_action(gauge_action,U,p) 

    Sold = calc_action(gauge_action,U,p) + Sfold
    println("Sold = ",Sold)

    for itrj=1:MDsteps
        U_update!(U,p,0.5,Δτ,Dim,gauge_action)

        P_update!(U,p,1.0,Δτ,Dim,gauge_action)
        #println(" U1 = ", U[1][1,1,1,1,1,1])
        #        println(" p = ", p[1][1,1,1,1,1])
        P_update_fermion!(U,p,1.0,Δτ,Dim,gauge_action,fermi_action,η)
        #error("dd")

        U_update!(U,p,0.5,Δτ,Dim,gauge_action)
    end
    Sfnew = evaluate_FermiAction(fermi_action,U,η)
    println("Sfnew = $Sfnew")
    Snew = calc_action(gauge_action,U,p) + Sfnew
    
    println("Sold = $Sold, Snew = $Snew")
    println("Snew - Sold = $(Snew-Sold)")

    accept = exp(Sold - Snew) >= rand()

    #ratio = min(1,exp(Snew-Sold))
    if accept != true #rand() > ratio
        substitute_U!(U,Uold)
        return false
    else
        return true
    end
end

function U_update!(U,p,ϵ,Δτ,Dim,gauge_action)
    temps = get_temporary_gaugefields(gauge_action)
    temp1 = temps[1]
    temp2 = temps[2]
    expU = temps[3]
    W = temps[4]

    for μ=1:Dim
        exptU!(expU,ϵ*Δτ,p[μ],[temp1,temp2])
        mul!(W,expU,U[μ])
        substitute_U!(U[μ],W)
        
    end
end

function P_update!(U,p,ϵ,Δτ,Dim,gauge_action) # p -> p +factor*U*dSdUμ
    NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    dSdUμ = temps[end]
    factor =  -ϵ*Δτ/(NC)

    for μ=1:Dim
        calc_dSdUμ!(dSdUμ,gauge_action,μ,U)
        mul!(temps[1],U[μ],dSdUμ) # U*dSdUμ
        Traceless_antihermitian_add!(p[μ],factor,temps[1])
    end
end

function P_update_fermion!(U,p,ϵ,Δτ,Dim,gauge_action,fermi_action,η)  # p -> p +factor*U*dSdUμ
    #NC = U[1].NC
    temps = get_temporary_gaugefields(gauge_action)
    UdSfdUμ = temps[1:Dim]
    factor =  -ϵ*Δτ

    #calc_p_UdSfdU!(p,fermi_action,U,η,factor)

    calc_UdSfdU!(UdSfdUμ,fermi_action,U,η)

    for μ=1:Dim
        Traceless_antihermitian_add!(p[μ],factor,UdSfdUμ[μ])
        #println(" p[μ] = ", p[μ][1,1,1,1,1])
    end
    
end

function test1()
    NX = 4
    NY = 4
    NZ = 4
    NT = 4
    Nwing = 0
    Dim = 4
    NC = 3

    U = Initialize_4DGaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")
    #U  =Initialize_Gaugefields(NC,Nwing,NX,NY,NZ,NT,condition = "cold")


    gauge_action = GaugeAction(U)
    plaqloop = make_loops_fromname("plaquette")
    append!(plaqloop,plaqloop')
    β = 5.5/2
    push!(gauge_action,β,plaqloop)
    
    show(gauge_action)

    #Initialize_4DWilsonFermion(U[1])
    #error("init")
    #@code_warntype Initialize_pseudofermion_fields(U[1],"Wilson")
    #error("init")
    

    #x = Initialize_pseudofermion_fields(U[1],params)

    #x = Initialize_4DWilsonFermion(U[1])

    #=
    xplus = shift_fermion(x,1)
    

    ix = 1
    iy = 1
    iz = 1
    it = 1
    ialpha  =1
    xplus[1,ix,iy,iz,it,ialpha]
    println("dd")

    @inbounds for ix= 1:2
         @code_llvm x.f[1,ix,iy,iz,it,ialpha]
    end

    vv = rand(ComplexF64,2,2,2,2,2,2)
    @inbounds for ix= 1:2
        @code_llvm vv[1,ix,iy,iz,it,ialpha]
    end

    #error("err")
    @code_warntype shift_fermion(x,1)
    #error("2")
    ix = 1
    iy = 1
    iz = 1
    it = 1
    ialpha  =1
    @time @inbounds x.f[1,ix,iy,iz,it,ialpha]
    @code_llvm   @inbounds x.f[1,ix,iy,iz,it,ialpha]
    println("dd")
    @code_lowered   Base.getindex(x,1,ix,iy,iz,it,ialpha)
    println("dt")
    println(x[1,ix,iy,iz,it,ialpha])
    @code_llvm   @inbounds  x[1,ix,iy,iz,it,ialpha]

    @code_llvm  x[1,ix,iy,iz,it,ialpha]

    @code_llvm xplus[1,ix,iy,iz,it,ialpha]
    println(xplus[1,ix,iy,iz,it,ialpha])
    @code_llvm xplus[1,ix,iy,iz,it,ialpha]

    @code_warntype xplus[1,ix,iy,iz,it,ialpha]

    @time xplus[1,ix,iy,iz,it,ialpha]
    for ix = 1:4
        @time xplus.parent[1,ix,iy,iz,it,ialpha]
       # @time xplus[1,ix,iy,iz,it,ialpha]
    end
    #error("dd")
    #return 

    =#

    x = Initialize_pseudofermion_fields(U[1],"Wilson",nowing = true)


    params = Dict()
    params["Dirac_operator"] = "Wilson"
    M = -1
    params["κ"] = 1/(2*Dim+M) #0.141139/2
    params["eps_CG"] = 1.0e-19
    params["verbose_level"] = 2
    #params["method_CG"] = "preconditiond_bicgstab"
    #params["method_CG"] = "bicgstab"
    params["faster version"] = true
    params["method_CG"] = "bicg"
    D = Dirac_operator(U,x,params)

    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)
    #gauss_sampling_in_action!(x,U,fermi_action)
    #println("Sfold = ", dot(x,x))
    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)




    #g5D = γ5D(D(U))
    g5D = D(U)
        
   

    radius = 0.2
    origin = 0
    #es,residuals,eigenvectors,is = eigensystem_old(g5D,x,length(x),radius,origin)
    es,residuals,eigenvectors,is = eigensystem(g5D,x,radius,origin,Z2_distribution_fermion!,
                uniform_distribution_fermion!,α=1,epsCG=1e-20,κ=1.5)
    #println(result[1])
    fp = open("eigenvalues_GGwilson_SS.txt","w")
    println("es")
    println(es)
    println("es")
    println(residuals)
    for i = 1:length(es)
        println(fp,real(es[i]),"\t",imag(es[i]),"\t",real(residuals[i]))
        println(real(es[i]),"\t",imag(es[i]),"\t",real(residuals[i]))
    end
    close(fp)

    
    Dmat = construct_sparsematrix(g5D)
    e2,v = eigen(Matrix(Dmat))
    fp = open("eigenvalues_GGwilson_2.txt","w")
    for i = 1:length(e2)
        println(fp,real(e2[i]),"\t",imag(e2[i]))
        println(real(e2[i]),"\t",imag(e2[i]))
    end
    close(fp)

    return
    

    b = similar(x)
    for i in 1:length(b.f)
        b.f[i] = rand()
    end



    #=
    y = zero(x)

    for i= 1:length(e2)

        c = 0 
        for j=1:length(e2)
            c += v[j,i]'*b.f[j]
        end

        for j= 1:length(e2)
            y.f[j] += sign(e2[i])*v[j,i]*c
        end
    end
    =#

    Nq = 150
    α = 0.1
    radius = 0.995
    origin_p = 1.01
    origin_m = -1.01

    vec_z = zeros(ComplexF64,2Nq)
    vec_w = zeros(ComplexF64,2Nq)
    for j in 1:Nq
        θj = (2π/Nq)*(j-1/2) # π/Nq   2π-π/Nq
        vec_z[j] = origin_p + radius*(cos(θj)+im*α*sin(θj))
        vec_w[j] = α*cos(θj)+im*sin(θj)

        vec_z[j+Nq] = origin_m + radius*(cos(θj)+im*α*sin(θj))
        vec_w[j+Nq] = α*cos(θj)+im*sin(θj)
    end

    y2 = zero(x)
    zseedin = origin_p+0.05im
    eps = 1e-20
    @time vec_x = shiftedbicg_Frommer2003_G_seed(vec_z,g5D,b,eps =eps,zseedin = zseedin) #(z I - A) x = b


    for j in 1:Nq
        coeff = radius*vec_w[j]/Nq
        axpy!(coeff,vec_x[j],y2)
        coeff = -radius*vec_w[j+Nq]/Nq
        axpy!(coeff,vec_x[j+Nq],y2)
        #vec_Sk[:,ik] += ρ*vec_w[j]*vec_z[j]^k*vec_y[:,j]/Nq
    end

    println(dot(b,b),"\t",dot(y2,y2))
    value = abs(dot(b,b) - dot(y2,y2))/abs(dot(b,b))
    println("value = $value")
    return

    fp = open("signHx_$(Nq).txt","w")

    for i= 1:length(e2)
        println(fp,real(y.f[i]),"\t",imag(y.f[i]),"\t",
                    real(y2.f[i]),"\t",imag(y2.f[i])
                    )
    end
    close(fp)

    println(dot(b,b),"\t",dot(y2,y2))
    value = abs(dot(b,b) - dot(y2,y2))/abs(dot(b,b))
    println("value = $value")


    axpby!(1,y,-1,y2)
    println(sqrt(real(dot(y2,y2)))/sqrt(real(dot(y,y))))


    return





    
    σi = zeros(ComplexF64,Nq)
    rmin = -2.0
    rmax = -rmin
    delta = 0.05
    dr = (rmax-rmin)/(div(Nq,2)-1)
    for j = 1:div(Nq,2)
        σi[j] =  (j-1)*dr + rmin + im*delta
        σi[j+div(Nq,2)] = conj(σi[j])
    end

    y2 = zero(x)
    zseedin = 1+0.05im

    vec_x = shiftedbicg_Frommer2003_G_seed(σi,g5D,b,eps =1e-20,zseedin = zseedin) #(z I - A) x = b

    for j = 1:div(Nq,2)
        si = ifelse( real(σi[j]) > 0,1,-1)
        axpy!(-si*dr/(2π*im),vec_x[j],y2)
        axpy!(si*dr/(2π*im),vec_x[j+div(Nq,2)],y2)
    end

    fp = open("signHx.txt","w")

    for i= 1:length(e2)
        println(fp,real(y.f[i]),"\t",imag(y.f[i]),"\t",
                    real(y2.f[i]),"\t",imag(y2.f[i])
                    )
    end
    close(fp)


    axpby!(1,y,-1,y2)
    println(sqrt(real(dot(y2,y2)))/sqrt(real(dot(y,y))))



    return


    a = zero(x)
    for i= 1:length(e2)
        for j= 1:length(e2)
            a.f[i] += conj(v[j,i])*b.f[j]
        end
    end

    y2 = zero(x)
    zseedin = 1+0.05im

    α = 0.1
    radius = 0.99
    origin = 1

    vec_z = zeros(ComplexF64,Nq)
    vec_w = zeros(ComplexF64,Nq)
    for j in 1:Nq
        θj = (2π/Nq)*(j-1/2) # π/Nq   2π-π/Nq
        vec_z[j] = origin + radius*(cos(θj)+im*α*sin(θj))
        vec_w[j] = α*cos(θj)+im*sin(θj)
    end

    y_z = zero(x)
    a_z = zero(x)

    for i=1:length(e2)
        for j in 1:Nq
            coeff = radius*vec_w[j]/Nq
            #coeff = radius*vec_w[j]*sign(vec_z[j])/Nq
            a_z.f[i] += coeff*a.f[i]/(vec_z[j]-e2[i])
        end
    end


    for α= 1:length(e2)
        for i= 1:length(e2)
            y_z.f[α] += v[α,i]*a_z.f[i]
        end
    end

    radius = 0.99
    origin = -1

    vec_z = zeros(ComplexF64,Nq)
    vec_w = zeros(ComplexF64,Nq)
    for j in 1:Nq
        θj = (2π/Nq)*(j-1/2) # π/Nq   2π-π/Nq
        vec_z[j] = origin + radius*(cos(θj)+im*α*sin(θj))
        vec_w[j] = α*cos(θj)+im*sin(θj)
    end

    a_z = zero(x)

    for i=1:length(e2)
        for j in 1:Nq
            #coeff = radius*vec_w[j]*sign(vec_z[j])/Nq
            coeff = radius*vec_w[j]/Nq
            a_z.f[i] += coeff*a.f[i]/(vec_z[j]-e2[i])
        end
    end

    for α= 1:length(e2)
        for i= 1:length(e2)
            y_z.f[α] += -v[α,i]*a_z.f[i]
        end
    end

#=
    fp = open("signHx_z.txt","w")
    for i= 1:length(e2)
        println(fp,real(y.f[i]),"\t",imag(y.f[i]),"\t",real(y_z.f[i]),"\t",imag(y_z.f[i]))
        println(real(y.f[i]),"\t",imag(y.f[i]),"\t",real(y_z.f[i]),"\t",imag(y_z.f[i]))

    end
    close(fp)
    

    axpby!(1,y,-1,y_z)
    println(sqrt(real(dot(y_z,y_z)))/sqrt(real(dot(y,y))))
    return
    =#


    #vec_x = shiftedbicg_Frommer2003_G_seed(σi,g5D,b,eps =1e-20,zseedin = zseedin) #(z I - A) x = b
    #(z I - A) x = b -> (-z I + A)(-x) = b
    vec_x = shiftedbicg_Frommer2003_G_seed(vec_z,g5D,b,eps =1e-20,zseedin = zseedin) #(z I - A) x = b
    
    y3 = zero(x)

    for i=1:length(σi)
        println("i = $i σi = $(vec_z[i])")
        mul!(y3,g5D,vec_x[i])
        axpby!(vec_z[i],vec_x[i], -1, y3) #Overwrite Y with X*a + Y*b
        #axpby!(σi[i],vec_x[i], -1, y3) #Overwrite Y with X*a + Y*b

        axpby!(-1,b, 1, y3) #Overwrite Y with X*a + Y*b
        #add_fermion!(y,-σi[i],vec_x[i])
        println(dot(y3,y3))
    end

    #=
    for j = 1:div(Nq,2)
        axpy!(-sign(σi[j])*dr/(2π*im),vec_x[j],y2)
        axpy!(sign(σi[j+div(Nq,2)])*dr/(2π*im),vec_x[j+div(Nq,2)],y2)
    end
    =#

    
    for j in 1:Nq
        coeff = radius*vec_w[j]*sign(vec_z[j])/Nq
        axpy!(coeff,vec_x[j],y2)
        #vec_Sk[:,ik] += ρ*vec_w[j]*vec_z[j]^k*vec_y[:,j]/Nq
    end
   
    


    fp = open("signHx.txt","w")

    for i= 1:length(e2)

        #println(y.f[i],"\t",y2.f[i])
        println(fp,real(y.f[i]),"\t",imag(y.f[i]),"\t",
                    real(y2.f[i]),"\t",imag(y2.f[i]),"\t",
                    real(y_z.f[i]),"\t",imag(y_z.f[i])
                    )
    end
    close(fp)

    axpby!(1,y,-1,y2)
    println(sqrt(real(dot(y2,y2)))/sqrt(real(dot(y,y))))

    #=
    fp = open("signHx_z.txt","w")
    for i= 1:length(e2)
        println(fp,real(y.f[i]),"\t",imag(y.f[i]),"\t",real(y_z.f[i]),"\t",imag(y_z.f[i]))
        println(real(y.f[i]),"\t",imag(y.f[i]),"\t",real(y_z.f[i]),"\t",imag(y_z.f[i]))

    end
    close(fp)
    =#

    axpby!(1,y,-1,y_z)
    println(sqrt(real(dot(y_z,y_z)))/sqrt(real(dot(y,y))))
    return

    return

    for j = 1:div(Nq,2)
        axpy!(-sign(σi[j])*dr/(2π*im),vec_x[j],y2)
        axpy!(sign(σi[j+div(Nq,2)])*dr/(2π*im),vec_x[j+div(Nq,2)],y2)
    end
    #println(dot(y,y2))

    for i= 1:length(e2)
        println(y.f[i],"\t",y2.f[i])
    end
    println(dot(y,y2))
    return



    radius = 0.1
    origin = 0.75
    result = eigensystem(g5D,x,radius,origin,Z2_distribution_fermion!,
                uniform_distribution_fermion!,α=1,epsCG=1e-20,κ=1.5)
    println(result[1])

    return


    radius = 0.03
    origin = 1
    result = eigensystem(D(U),x,radius,origin,Z2_distribution_fermion!,
                uniform_distribution_fermion!,α=1,epsCG=1e-20,κ=1.5)
    println(result[1])


    b = similar(x)
    y = similar(x)
    rtin = similar(x)

    for i in 1:length(b.f)
        b.f[i] = rand([-1,1])
        rtin.f[i] = rand([-1,1])
    end
    #b[1,1,1,1,1,1] = 1
    #b[1,1,1,1,1,2] = 1

    #shiftedbicgstab(x,D,b)
    #mul!(y,D,x)
    #axpby!(-1,b, 1, y) 
    #println(dot(y,y))
    #return


    N = 10
    Nq = N
    zseed = 0.1
    γ = 1
    ρ = 0.1
    α= 1

    σi = zeros(ComplexF64,Nq)

    for j = 1:Nq
        θ = (2π/Nq)*(j-1/2)
        σi[j] = γ + ρ*(cos(θ)+im*sin(θ))
    end


    #σi = rand(ComplexF64,N)

    vec_x = shiftedbicg_Frommer2003_seed(σi,D(U),b,zseed=σi[1],eps =1e-20) #(zI - D)*x = b
    #vec_x = shiftedbicg_Frommer2003(σi,D(U),b) 
    for i=1:length(σi)
        println("i = $i σi = $(σi[i])")
        mul!(y,D,vec_x[i])
        axpby!(σi[i],vec_x[i], 1, y) #Overwrite Y with X*a + Y*b
        axpby!(-1,b, 1, y) #Overwrite Y with X*a + Y*b
        #add_fermion!(y,-σi[i],vec_x[i])
        println(dot(y,y))
    end
    return

    #=
    for i=1:length(σi)
        println("i = $i σi = $(σi[i])")
        mul!(y,D,vec_x[i])
        axpby!(σi[i],vec_x[i], -1, y) #Overwrite Y with X*a + Y*b
        axpby!(-1,b, 1, y) #Overwrite Y with X*a + Y*b
        #add_fermion!(y,-σi[i],vec_x[i])
        println(dot(y,y))
    end

    return

    x,vec_x = shiftedbicg_inSS(σi,D,b,eps=1e-28,zseedin=zseed)
    mul!(y,D,x)
    axpy!(zseed,x,y)
    axpby!(-1,b, 1, y) 
    println(dot(y,y))
    #println(dot(x,vec_x[1]))
    for i=1:length(σi)
        println("i = $i σi = $(σi[i])")
        mul!(y,D,vec_x[i])
        axpby!(σi[i],vec_x[i], 1, y) #Overwrite Y with X*a + Y*b
        axpby!(-1,b, 1, y) #Overwrite Y with X*a + Y*b
        #add_fermion!(y,-σi[i],vec_x[i])
        println(dot(y,y))
    end
    return
    
    =#


    result = eigensystem(D(U),x,length(x),0.05,1,α=1,epsCG=1e-20,κ=1.5)
    println(result[1])
    e = result[1]

    fp = open("eigenvalues_SS_wilson.txt","w")
    for i = 1:length(e)
        println(fp,real(e[i]),"\t",imag(e[i]))
        println(real(e[i]),"\t",imag(e[i]))
    end
    close(fp)
    

    Dmat = construct_sparsematrix(D(U))
    e2,v = eigen(Matrix(Dmat))
    fp = open("eigenvalues_wilson.txt","w")
    for i = 1:length(e2)
        println(fp,real(e2[i]),"\t",imag(e2[i]))
        println(real(e2[i]),"\t",imag(e2[i]))
    end
    close(fp)
    
    return

    parameters_action = Dict()
    fermi_action = FermiAction(D,parameters_action)
    #gauss_sampling_in_action!(x,U,fermi_action)
    #println("Sfold = ", dot(x,x))
    y = similar(x)

    
    MDtest!(gauge_action,U,Dim,fermi_action,x,y)

end

function gauss_distribution(nv) 
    variance = 1
    nvh = div(nv,2)
    granf = zeros(Float64,nv)
    for i=1:nvh
        rho = sqrt(-2*log(rand())*variance)
        theta = 2pi*rand()
        granf[i] = rho*cos(theta)
        granf[i+nvh] = rho*sin(theta)
    end
    if 2*nvh == nv
        return granf
    end

    granf[nv] = sqrt(-2*log(rand())*variance) * cos(2pi*urand())
    return granf
end



test1()