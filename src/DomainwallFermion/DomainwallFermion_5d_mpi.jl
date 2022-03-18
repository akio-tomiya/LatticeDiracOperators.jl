import Gaugefields:comm,setvalue!

"""
Struct for DomainwallFermion
"""
struct DomainwallFermion_5D_mpi{NC,WilsonFermion} <: Abstract_DomainwallFermion_5D{NC,WilsonFermion}
    w::Array{WilsonFermion,1}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    L5::Int64   
    Dirac_operator::String
    NWilson::Int64
    PEs::NTuple{4,Int64}
    PN::NTuple{4,Int64}
    mpiinit::Bool
    myrank::Int64
    nprocs::Int64
    myrank_xyzt::NTuple{4,Int64}
    mpi::Bool
    nowing::Bool

    function DomainwallFermion_5D_mpi(L5,NC::T,NX::T,NY::T,NZ::T,NT::T,PEs;nowing = false) where T<: Integer

        NV = NX*NY*NZ*NT
        @assert NX % PEs[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs"
        @assert NY % PEs[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs"
        @assert NZ % PEs[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs"
        @assert NT % PEs[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs"

        PN = (NX ÷ PEs[1],
                    NY ÷ PEs[2],
                    NZ ÷ PEs[3],
                    NT ÷ PEs[4],
            )

        nprocs = MPI.Comm_size(comm)
        @assert prod(PEs) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        myrank_xyzt = get_myrank_xyzt(myrank,PEs)

        #x = WilsonFermion_4D_mpi(NC,NX,NY,NZ,NT,PEs)

        if nowing
            x = WilsonFermion_4D_nowing_mpi(NC,NX,NY,NZ,NT,PEs)
            #error("Dirac_operator  = $Dirac_operator with nowing = $nowing is not supported")
        else
            x = WilsonFermion_4D_mpi(NC,NX,NY,NZ,NT,PEs)
        end


        xtype = typeof(x)
        w = Array{xtype,1}(undef,L5)
        w[1] = x
        for i=2:L5
            w[i] = similar(x)
        end
        #println(w[2][1,1,1,1,1,1])
        NWilson = length(x)
        Dirac_operator = "Domainwall"
        mpi = true
        mpiinit = true
        return new{NC,xtype}(w,NC,NX,NY,NZ,NT,L5,Dirac_operator,NWilson,
                Tuple(PEs),PN,mpiinit,myrank,nprocs,myrank_xyzt,mpi,nowing)
    end

end




function Base.similar(x::DomainwallFermion_5D_mpi{NC,WilsonFermion} ) where {NC,WilsonFermion}
    return DomainwallFermion_5D_mpi(x.L5,NC,x.NX,x.NY,x.NZ,x.NT,x.PEs,nowing=x.nowing)
end

#=

function D5DWx!(xout::DomainwallFermion_5D_mpi{NC,WilsonFermion} ,U::Array{G,1},
    x::DomainwallFermion_5D_mpi{NC,WilsonFermion} ,m,A,L5) where  {NC,WilsonFermion,G <: AbstractGaugefields}

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    clear_fermion!(xout)
    ratio = 1
    #ratio = xout.L5/L5
    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5
        
        for i5=1:xout.L5
            if i5 <= div(L5,2) || i5 >= xout.L5-div(L5,2)+1
                push!(irange,i5)
            else
                push!(irange_out,i5)
            end

        end
        
       
        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5  
    end
    

    for i5 in irange 
        j5=i5
        D4x!(xout.w[i5],U,x.w[j5],A,4) #Dw*x
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(1/(2*A.κ) + 1)
        set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(ratio,xout.w[i5],ratio*massfactor,x.w[j5]) #D = x + Dw*x
        set_wing_fermion!(xout.w[i5])  

        #println("xout ",xout.w[i5][1,1,1,1,1,1])

    
        j5=i5+1
        if 1 <= j5 <= xout.L5
            #-P_- -> - P_+ :gamma_5 of LTK definition
            if xout.L5 != 2
                #mul_1minusγ5x_add!(xout.w[i5],x.w[j5],-1*ratio) 
                mul_1plusγ5x_add!(xout.w[i5],x.w[j5],ratio) 
                set_wing_fermion!(xout.w[i5])  
                
            end
        end

        j5=i5-1
        if 1 <= j5 <= xout.L5
            #-P_+ -> - P_- :gamma_5 of LTK definition
            if xout.L5 != 2
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],-1*ratio) 
                mul_1minusγ5x_add!(xout.w[i5],x.w[j5],ratio) 
                set_wing_fermion!(xout.w[i5])  
            end
        end

        if xout.L5 != 1
            if i5==1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1minusγ5x_add!(xout.w[i5],x.w[j5],-m*ratio) 
                set_wing_fermion!(xout.w[i5])  
            end

            if i5== xout.L5
                j5 = 1
                #mul_1minusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1plusγ5x_add!(xout.w[i5],x.w[j5],-m*ratio) 
                set_wing_fermion!(xout.w[i5])  
            end
        end

        #println("xout ",xout.w[i5][1,1,1,1,1,1])

    end  
    


    if L5 != xout.L5
        for i5 in irange_out
            axpy!(1,x.w[i5],xout.w[i5])
        end
    end

    set_wing_fermion!(xout)   

    return
end



function D5DWdagx!(xout::DomainwallFermion_5D_mpi{NC,WilsonFermion} ,U::Array{G,1},
    x::DomainwallFermion_5D_mpi{NC,WilsonFermion} ,m,A,L5) where  {NC,WilsonFermion,G <: AbstractGaugefields}

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    clear_fermion!(xout)
    ratio = 1
    #ratio = xout.L5/L5

    if L5 != xout.L5
        @assert L5 % 2 == 0
        irange = Int64[]
        irange_out = Int64[]
        #irange = 1:L5
        #irange_out = (L5+1):xout.L5
        
        for i5=1:xout.L5
            if i5 <= div(L5,2) || i5 >= xout.L5-div(L5,2)+1
                push!(irange,i5)
            else
                push!(irange_out,i5)
            end

        end
        
        #for i5 in irange_out
        #    axpy!(1,x.w[i5],xout.w[i5])
        #end
    else
        irange = 1:L5  
    end
    


    for i5 in irange
        j5=i5
        #Ddagx!(xout.w[i5],U,x.w[j5],A) #Ddagw*x
        D4dagx!(xout.w[i5],U,x.w[j5],A,4) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(1/(2*A.κ) + 1)
        #println(massfactor)

        #Wdagx!(xout.w[i5],U,x.w[j5],temps) #Ddagw*x
        set_wing_fermion!(xout.w[i5])
        add!(ratio,xout.w[i5],ratio*massfactor,x.w[j5]) #D = x + Dw*x
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        set_wing_fermion!(xout.w[i5])  

    
        j5=i5+1
        if 1 <= j5 <= xout.L5
            #-P_-
            if xout.L5 != 2
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],-1*ratio) 
                mul_1minusγ5x_add!(xout.w[i5],x.w[j5],ratio) 
                set_wing_fermion!(xout.w[i5])  
            end
        end

        j5=i5-1
        if 1 <= j5 <= xout.L5
            #-P_+
            if xout.L5 != 2
                #mul_1minusγ5x_add!(xout.w[i5],x.w[j5],-1*ratio) 
                mul_1plusγ5x_add!(xout.w[i5],x.w[j5],ratio) 
                set_wing_fermion!(xout.w[i5])  
            end
        end

        if L5 != 1
            if i5==1
                j5 = xout.L5
                #mul_1minusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1plusγ5x_add!(xout.w[i5],x.w[j5],-m*ratio) 
                set_wing_fermion!(xout.w[i5])  
            end

            if i5==xout.L5
                j5 = 1
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1minusγ5x_add!(xout.w[i5],x.w[j5],-m*ratio) 
                set_wing_fermion!(xout.w[i5])  
            end
        end

    end  

    #if L5 != xout.L5
    #    for i5=L5+1:xout.L5
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

    if L5 != xout.L5
        for i5 in irange_out
            axpy!(1,x.w[i5],xout.w[i5])
        end
    end


    set_wing_fermion!(xout)   

    return
end


=#

#=

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::DomainwallFermion_5D_mpi{NC,NDW}) where {NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.w[1].f)[6]
    σ = sqrt(1/2)

    L5 = length(x.w)
    for iL = 1:L5
        for ialpha = 1:n6
            for it=1:x.PN[4]
                for iz=1:x.PN[3]
                    for iy=1:x.PN[2]
                        for ix=1:x.PN[1]
                            for ic=1:NC 
                                v = σ*randn()+im*σ*randn()
                                setvalue!(x.w[iL],v,ic,ix,iy,iz,it,ialpha)
                                #x[ic,ix,iy,iz,it,ialpha] = σ*randn()+im*σ*randn()
                            end
                        end
                    end
                end
            end
        end
        set_wing_fermion!(x.w[iL])
    end
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::DomainwallFermion_5D_mpi{NC,NDW},randomfunc,σ) where {NC,NDW}
  
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.w[1].f)[6]
    #σ = sqrt(1/2)

    L5 = length(x.w)
    for iL = 1:L5


        for mu = 1:n6
            for ic=1:NC
                for it=1:x.PN[4]
                    for iz=1:x.PN[3]
                        for iy=1:x.PN[2]
                            for ix=1:x.PN[1]
                                v1 = sqrt(-log(randomfunc()+1e-10))
                                v2 = 2pi*randomfunc()

                                xr = v1*cos(v2)
                                xi = v1 * sin(v2)

                                v = σ*xr + σ*im*xi
                                setvalue!(x.w[iL],v,ic,ix,iy,iz,it,mu)

                                #x[ic,ix,iy,iz,it,mu] = σ*xr + σ*im*xi
                            end
                        end
                    end
                end
            end
        end

        set_wing_fermion!(x.w[iL])
    end

    return
end


=#