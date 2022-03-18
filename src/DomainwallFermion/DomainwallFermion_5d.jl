
abstract type Abstract_DomainwallFermion_5D{NC,WilsonFermion} <: AbstractFermionfields_5D{NC}  end

"""
Struct for DomainwallFermion
"""
struct DomainwallFermion_5D{NC,WilsonFermion} <: Abstract_DomainwallFermion_5D{NC,WilsonFermion}
    w::Array{WilsonFermion,1}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    L5::Int64   
    Dirac_operator::String
    NWilson::Int64
    nowing::Bool

    function DomainwallFermion_5D(L5,NC::T,NX::T,NY::T,NZ::T,NT::T;nowing = false) where T<: Integer
        if nowing 
            x = WilsonFermion_4D_nowing(NC,NX,NY,NZ,NT)
        else
            x = WilsonFermion_4D_wing(NC,NX,NY,NZ,NT)
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
        return new{NC,xtype}(w,NC,NX,NY,NZ,NT,L5,Dirac_operator,NWilson,nowing)
    end

end




function Base.similar(x::Abstract_DomainwallFermion_5D{NC,WilsonFermion} ) where {NC,WilsonFermion}
    return DomainwallFermion_5D(x.L5,NC,x.NX,x.NY,x.NZ,x.NT,nowing=x.nowing)
end

function apply_J!(xout::Abstract_DomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_DomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
    clear_fermion!(xout)

    L5 = xout.L5
    for i5=1:L5
        j5 = L5-i5+1
        substitute_fermion!(xout.w[i5],x.w[j5])
    end
end

function apply_P!(xout::Abstract_DomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_DomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
    L5 = xout.L5
    clear_fermion!(xout)

    for i5=1:L5
        j5 = i5
        #P_- -> P_+ in this definition
        mul_1plusγ5x_add!(xout.w[i5],x.w[j5],1) 
        set_wing_fermion!(xout.w[i5])  

        #P_+ -> P_- in this definition
        j5 = i5+1
        j5 += ifelse(j5 > L5,-L5,0)
        mul_1minusγ5x_add!(xout.w[i5],x.w[j5],1) 
        set_wing_fermion!(xout.w[i5])  
    end
end

function apply_Pdag!(xout::Abstract_DomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_DomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
    L5 = xout.L5
    clear_fermion!(xout)

    for i5=1:L5
        j5 = i5
        #P_- -> P_+ in this definition
        mul_1plusγ5x_add!(xout.w[i5],x.w[j5],1) 
        set_wing_fermion!(xout.w[i5])  

        #P_+ -> P_- in this definition
        j5 = i5-1
        j5 += ifelse(j5 < 1,L5,0)
        mul_1minusγ5x_add!(xout.w[i5],x.w[j5],1) 
        set_wing_fermion!(xout.w[i5])  
    end
end


function D5DWx!(xout::Abstract_DomainwallFermion_5D{NC,WilsonFermion},U::Array{G,1},
    x::Abstract_DomainwallFermion_5D{NC,WilsonFermion},m,A,L5) where  {NC,WilsonFermion,G <: AbstractGaugefields}

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

function D5DWdagx!(xout::Abstract_DomainwallFermion_5D{NC,WilsonFermion} ,U::Array{G,1},
    x::Abstract_DomainwallFermion_5D{NC,WilsonFermion} ,m,A,L5) where  {NC,WilsonFermion,G <: AbstractGaugefields}

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


"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::Abstract_DomainwallFermion_5D{NC,WilsonFermion}) where {NC,WilsonFermion}
    L5 = length(x.w)
    for iL=1:L5
        gauss_distribution_fermion!(x.w[iL])
    end
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::Abstract_DomainwallFermion_5D{NC,WilsonFermion},randomfunc,σ) where {NC,WilsonFermion}
    L5 = length(x.w)
    for iL=1:L5
        gauss_distribution_fermion!(x.w[iL],randomfunc,σ)
    end
    return

end
