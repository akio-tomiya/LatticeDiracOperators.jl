
abstract type Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion} <: AbstractFermionfields_5D{NC}  end

"""
Struct for MobiusDomainwallFermion
"""
struct MobiusDomainwallFermion_5D{NC,WilsonFermion} <: Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}
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

    function MobiusDomainwallFermion_5D(L5,NC::T,NX::T,NY::T,NZ::T,NT::T;nowing = false) where T<: Integer
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
        Dirac_operator = "MobiusDomainwall"
        return new{NC,xtype}(w,NC,NX,NY,NZ,NT,L5,Dirac_operator,NWilson,nowing)
    end

end




function Base.similar(x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion} ) where {NC,WilsonFermion}
    return MobiusDomainwallFermion_5D(x.L5,NC,x.NX,x.NY,x.NZ,x.NT,nowing=x.nowing)
end

function apply_J!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
    clear_fermion!(xout)

    L5 = xout.L5
    for i5=1:L5
        j5 = L5-i5+1
        substitute_fermion!(xout.w[i5],x.w[j5])
    end
end

function apply_P!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
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

function apply_Pdag!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where  {NC,WilsonFermion}
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

function apply_1pD!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},L5,U::Array{G,1},A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},factor) where  {NC,WilsonFermion,G <: AbstractGaugefields}
    clear_fermion!(xout)

    if factor == 0
        for i5=1:xout.L5
            axpy!(1,x.w[i5],xout.w[i5])
        end
        return
    end

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
    ratio = 1

    for i5 in irange 
        j5=i5
        D4x!(xout.w[i5],U,x.w[j5],A,4) #Dw*x
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(1/(2*A.κ) + 1)
        set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(ratio,xout.w[i5],factor*ratio*massfactor,x.w[j5]) #D = x + Dw*x
        set_wing_fermion!(xout.w[i5])  
    end

    if L5 != xout.L5
        for i5 in irange_out
            axpy!(1,x.w[i5],xout.w[i5])
        end
    end

end

function apply_1pDdag!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},L5,U::Array{G,1},A,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},factor) where  {NC,WilsonFermion,G <: AbstractGaugefields}
    clear_fermion!(xout)

    if factor == 0
        for i5=1:xout.L5
            axpy!(1,x.w[i5],xout.w[i5])
        end
        return
    end

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
    ratio = 1

    for i5 in irange 
        j5=i5
        D4dagx!(xout.w[i5],U,x.w[j5],A,4) #Dw*x
        #Dx!(xout.w[i5],U,x.w[j5],A) #Dw*x
        #Wx!(xout.w[i5],U,x.w[j5],temps) #Dw*x
        #1/(2*A.κ)
        massfactor = -(1/(2*A.κ) + 1)
        set_wing_fermion!(xout.w[i5])
        #add!(ratio,xout.w[i5],ratio,x.w[j5]) #D = x + Ddagw*x
        add!(ratio,xout.w[i5],factor*ratio*massfactor,x.w[j5]) #D = x + Dw*x
        set_wing_fermion!(xout.w[i5])  
    end

    if L5 != xout.L5
        for i5 in irange_out
            axpy!(1,x.w[i5],xout.w[i5])
        end
    end

end

function apply_F!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},L5,m,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},temp1) where  {NC,WilsonFermion}
    clear_fermion!(xout)

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
    ratio = 1

    for i5 in irange 
        j5=i5+1
        if 1 <= j5 <= xout.L5
            #-P_- -> - P_+ :gamma_5 of LTK definition
            if xout.L5 != 2
                # ((1 + (b-c)/2 DW )*P_-*x)
                mul_1plusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end


        j5=i5-1
        if 1 <= j5 <= xout.L5
            #-P_+ -> - P_- :gamma_5 of LTK definition
            if xout.L5 != 2
                mul_1minusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end

        if xout.L5 != 1
            if i5==1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1minusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],-m*ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end

            if i5== xout.L5
                j5 = 1
                mul_1plusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],-m*ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end
    end

    #if L5 != xout.L5
    #    for i5 in irange_out
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

end

function apply_Fdag!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},L5,m,
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},temp1) where  {NC,WilsonFermion}
    clear_fermion!(xout)

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
    ratio = 1

    for i5 in irange 
        j5=i5+1
        if 1 <= j5 <= xout.L5
            #-P_- -> - P_+ :gamma_5 of LTK definition
            if xout.L5 != 2
                # ((1 + (b-c)/2 DW )*P_-*x)
                mul_1minusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end


        j5=i5-1
        if 1 <= j5 <= xout.L5
            #-P_+ -> - P_- :gamma_5 of LTK definition
            if xout.L5 != 2
                mul_1plusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end

        if xout.L5 != 1
            if i5==1
                j5 = xout.L5
                #mul_1plusγ5x_add!(xout.w[i5],x.w[j5],m*ratio) 
                mul_1plusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],-m*ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end

            if i5== xout.L5
                j5 = 1
                mul_1minusγ5x!(temp1.w[i5],x.w[j5])
                add!(ratio,xout.w[i5],-m*ratio,temp1.w[i5])
                set_wing_fermion!(xout.w[i5])  
            end
        end
    end

    #if L5 != xout.L5
    #    for i5 in irange_out
    #        axpy!(1,x.w[i5],xout.w[i5])
    #    end
    #end

end


function D5DWx!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},U::Array{G,1},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},m,A,L5,b,c,
        temp1,temp2) where  {NC,WilsonFermion,G <: AbstractGaugefields}


    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    coeff_plus = (b+c)/2
    #coeff_plus = 1
    coeff_minus = -(b-c)/2
    #coeff_minus  = 0
    clear_fermion!(xout)
    ratio = 1

    factor = coeff_plus 
    #xout = (1 + factor*D)*x
    apply_1pD!(xout,L5,U,A,x,factor) 

    #temp2 = F*x
    apply_F!(temp2,L5,m,x,temp1) 
    factor = coeff_minus
    #xout = (1 + factor*D)*F*x
    apply_1pD!(temp1,L5,U,A,temp2,factor) 
    for i5=1:L5
        axpy!(1,temp1.w[i5],xout.w[i5])
    end

    set_wing_fermion!(xout)   


    return
end

function D5DWdagx!(xout::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion} ,U::Array{G,1},
    x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion} ,m,A,L5,b,c,
    temp1,temp2) where  {NC,WilsonFermion,G <: AbstractGaugefields}

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    clear_fermion!(xout)
    ratio = 1
    coeff_plus = (b+c)/2
    coeff_minus = -(b-c)/2
    #coeff_plus = 1
    #coeff_minus  = 0
    #ratio = xout.L5/L5


    factor = coeff_minus
    #xout = Fdag*(1 + factor*Ddag)*x
    apply_1pDdag!(temp2,L5,U,A,x,factor)
    apply_Fdag!(xout,L5,m,temp2,temp1) 

    factor = coeff_plus 
    #xout = (1 + factor*Ddag)*x
    apply_1pDdag!(temp1,L5,U,A,x,factor) 

    for i5=1:L5
        axpy!(1,temp1.w[i5],xout.w[i5])
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
function gauss_distribution_fermion!(x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion}) where {NC,WilsonFermion}
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
function gauss_distribution_fermion!(x::Abstract_MobiusDomainwallFermion_5D{NC,WilsonFermion},randomfunc,σ) where {NC,WilsonFermion}
    L5 = length(x.w)
    for iL=1:L5
        gauss_distribution_fermion!(x.w[iL],randomfunc,σ)
    end
    return

end
