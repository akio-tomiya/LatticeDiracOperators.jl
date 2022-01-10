import Base

"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_wing{NC,NDW} <: AbstractFermionfields_4D{NC}
    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    #BoundaryCondition::Vector{Int8}


    function WilsonFermion_4D_wing(NC::T,NX::T,NY::T,NZ::T,NT::T) where T<: Integer
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW,NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW}(f,NC,NX,NY,NZ,NT,NG,NDW,Dirac_operator)
    end

    function WilsonFermion_4D_wing{NC}(NX::T,NY::T,NZ::T,NT::T) where {T<: Integer,NC}
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW,NG)
        Dirac_operator = "Wilson"
        return new{NC,NDW}(f,NC,NX,NY,NZ,NT,NG,NDW,Dirac_operator)
    end


end

function Base.setindex!(x::WilsonFermion_4D_wing{NC,NDW},v,i1,i2,i3,i4,i5,i6)  where {NC,NDW}
    @inbounds x.f[i1,i2 + NDW,i3 + NDW,i4 + NDW,i5 + NDW,i6] = v
end

function Base.getindex(x::WilsonFermion_4D_wing{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
    @inbounds return x.f[i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6]
end

function Base.getindex(x::WilsonFermion_4D_wing{NC,NDW},i1::N,i2::N,i3::N,i4::N,i5::N,i6::N) where {NC,NDW,N<: Integer}
    return @inbounds x.f[i1,i2 + NDW,i3 + NDW,i4 + NDW,i5 + NDW,i6]
end


#=
function Base.getindex(F::Shifted_fermionfields_4D{NC,WilsonFermion_4D_wing{NC,NDW}},i1::N,i2::N,i3::N,i4::N,i5::N,i6::N)  where {NC,NDW,N <: Integer}
    
    @inbounds begin
    si2 = i2 + NDW  + F.shift[1]
    si3 = i3 + NDW  + F.shift[2]
    si4 = i4 + NDW  + F.shift[3]
    si5 = i5 + NDW  + F.shift[4]
    end
    #v = F.parent.f
    #return  v[i1,si2,si3,si4,si5,i6]
    
   return  @inbounds F.parent.f[i1,si2,si3,si4,si5,i6]
end
=#

#=

function Base.getindex(x::WilsonFermion_4D_wing{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
    #=
    i2new = i2 .+ x.NDW
    i3new = i3 .+ x.NDW
    i4new = i4 .+ x.NDW
    i5new = i5 .+ x.NDW
    @inbounds return x.f[i1,i2new,i3new,i4new,i5new,i6]
    =#
    #return @inbounds Base.getindex(x.f,i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6)
    @inbounds return x.f[i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6]
end

=#

function Base.similar(x::T) where T <: WilsonFermion_4D_wing
    return WilsonFermion_4D_wing(x.NC,x.NX,x.NY,x.NZ,x.NT)
end

#=
function Base.getindex(x::T,i1,i2,i3,i4,i5,i6) where T <: WilsonFermion_4D_wing{NC} where NC
    i2new = i2 + x.NDW
    i3new = i3 + x.NDW
    i4new = i4 + x.NDW
    i5new = i5 + x.NDW
    f1 = x.f
    #println((i1,i2new,i3new,i4new,i5new,i6))
    @inbounds v = f1[i1,i2new,i3new,i4new,i5new,i6]
    #x.f[i1,i2new,i3new,i4new,i5new,i6]
    return v
    @inbounds return x.f[i1,i2new,i3new,i4new,i5new,i6]
    #@inbounds return x.f[i1,i2 .+ x.NDW,i3 .+ x.NDW,i4 .+ x.NDW,i5 .+ x.NDW,i6]
end
=#

function set_wing_fermion!(a::WilsonFermion_4D_wing{NC,NDW},boundarycondition) where {NC,NDW} 
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha=1:4
        for it=1:NT
            for iz = 1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,0,iy,iz,it,ialpha] = boundarycondition[1]*a[k,NX,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,NX+1,iy,iz,it,ialpha] = boundarycondition[1]*a[k,1,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    #Y-direction
    for ialpha = 1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,0,iz,it,ialpha] = boundarycondition[2]*a[k,ix,NY,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,NY+1,iz,it,ialpha] = boundarycondition[2]*a[k,ix,1,iz,it,ialpha]
                    end
                end
            end
        end
    end

    
    for ialpha=1:4
        # Z-direction
        for it=1:NT
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,0,it,ialpha] = boundarycondition[3]*a[k,ix,iy,NZ,it,ialpha]
                        a[k,ix,iy,NZ+1,it,ialpha] = boundarycondition[3]*a[k,ix,iy,1,it,ialpha]

                    end
                end
            end
        end

        #T-direction
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,iz,0,ialpha] = boundarycondition[4]*a[k,ix,iy,iz,NT,ialpha]
                        a[k,ix,iy,iz,NT+1,ialpha] =boundarycondition[4]*a[k,ix,iy,iz,1,ialpha]
                    end
                end
            end
        end

    end

end




function Wx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    #temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]

    clear_fermion!(temp)
    #set_wing_fermion!(x)
    for ν=1:4
        
        xplus = shift_fermion(x,ν)
        #println(xplus)
        

        mul!(temp1,U[ν],xplus)
       

        #fermion_shift!(temp1,U,ν,x)

        #... Dirac multiplication

        mul!(temp1,view(A.rminusγ,:,:,ν))

        

        xminus = shift_fermion(x,-ν)
        Uminus = shift_U(U[ν],-ν)


        mul!(temp2,Uminus',xminus)
     
        #
        #fermion_shift!(temp2,U,-ν,x)
        #mul!(temp2,view(x.rplusγ,:,:,ν),temp2)
        mul!(temp2,view(A.rplusγ,:,:,ν))

        add_fermion!(temp,A.hopp[ν],temp1,A.hopm[ν],temp2)

    end

    clear_fermion!(xout)
    add_fermion!(xout,1,x,-1,temp)

    set_wing_fermion!(xout,A.boundarycondition)

    #display(xout)
    #    exit()
    return
end



function Wdagx!(xout::T,U::Array{G,1},
    x::T,A) where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    #,temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields}
    temp = A._temporary_fermi[4] #temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    clear_fermion!(temp)
    #set_wing_fermion!(x)
    for ν=1:4
        xplus = shift_fermion(x,ν)
        mul!(temp1,U[ν],xplus)

        #fermion_shift!(temp1,U,ν,x)

        #... Dirac multiplication
        #mul!(temp1,view(x.rminusγ,:,:,ν),temp1)
        mul!(temp1,view(A.rplusγ,:,:,ν))
        
        
        #
        xminus = shift_fermion(x,-ν)
        Uminus = shift_U(U[ν],-ν)

        mul!(temp2,Uminus',xminus)
        #fermion_shift!(temp2,U,-ν,x)
        #mul!(temp2,view(x.rminusγ,:,:,ν),temp2)
        mul!(temp2,view(A.rminusγ,:,:,ν))


        add_fermion!(temp,A.hopp[ν],temp1,A.hopm[ν],temp2)
        
        
        
    end

    clear_fermion!(xout)
    add_fermion!(xout,1,x,-1,temp)
    set_wing_fermion!(xout,A.boundarycondition)

    #display(xout)
    #    exit()
    return
end

function calc_beff!(xout,U,x,A) #be + K Teo bo
    iseven = false
    temp = A._temporary_fermi[4]#temps[4]
    Toex!(temp,U,x,A,iseven) 

    iseven = true
    add_fermion!(xout,1,x,A.κ,temp,iseven)

end

function Toex!(xout::T,U::Array{G,1},x::T,A,iseven)  where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields} #T_oe xe
    #temp = A._temporary_fermi[4]#temps[4]
    temp1 = A._temporary_fermi[1] #temps[1]
    temp2 = A._temporary_fermi[2] #temps[2]

    #temp = temps[4]
    #temp1 = temps[1]
    #temp2 = temps[2]
    if iseven 
        isodd = false
    else
        isodd  =true
    end

    #clear_fermion!(temp,isodd)
    clear_fermion!(xout,isodd)
    #set_wing_fermion!(x)
    for ν=1:4
        
        xplus = shift_fermion(x,ν)
        #println(xplus)
        

        mul!(temp1,U[ν],xplus,isodd)
       

        #fermion_shift!(temp1,U,ν,x)

        #... Dirac multiplication

        mul!(temp1,view(A.rminusγ,:,:,ν),isodd)

        

        xminus = shift_fermion(x,-ν)
        Uminus = shift_U(U[ν],-ν)


        mul!(temp2,Uminus',xminus,isodd)
     
        #
        #fermion_shift!(temp2,U,-ν,x)
        #mul!(temp2,view(x.rplusγ,:,:,ν),temp2)
        mul!(temp2,view(A.rplusγ,:,:,ν),isodd)

        add_fermion!(xout,A.hopp[ν],temp1,A.hopm[ν],temp2,isodd)

    end

    #clear_fermion!(xout,isodd)
    #add_fermion!(xout,1,x,-1,temp)

    set_wing_fermion!(xout,A.boundarycondition)

end

function add_fermion!(c::WilsonFermion_4D_wing{NC,NDW},α::Number,a::T1,β::Number,b::T2,iseven) where {NC,NDW,T1 <: Abstractfermion,T2 <: Abstractfermion}#c += alpha*a + beta*b
    n1,n2,n3,n4,n5,n6 = size(c.f)

    @inbounds for i6=1:n6
        for i5=1:n5
            it = i5 -NDW
            for i4=1:n4
                iz = i4 -NDW
                for i3=1:n3
                    iy = i3 - NDW
                    for i2=1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0,true,false)
                        if evenodd == iseven
                            @simd for i1=1:NC
                                #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                                c.f[i1,i2,i3,i4,i5,i6] += α*a.f[i1,i2,i3,i4,i5,i6] + β*b.f[i1,i2,i3,i4,i5,i6] 
                            end
                        end
                    end
                end
            end
        end
    end
    return
end

function add_fermion!(c::WilsonFermion_4D_wing{NC,NDW},α::Number,a::T1,iseven::Bool) where {NC,NDW,T1 <: Abstractfermion,T2 <: Abstractfermion}#c += alpha*a + beta*b
    n1,n2,n3,n4,n5,n6 = size(c.f)

    @inbounds for i6=1:n6
        for i5=1:n5
            it = i5 -NDW
            for i4=1:n4
                iz = i4 -NDW
                for i3=1:n3
                    iy = i3 - NDW
                    for i2=1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0,true,false)
                        if evenodd == iseven
                            @simd for i1=1:NC
                                #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                                c.f[i1,i2,i3,i4,i5,i6] += α*a.f[i1,i2,i3,i4,i5,i6] 
                            end
                        end
                    end
                end
            end
        end
    end
    return
end

function WWx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_4D_wing,G <: AbstractGaugefields} #(1 - K^2 Teo Toe) xe
    iseven = true
    temp = A._temporary_fermi[4]#temps[4]
    temp2 = A._temporary_fermi[5]#temps[4]
    Toex!(temp2,U,x,A,iseven) 

    iseven = false
    Toex!(temp,U,temp2,A,iseven) 
    add_fermion!(xout,1,x,-1,temp)

    iseven = true
    set_wing_fermion!(xout,A.boundarycondition)

    return
end

function clear_fermion!(a::WilsonFermion_4D_wing{NC,NDW} ,iseven) where {NC,NDW} 
    n1,n2,n3,n4,n5,n6 = size(a.f)
    @inbounds for i6=1:n6
        for i5=1:n5
            it = i5-NDW
            for i4=1:n4
                iz = i4-NDW
                for i3=1:n3
                    iy = i3 - NDW
                    for i2=1:n2
                        ix = i2 - NDW
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            @simd for i1=1:NC
                                a.f[i1,i2,i3,i4,i5,i6]= 0
                            end
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(x::WilsonFermion_4D_wing{NC,NDW},A::TA) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            x[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            x[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            x[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            x[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(x::WilsonFermion_4D_wing{NC,NDW},A::TA,iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            x[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            x[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            x[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            x[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                        end
                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC,NDW},A::TA,x::WilsonFermion_4D_wing{NC}) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            xout[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            xout[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            xout[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC,NDW},A::TA,x::WilsonFermion_4D_wing{NC},iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            xout[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            xout[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            xout[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                        end

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC,NDW},x::WilsonFermion_4D_wing{NC},A::TA) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            xout[ic,ix,iy,iz,it,2] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            xout[ic,ix,iy,iz,it,3] = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            xout[ic,ix,iy,iz,it,4] = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC,NDW},x::WilsonFermion_4D_wing{NC},A::TA,iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    for ic=1:NC
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for ix=1:NX
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            xout[ic,ix,iy,iz,it,2] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            xout[ic,ix,iy,iz,it,3] = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            xout[ic,ix,iy,iz,it,4] = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                        end
                    end
                end
            end
        end
    end
    
end

#=
function set_wing_fermion!(a::WilsonFermion_4D_wing{NC},boundarycondition) where NC 
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha=1:4
        for it=1:NT
            for iz = 1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,0,iy,iz,it,ialpha] = boundarycondition[1]*a[k,NX,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for iy=1:NY
                    @simd for k=1:NC
                        a[k,NX+1,iy,iz,it,ialpha] =boundarycondition[1]*a[k,1,iy,iz,it,ialpha]
                    end
                end
            end
        end
    end

    #Y-direction
    for ialpha = 1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,0,iz,it,ialpha] =boundarycondition[2]*a[k,ix,NY,iz,it,ialpha]
                    end
                end
            end
        end
    end

    for ialpha=1:4
        for it=1:NT
            for iz=1:NZ
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,NY+1,iz,it,ialpha] = boundarycondition[2]*a[k,ix,1,iz,it,ialpha]
                    end
                end
            end
        end
    end

    
    for ialpha=1:4
        # Z-direction
        for it=1:NT
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,0,it,ialpha] = boundarycondition[3]*a[k,ix,iy,NZ,it,ialpha]
                        a[k,ix,iy,NZ+1,it,ialpha] = boundarycondition[3]*a[k,ix,iy,1,it,ialpha]

                    end
                end
            end
        end

        #T-direction
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    @simd for k=1:NC
                        a[k,ix,iy,iz,0,ialpha] = boundarycondition[4]*a[k,ix,iy,iz,NT,ialpha]
                        a[k,ix,iy,iz,NT+1,ialpha] =boundarycondition[4]*a[k,ix,iy,iz,1,ialpha]
                    end
                end
            end
        end

    end

end

=#

"""
c--------------------------------------------------------------------------c
c     y = gamma_5 * x
c     here
c                  ( -1       )
c        GAMMA5 =  (   -1     )
c                  (     +1   )
c                  (       +1 )
c--------------------------------------------------------------------------c
    """
    function mul_γ5x!(y::WilsonFermion_4D_wing{NC},x::WilsonFermion_4D_wing{NC}) where NC
        NX = x.NX
        NY = x.NY
        NZ = x.NZ
        NT = x.NT
        for ig=1:4
            for ic=1:NC
                for it=1:NT
                    for iz=1:NZ
                        for iy=1:NY
                            for ix=1:NX
                                @simd for ic=1:NC
                                    y[ic,ix,iy,iz,it,ig] =x[ic,ix,iy,iz,it,ig]*ifelse(ig <= 2,-1,1)
                                end
                            end
                        end
                    end
                end
            end
        end
    end