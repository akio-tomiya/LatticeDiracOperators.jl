"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_wing{NC} <: AbstractFermionfields_4D{NC}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    f::Array{ComplexF64,6}
    Dirac_operator::String


    function WilsonFermion_4D_wing(NC::T,NX::T,NY::T,NZ::T,NT::T) where T<: Integer
        NG = 4
        NDW = 1
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW,NG)
        Dirac_operator = "Wilson"
        return new{NC}(NC,NX,NY,NZ,NT,NG,NDW,f,Dirac_operator)
    end


end

function Base.similar(x::T) where T <: WilsonFermion_4D_wing
    return WilsonFermion_4D_wing(x.NC,x.NX,x.NY,x.NZ,x.NT)
end

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

function LinearAlgebra.mul!(x::WilsonFermion_4D_wing{NC},A::TA) where {TA <: AbstractMatrix, NC}
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

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC},A::TA,x::WilsonFermion_4D_wing{NC}) where {TA <: AbstractMatrix, NC}
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

function LinearAlgebra.mul!(xout::WilsonFermion_4D_wing{NC},x::WilsonFermion_4D_wing{NC},A::TA) where {TA <: AbstractMatrix, NC}
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