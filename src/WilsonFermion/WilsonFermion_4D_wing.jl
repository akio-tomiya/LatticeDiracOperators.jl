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
