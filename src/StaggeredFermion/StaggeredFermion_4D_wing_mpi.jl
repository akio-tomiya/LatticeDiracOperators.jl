import Gaugefields:staggered_U

struct StaggeredFermion_4D_mpi{NC} <: AbstractFermionfields_4D{NC}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    NG::Int64 #size of the Gamma matrix. In Staggered fermion, this is one. 
    NV::Int64
    f::Array{ComplexF64,6}
    Dirac_operator::String
    PEs::NTuple{4,Int64}
    PN::NTuple{4,Int64}
    mpiinit::Bool
    myrank::Int64
    nprocs::Int64
    myrank_xyzt::NTuple{4,Int64}
    mpi::Bool
    

    function StaggeredFermion_4D_mpi(NC,NX,NY,NZ,NT,PEs) 
        NG = 1
        NDW = 1
        NV = NC*NX*NY*NZ*NT*NG
        @assert NX % PEs[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs"
        @assert NY % PEs[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs"
        @assert NZ % PEs[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs"
        @assert NT % PEs[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs"

        
        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64,NC,NX+2NDW,NY+2NDW,NZ+2NDW,NT+2NDW,NG)
        Dirac_operator = "Staggered"
        return new{NC}(NC,NX,NY,NZ,NT,NDW,NG,NV,f,Dirac_operator)
    end
end

function Base.size(x::StaggeredFermion_4D_mpi{NC}) where NC
    return (x.NC,x.NX,x.NY,x.NZ,x.NT,x.NG)
    #return (x.NV,)
end

function Base.length(x::StaggeredFermion_4D_mpi{NC}) where {NC}
    return NC*x.NX*x.NY*x.NZ*x.NT*x.NG
end

function Base.similar(x::T) where T <: StaggeredFermion_4D_mpi
    return StaggeredFermion_4D_mpi(x.NC,x.NX,x.NY,x.NZ,x.NT)
end


function Dx!(xout::T,U::Array{G,1},
    x::T,temps::Array{T,1},boundarycondition) where  {T <: StaggeredFermion_4D_mpi,G <:AbstractGaugefields}
    #temp = temps[4]
    temp1 = temps[1]
    temp2 = temps[2]

    #clear!(temp)
    set_wing_fermion!(x,boundarycondition)
    clear_fermion!(xout)
    for ν=1:4
        xplus = shift_fermion(x,ν)
        Us = staggered_U(U[ν],ν)
        mul!(temp1,Us,xplus)


        xminus = shift_fermion(x,-ν)
        Uminus = shift_U(U[ν],-ν)
        Uminus_s = staggered_U(Uminus,ν)
        mul!(temp2,Uminus_s',xminus)
        
        add_fermion!(xout,0.5,temp1,-0.5,temp2)

        #fermion_shift!(temp1,U,ν,x)
        #fermion_shift!(temp2,U,-ν,x)
        #add!(xout,0.5,temp1,-0.5,temp2)
        
    end

    
    set_wing_fermion!(xout,boundarycondition)

    return
end

function clear_fermion!(x::StaggeredFermion_4D_mpi{NC},evensite) where NC
    ibush = ifelse(evensite,0,1)
    for it=1:x.NT
        for iz=1:x.NZ
            for iy=1:x.NY
                xran =1+(1+ibush+iy+iz+it)%2:2:x.NX
                for ix in xran
                    @simd for ic=1:NC
                        x[ic,ix,iy,iz,it,1] = 0
                    end
                end
            end
        end
    end
    return
end


function set_wing_fermion!(a::StaggeredFermion_4D_mpi{NC},boundarycondition) where NC 
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX

    #!  X-direction
    for ialpha=1:1
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

    for ialpha=1:1
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
    for ialpha = 1:1
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

    for ialpha=1:1
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

    
    for ialpha=1:1
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
                        a[k,ix,iy,iz,NT+1,ialpha] = boundarycondition[4]*a[k,ix,iy,iz,1,ialpha]
                    end
                end
            end
        end

    end

end
