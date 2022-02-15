import Base

import Gaugefields:comm,setvalue!
import Gaugefields.AbstractGaugefields_module:getvalue

"""
Struct for WilsonFermion
"""
struct WilsonFermion_4D_mpi{NC,NDW} <: WilsonFermion_4D{NC} #AbstractFermionfields_4D{NC}
    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    PEs::NTuple{4,Int64}
    PN::NTuple{4,Int64}
    mpiinit::Bool
    myrank::Int64
    nprocs::Int64
    myrank_xyzt::NTuple{4,Int64}
    mpi::Bool
    #BoundaryCondition::Vector{Int8}


    function WilsonFermion_4D_mpi(NC::T,NX::T,NY::T,NZ::T,NT::T,PEs) where T<: Integer
        NG = 4
        NDW = 1
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




        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(ComplexF64,NC,PN[1]+2NDW,PN[2]+2NDW,PN[3]+2NDW,PN[4]+2NDW,NG)
        Dirac_operator = "Wilson"
        mpi = true
        mpiinit = true
        return new{NC,NDW}(f,NC,NX,NY,NZ,NT,NG,NDW,Dirac_operator,Tuple(PEs),PN,mpiinit,myrank,nprocs,myrank_xyzt,mpi)
    end


end

function Base.similar(x::T) where T <:  WilsonFermion_4D_mpi
    return WilsonFermion_4D_mpi(x.NC,x.NX,x.NY,x.NZ,x.NT,x.PEs)
end


function get_myrank_xyzt(myrank,PEs)
    #myrank = (((myrank_t)*PEs[3]+myrank_z)*PEs[2] + myrank_y)*PEs[1] + myrank_x
    myrank_x = myrank % PEs[1] 
    i = (myrank - myrank_x) ÷ PEs[1]
    myrank_y = i % PEs[2]
    i = (i - myrank_y) ÷ PEs[2]
    myrank_z = i % PEs[3]
    myrank_t = (i - myrank_z) ÷ PEs[3]

    return myrank_x,myrank_y,myrank_z,myrank_t
end

function get_myrank(myrank_xyzt,PEs)
    @inbounds return (((myrank_xyzt[4])*PEs[3]+myrank_xyzt[3])*PEs[2] + myrank_xyzt[2])*PEs[1] + myrank_xyzt[1]
end

function Base.setindex!(x::WilsonFermion_4D_mpi{NC,NDW},v,i1,i2,i3,i4,i5,i6)  where {NC,NDW}
    error("Each element can not be accessed by global index in $(typeof(x)). Use setvalue! function")
    #@inbounds x.f[i1,i2 + NDW,i3 + NDW,i4 + NDW,i5 + NDW,i6] = v
end

function Base.getindex(x::WilsonFermion_4D_mpi{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
    error("Each element can not be accessed by global index in $(typeof(x)) Use getvalue function")
    #@inbounds return x.f[i1,i2 .+ NDW,i3 .+ NDW,i4 .+ NDW,i5 .+ NDW,i6]
end


@inline function getvalue(x::WilsonFermion_4D_mpi{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
    return x.f[i1,i2 .+ x.NDW,i3 .+ x.NDW,i4 .+ x.NDW,i5 .+ x.NDW,i6]
end

@inline  function setvalue!(x::WilsonFermion_4D_mpi{NC,NDW},v,i1,i2,i3,i4,i5,i6) where {NC,NDW}
    x.f[i1,i2 .+ x.NDW,i3 .+ x.NDW,i4 .+ x.NDW,i5 .+ x.NDW,i6] = v
end

function setvalue!(F::T,v,i1,i2,i3,i4,i5,i6)  where T <: Shifted_fermionfields_4D
    error("type $(typeof(F)) has no setindex method. This type is read only.")
end

function getvalue(F::T,i1,i2,i3,i4,i5,i6)  where T <: Shifted_fermionfields_4D
    @inbounds return getvalue(F.parent,i1,i2.+ F.shift[1],i3.+ F.shift[2],i4.+ F.shift[3],i5.+ F.shift[4],i6)
end


"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::WilsonFermion_4D_mpi{NC,NDW}) where {NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    σ = sqrt(1/2)

    for ialpha = 1:n6
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                        for ic=1:NC 
                            v = σ*randn()+im*σ*randn()
                            setvalue!(x,v,ic,ix,iy,iz,it,ialpha)
                            #x[ic,ix,iy,iz,it,ialpha] = σ*randn()+im*σ*randn()
                        end
                    end
                end
            end
        end
    end
    set_wing_fermion!(x)
    return
end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::WilsonFermion_4D_mpi{NC,NDW},randomfunc,σ) where {NC,NDW}
  
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    #σ = sqrt(1/2)

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
                            setvalue!(x,v,ic,ix,iy,iz,it,mu)

                            #x[ic,ix,iy,iz,it,mu] = σ*xr + σ*im*xi
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)

    return
end

function gauss_distribution_fermion!(x::AbstractFermionfields_4D{NC},randomfunc) where NC
    σ = 1
    gauss_distribution_fermion!(x,randomfunc,σ)
end


#=
function Base.getindex(F::Shifted_fermionfields_4D{NC,WilsonFermion_4D_mpi{NC,NDW}},i1::N,i2::N,i3::N,i4::N,i5::N,i6::N)  where {NC,NDW,N <: Integer}
    
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

function Base.getindex(x::WilsonFermion_4D_mpi{NC,NDW},i1,i2,i3,i4,i5,i6) where {NC,NDW}
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


#=
function Base.getindex(x::T,i1,i2,i3,i4,i5,i6) where T <: WilsonFermion_4D_mpi{NC} where NC
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

function set_wing_fermion!(a::WilsonFermion_4D_mpi{NC,NDW},boundarycondition) where {NC,NDW}
    NT = a.NT
    NY = a.NY
    NZ = a.NZ
    NX = a.NX
    PEs = a.PEs
    PN = a.PN
    myrank = a.myrank
    myrank_xyzt = a.myrank_xyzt
    myrank_xyzt_send = a.myrank_xyzt
    NG = 4
    

    #X direction 
    #Now we send data
    #from NX to 1
    N = PN[2]*PN[3]*PN[4]*NDW*NC*NG
    send_mesg1 = Array{ComplexF64}(undef, N)
    recv_mesg1 = Array{ComplexF64}(undef, N)

    count = 0
    for ialpha = 1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for iy=1:PN[2]
                    for id=1:NDW
                        ix_real = myrank_xyzt[1]*PN[1] + PN[1]+(id-NDW)
                        phase =ifelse(ix_real >= NX,boundarycondition[1],1)
                        for k2=1:NC
                            #for k1=1:NC
                                count += 1
                                
                                #a[k,0,iy,iz,it,ialpha] = boundarycondition[1]*a[k,NX,iy,iz,it,ialpha]
                                send_mesg1[count] = phase*getvalue(a,k2,PN[1]+(id-NDW),iy,iz,it,ialpha)
                                #u[k1,k2,-NDW+id,iy,iz,it] = u[k1,k2,NX+(id-NDW),iy,iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    px = myrank_xyzt[1] + 1
    px += ifelse(px >= PEs[1],-PEs[1],0)        
    myrank_xyzt_send = (px,myrank_xyzt[2],myrank_xyzt[3],myrank_xyzt[4])
    myrank_send1 = get_myrank(myrank_xyzt_send,PEs)
    #=
    for ip=0:u.nprocs-1
        if ip == u.myrank
            println("rank = $myrank, myrank_send1 = $(myrank_send1)")
        end
        MPI.Barrier(comm)

    end
    =#
    
    sreq1 = MPI.Isend(send_mesg1, myrank_send1, myrank_send1+32, comm) #from left to right 0 -> 1

    N = PN[2]*PN[3]*PN[4]*NDW*NC*NG
    send_mesg2 = Array{ComplexF64}(undef, N)
    recv_mesg2 = Array{ComplexF64}(undef, N)

    count = 0
    for ialpha =1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for iy=1:PN[2]
                    for id=1:NDW
                        ix_real = myrank_xyzt[1]*PN[1] + id
                        phase =ifelse(ix_real <= NDW,boundarycondition[1],1)
                        for k2=1:NC
                            #for k1=1:NC
                                count += 1
                                send_mesg2[count] = phase*getvalue(a,k2,id,iy,iz,it,ialpha)
                            #end
                        end
                    end
                end
            end
        end
    end
    px = myrank_xyzt[1] - 1
    px += ifelse(px < 0,PEs[1],0)
    #println("px = $px")        
    myrank_xyzt_send = (px,myrank_xyzt[2],myrank_xyzt[3],myrank_xyzt[4])
    myrank_send2 = get_myrank(myrank_xyzt_send,PEs)
    #=
    for ip=0:u.nprocs-1
        if ip == u.myrank
            println("rank = $myrank, myrank_send2 = $(myrank_send2)")
        end
        MPI.Barrier(comm)

    end
    =#


    
    sreq2 = MPI.Isend(send_mesg2, myrank_send2, myrank_send2+64, comm) #from right to left 0 -> -1

    #=
    myrank = 1: myrank_send1 = 2, myrank_send2 = 0
        sreq1: from 1 to 2 2
        sreq2: from 1 to 0 2
    myrank = 2: myrank_send1 = 3, myrank_send2 = 1
        sreq1: from 2 to 3 3
        sreq2: from 2 to 1 1
        rreq1: from 1 to 2 2 -> sreq1 at myrank 1
        rreq2: from 3 to 2 2 
    myrank = 3: myrank_send1 = 4, myrank_send2 = 2
        sreq1: from 3 to 4 4
        sreq2: from 3 to 2 2
    =#

    rreq1 = MPI.Irecv!(recv_mesg1, myrank_send2, myrank+32, comm) #from -1 to 0
    rreq2 = MPI.Irecv!(recv_mesg2, myrank_send1, myrank+64, comm) #from 1 to 0

    stats = MPI.Waitall!([rreq1, sreq1,rreq2,sreq2])
    MPI.Barrier(comm)

    count = 0
    for ialpha = 1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for iy=1:PN[2]
                    for id=1:NDW
                        
                        for k2=1:NC
                            #for k1=1:NC
                                count += 1
                                v = recv_mesg1[count]
                                setvalue!(a,v,k2,-NDW+id,iy,iz,it,ialpha)
                                #send_mesg1[count] = getvalue(u,k1,k2,PN[1]+(id-NDW),iy,iz,it)
                                #u[k1,k2,-NDW+id,iy,iz,it] = u[k1,k2,NX+(id-NDW),iy,iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    count = 0
    for ialpha=1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for iy=1:PN[2]
                    for id=1:NDW
                        for k2=1:NC
                            #for k1=1:NC
                                count += 1
                                v = recv_mesg2[count]
                                setvalue!(a,v,k2,PN[1]+id,iy,iz,it,ialpha)
                                #u[k1,k2,NX+id,iy,iz,it] = u[k1,k2,id,iy,iz,it]
                                #send_mesg2[count] = getvalue(u,k1,k2,id,iy,iz,it)
                            #end
                        end
                    end
                end
            end
        end
    end


    #N = PN[1]*PN[3]*PN[4]*NDW*NC*NC
    N = PN[4]*PN[3]*length(-NDW+1:PN[1]+NDW)*NDW*NC*NG
    send_mesg1 = Array{ComplexF64}(undef, N)
    recv_mesg1 = Array{ComplexF64}(undef, N)
    send_mesg2 = Array{ComplexF64}(undef, N)
    recv_mesg2 = Array{ComplexF64}(undef, N)

    #Y direction 
    #Now we send data
    count = 0
    for ialpha=1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for ix=-NDW+1:PN[1]+NDW
                    for id=1:NDW
                        iy_real = myrank_xyzt[2]*PN[2] + PN[2]+(id-NDW)
                        phase =ifelse(iy_real >= NY,boundarycondition[2],1)
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                send_mesg1[count] = phase*getvalue(a,k1,ix,PN[2]+(id-NDW),iz,it,ialpha)
                                #u[k1,k2,ix,-NDW+id,iz,it] = u[k1,k2,ix,NY+(id-NDW),iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    py = myrank_xyzt[2] + 1
    py += ifelse(py >= PEs[2],-PEs[2],0)        
    myrank_xyzt_send = (myrank_xyzt[1],py,myrank_xyzt[3],myrank_xyzt[4])
    myrank_send1 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send1 = $(myrank_send1)")
    sreq1 = MPI.Isend(send_mesg1, myrank_send1, myrank_send1+32, comm) #from left to right 0 -> 1


    count = 0
    for ialpha=1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for ix=-NDW+1:PN[1]+NDW
                    for id=1:NDW
                        iy_real = myrank_xyzt[2]*PN[2] + id
                        phase =ifelse(iy_real <= NDW,boundarycondition[2],1)
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                send_mesg2[count] = phase*getvalue(a,k1,ix,id,iz,it,ialpha)
                                #u[k1,k2,ix,NY+id,iz,it] = u[k1,k2,ix,id,iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    py = myrank_xyzt[2] - 1
    py += ifelse(py < 0,PEs[2],0)
    #println("py = $py")        
    myrank_xyzt_send = (myrank_xyzt[1],py,myrank_xyzt[3],myrank_xyzt[4])
    myrank_send2 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send2 = $(myrank_send2)")
    sreq2 = MPI.Isend(send_mesg2, myrank_send2, myrank_send2+64, comm) #from right to left 0 -> -1

    rreq1 = MPI.Irecv!(recv_mesg1, myrank_send2, myrank+32, comm) #from -1 to 0
    rreq2 = MPI.Irecv!(recv_mesg2, myrank_send1, myrank+64, comm) #from 1 to 0

    stats = MPI.Waitall!([rreq1, sreq1,rreq2,sreq2])

    count = 0
    for ialpha=1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for ix=-NDW+1:PN[1]+NDW
                    for id=1:NDW

                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                v = recv_mesg1[count] 
                                setvalue!(a,v,k1,ix,-NDW+id,iz,it,ialpha)
                                #send_mesg1[count] = getvalue(u,k1,k2,ix,PN[2]+(id-NDW),iz,it)
                                #u[k1,k2,ix,-NDW+id,iz,it] = u[k1,k2,ix,NY+(id-NDW),iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    count = 0
    for ialpha=1:4
        for it=1:PN[4]
            for iz=1:PN[3]
                for ix=-NDW+1:PN[1]+NDW
                    for id=1:NDW
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                v = recv_mesg2[count]
                                setvalue!(a,v,k1,ix,PN[2]+id,iz,it,ialpha)
                                #send_mesg2[count] = getvalue(u,k1,k2,ix,id,iz,it)
                                #u[k1,k2,ix,NY+id,iz,it] = u[k1,k2,ix,id,iz,it]
                            #end
                        end
                    end
                end
            end
        end
    end


    MPI.Barrier(comm)

    #Z direction 
    #Now we send data

    N = NDW*PN[4]*length(-NDW+1:PN[2]+NDW)*length(-NDW+1:PN[1]+NDW)*NC*NG
    send_mesg1 = Array{ComplexF64}(undef, N)
    recv_mesg1 = Array{ComplexF64}(undef, N)
    send_mesg2 = Array{ComplexF64}(undef, N)
    recv_mesg2 = Array{ComplexF64}(undef, N)

    count = 0
    for ialpha = 1:4
        for id=1:NDW
            iz_real1 = myrank_xyzt[3]*PN[3] + PN[3]+(id-NDW)
            phase1 =ifelse(iz_real1 >= NZ,boundarycondition[3],1)
            iz_real2 = myrank_xyzt[3]*PN[3] + id
            phase2 =ifelse(iz_real2 <= NDW,boundarycondition[3],1)
            for it=1:PN[4]
                for iy=-NDW+1:PN[2]+NDW
                    for ix=-NDW+1:PN[1]+NDW
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                send_mesg1[count] = phase1*getvalue(a,k1,ix,iy,PN[3]+(id-NDW),it,ialpha)
                                send_mesg2[count] = phase2*getvalue(a,k1,ix,iy,id,it,ialpha)
                                #u[k1,k2,ix,iy,id-NDW,it] = u[k1,k2,ix,iy,NZ+(id-NDW),it]
                                #u[k1,k2,ix,iy,NZ+id,it] = u[k1,k2,ix,iy,id,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    pz = myrank_xyzt[3] + 1
    pz += ifelse(pz >= PEs[3],-PEs[3],0)        
    myrank_xyzt_send = (myrank_xyzt[1],myrank_xyzt[2],pz,myrank_xyzt[4])
    myrank_send1 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send1 = $(myrank_send1)")
    sreq1 = MPI.Isend(send_mesg1, myrank_send1, myrank_send1+32, comm) #from left to right 0 -> 1

    pz = myrank_xyzt[3] - 1
    pz += ifelse(pz < 0,PEs[3],0)
    #println("pz = $pz")        
    myrank_xyzt_send = (myrank_xyzt[1],myrank_xyzt[2],pz,myrank_xyzt[4])
    myrank_send2 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send2 = $(myrank_send2)")
    sreq2 = MPI.Isend(send_mesg2, myrank_send2, myrank_send2+64, comm) #from right to left 0 -> -1

    rreq1 = MPI.Irecv!(recv_mesg1, myrank_send2, myrank+32, comm) #from -1 to 0
    rreq2 = MPI.Irecv!(recv_mesg2, myrank_send1, myrank+64, comm) #from 1 to 0

    stats = MPI.Waitall!([rreq1, sreq1,rreq2,sreq2])

    count = 0
    for ialpha=1:4
        for id=1:NDW
            for it=1:PN[4]
                for iy=-NDW+1:PN[2]+NDW
                    for ix=-NDW+1:PN[1]+NDW
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                v = recv_mesg1[count]
                                setvalue!(a,v,k1,ix,iy,id-NDW,it,ialpha)
                                v = recv_mesg2[count]
                                setvalue!(a,v,k1,ix,iy,PN[3]+id,it,ialpha)
                                #u[k1,k2,ix,iy,id-NDW,it] = u[k1,k2,ix,iy,NZ+(id-NDW),it]
                                #u[k1,k2,ix,iy,NZ+id,it] = u[k1,k2,ix,iy,id,it]
                            #end
                        end
                    end
                end
            end
        end
    end

    MPI.Barrier(comm)
    
    #T direction 
    #Now we send data

    N = NDW*length(-NDW+1:PN[3]+NDW)*length(-NDW+1:PN[2]+NDW)*length(-NDW+1:PN[1]+NDW)*NC*NG
    send_mesg1 = Array{ComplexF64}(undef, N)
    recv_mesg1 = Array{ComplexF64}(undef, N)
    send_mesg2 = Array{ComplexF64}(undef, N)
    recv_mesg2 = Array{ComplexF64}(undef, N)

    count = 0
    for ialpha=1:4
        for id=1:NDW
            it_real1 = myrank_xyzt[4]*PN[4] + PN[4]+(id-NDW)
            phase1 =ifelse(it_real1 >= NT,boundarycondition[4],1)
            it_real2 = myrank_xyzt[4]*PN[4] + id
            phase2 =ifelse(it_real2 <= NDW,boundarycondition[4],1)

            for iz=-NDW+1:PN[3]+NDW
                for iy=-NDW+1:PN[2]+NDW
                    for ix=-NDW+1:PN[1]+NDW
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                send_mesg1[count] = phase1*getvalue(a,k1,ix,iy,iz,PN[4]+(id-NDW),ialpha)
                                send_mesg2[count] = phase2*getvalue(a,k1,ix,iy,iz,id,ialpha)
                                #u[k1,k2,ix,iy,iz,id-NDW] = u[k1,k2,ix,iy,iz,PN[4]+(id-NDW)]
                                #u[k1,k2,ix,iy,iz,PN[4]+id] = u[k1,k2,ix,iy,iz,id]
                            #end
                        end
                    end
                end
            end
        end
    end

    pt = myrank_xyzt[4] + 1
    pt += ifelse(pt >= PEs[4],-PEs[4],0)        
    myrank_xyzt_send = (myrank_xyzt[1],myrank_xyzt[2],myrank_xyzt[3],pt)
    myrank_send1 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send1 = $(myrank_send1)")
    sreq1 = MPI.Isend(send_mesg1, myrank_send1, myrank_send1+32, comm) #from left to right 0 -> 1

    pt = myrank_xyzt[4] - 1
    pt += ifelse(pt < 0,PEs[4],0)
    #println("pt = $pt")        
    myrank_xyzt_send = (myrank_xyzt[1],myrank_xyzt[2],myrank_xyzt[3],pt)
    myrank_send2 = get_myrank(myrank_xyzt_send,PEs)
    #println("rank = $rank, myrank_send2 = $(myrank_send2)")
    sreq2 = MPI.Isend(send_mesg2, myrank_send2, myrank_send2+64, comm) #from right to left 0 -> -1

    rreq1 = MPI.Irecv!(recv_mesg1, myrank_send2, myrank+32, comm) #from -1 to 0
    rreq2 = MPI.Irecv!(recv_mesg2, myrank_send1, myrank+64, comm) #from 1 to 0

    stats = MPI.Waitall!([rreq1, sreq1,rreq2,sreq2])

    count = 0
    for ialpha=1:4
        for id=1:NDW
            for iz=-NDW+1:PN[3]+NDW
                for iy=-NDW+1:PN[2]+NDW
                    for ix=-NDW+1:PN[1]+NDW
                        for k1=1:NC
                            #for k2=1:NC
                                count += 1
                                v = recv_mesg1[count]
                                setvalue!(a,v,k1,ix,iy,iz,id-NDW,ialpha)
                                v = recv_mesg2[count]
                                setvalue!(a,v,k1,ix,iy,iz,PN[4]+id,ialpha)

                                #send_mesg1[count] = getvalue(u,k1,k2,ix,iy,iz,PN[4]+(id-NDW))
                                #send_mesg2[count] = getvalue(u,k1,k2,ix,iy,iz,id)
                                #u[k1,k2,ix,iy,iz,id-NDW] = u[k1,k2,ix,iy,iz,PN[4]+(id-NDW)]
                                #u[k1,k2,ix,iy,iz,PN[4]+id] = u[k1,k2,ix,iy,iz,id]
                            #end
                        end
                    end
                end
            end
        end
    end
    #error("rr22r")


    MPI.Barrier(comm)

    return
end






function Wx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_4D_mpi,G <: AbstractGaugefields}
    #temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_mpi,G <: AbstractGaugefields}
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
    x::T,A) where  {T <: WilsonFermion_4D_mpi,G <: AbstractGaugefields}
    #,temps::Array{T,1},boundarycondition) where  {T <: WilsonFermion_4D_mpi,G <: AbstractGaugefields}
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

function add_fermion!(c::WilsonFermion_4D_mpi{NC,NDW},α::Number,a::T1,β::Number,b::T2) where {NC,T1 <: Abstractfermion,T2 <: Abstractfermion,NDW}#c += alpha*a + beta*b
    n1,n2,n3,n4,n5,n6 = size(c.f)

    @inbounds  for ialpha=1:4
        for it=1:c.PN[4]
            for iz=1:c.PN[3]
                for iy=1:c.PN[2]
                    for ix=1:c.PN[1]
                        @simd for k1=1:NC
                            v = getvalue(c,k1,ix,iy,iz,it,ialpha) + α*getvalue(a,k1,ix,iy,iz,it,ialpha)+β*getvalue(b,k1,ix,iy,iz,it,ialpha)
                                setvalue!(c,v,k1,ix,iy,iz,it,ialpha)
                            #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                            #c.f[i1,i2,i3,i4,i5,i6] += α*a.f[i1,i2,i3,i4,i5,i6] + β*b.f[i1,i2,i3,i4,i5,i6] 
                        end
                    end
                end
            end
        end
    end
    return
end


function add_fermion!(c::WilsonFermion_4D_mpi{NC,NDW},α::Number,a::T1) where {NC,T1 <: Abstractfermion,NDW}#c += alpha*a 
    #n1,n2,n3,n4,n5,n6 = size(c.f)

    @inbounds  for ialpha=1:4
        for it=1:c.PN[4]
            for iz=1:c.PN[3]
                for iy=1:c.PN[2]
                    for ix=1:c.PN[1]
                        @simd for k1=1:NC
                            v = getvalue(c,k1,ix,iy,iz,it,ialpha) + α*getvalue(a,k1,ix,iy,iz,it,ialpha)
                            setvalue!(c,v,k1,ix,iy,iz,it,ialpha)
                            #println(a.f[i1,i2,i3,i4,i5,i6],"\t",b.f[i1,i2,i3,i4,i5,i6] )
                            #c.f[i1,i2,i3,i4,i5,i6] += α*a.f[i1,i2,i3,i4,i5,i6] 
                        end
                    end
                end
            end
        end
    end
    return
end



function add_fermion!(c::WilsonFermion_4D_mpi{NC,NDW},α::Number,a::T1,β::Number,b::T2,iseven) where {NC,NDW,T1 <: Abstractfermion,T2 <: Abstractfermion}#c += alpha*a + beta*b

    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds  for ialpha=1:4
        for it=1:c.PN[4]
            for iz=1:c.PN[3]
                for iy=1:c.PN[2]
                    for ix=1:c.PN[1]
                        #for k2=1:NC    
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0,true,false)
                        if evenodd == iseven                           
                            @simd for k1=1:NC
                                v = getvalue(c,k1,ix,iy,iz,it,ialpha) + α*getvalue(a,k1,ix,iy,iz,it,ialpha)+β*getvalue(b,k1,ix,iy,iz,it,ialpha)
                                setvalue!(c,v,k1,ix,iy,iz,it,ialpha)
                                #c[k1,k2,ix,iy,iz,it] += α*a[k1,k2,ix,iy,iz,it]
                            end
                        end
                    end
                end
            end
        end
    end
    #set_wing_fermion!(c,iseven)

    return
end

function add_fermion!(c::WilsonFermion_4D_mpi{NC,NDW},α::Number,a::T1,iseven::Bool) where {NC,NDW,T1 <: Abstractfermion,T2 <: Abstractfermion}#c += alpha*a + beta*b


    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds  for ialpha=1:4
        for it=1:c.PN[4]
            for iz=1:c.PN[3]
                for iy=1:c.PN[2]
                    for ix=1:c.PN[1]
                        #for k2=1:NC     
                        evenodd = ifelse((ix + iy + iz + it) % 2 == 0,true,false)
                        if evenodd == iseven                    
                            @simd for k1=1:NC
                                v = getvalue(c,k1,ix,iy,iz,it,ialpha) + α*getvalue(a,k1,ix,iy,iz,it,ialpha)
                                setvalue!(c,v,k1,ix,iy,iz,it,ialpha)
                                #c[k1,k2,ix,iy,iz,it] += α*a[k1,k2,ix,iy,iz,it]
                            end
                        end
                    end
                end
            end
        end
    end
    #set_wing_fermion!(c,iseven)


    return
end

function WWx!(xout::T,U::Array{G,1},x::T,A)  where  {T <: WilsonFermion_4D_mpi,G <: AbstractGaugefields} #(1 - K^2 Teo Toe) xe
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

function clear_fermion!(a::WilsonFermion_4D_mpi{NC,NDW} ,iseven) where {NC,NDW} 
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


function LinearAlgebra.dot(a::WilsonFermion_4D_mpi{NC,NDW},b::WilsonFermion_4D_mpi{NC,NDW}) where {NC,NDW}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    NG = a.NG

    c = 0.0im
    @inbounds for α=1:NG
        for it=1:a.PN[4]
            for iz=1:a.PN[3]
                for iy=1:a.PN[2]
                    for ix=1:a.PN[1]
                        @simd for ic=1:NC
                            va = getvalue(a,ic,ix,iy,iz,it,α)
                            vb = getvalue(b,ic,ix,iy,iz,it,α)
                            c += conj(va)*vb
                            #c+= conj(a[ic,ix,iy,iz,it,α])*b[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end  

    c = MPI.Allreduce(c,MPI.SUM,comm)
    return c
end

function LinearAlgebra.mul!(y::WilsonFermion_4D_mpi{3,NDW},A::T,x::T3) where {T<:Abstractfields,T3 <:Abstractfermion,NDW}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for ialpha=1:NG
        for it=1:y.PN[4]
            for iz=1:y.PN[3]
                for iy=1:y.PN[2]
                    for ix=1:y.PN[1]
                        #println(ix)
                        x1 = getvalue(x,1,ix,iy,iz,it,ialpha)#  x[ic,ix,iy,iz,it,1]
                        x2 = getvalue(x,2,ix,iy,iz,it,ialpha)
                        x3 = getvalue(x,3,ix,iy,iz,it,ialpha)

                        v= getvalue(A,1,1,ix,iy,iz,it)*x1 + 
                                                    getvalue(A,1,2,ix,iy,iz,it)*x2+ 
                                                    getvalue(A,1,3,ix,iy,iz,it)*x3
                        setvalue!(y,v,1,ix,iy,iz,it,ialpha)
                        v = getvalue(A,2,1,ix,iy,iz,it)*x1+ 
                                                    getvalue(A,2,2,ix,iy,iz,it)*x2 + 
                                                    getvalue(A,2,3,ix,iy,iz,it)*x3
                        setvalue!(y,v,2,ix,iy,iz,it,ialpha)
                        v= getvalue(A,3,1,ix,iy,iz,it)*x1+ 
                                                    getvalue(A,3,2,ix,iy,iz,it)*x2 + 
                                                    getvalue(A,3,3,ix,iy,iz,it)*x3
                        setvalue!(y,v,3,ix,iy,iz,it,ialpha)
                        # =#
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(x::WilsonFermion_4D_mpi{NC,NDW},A::TA) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,1)
                            v = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,2)
                            v = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,3)
                            v = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,4)

                            #x[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            #x[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            #x[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            #x[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(x::WilsonFermion_4D_mpi{NC,NDW},A::TA,iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,1)
                            v = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,2)
                            v = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,3)
                            v = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                            setvalue!(x,v,ic,ix,iy,iz,it,4)

                            #x[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            #x[ic,ix,iy,iz,it,2] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            #x[ic,ix,iy,iz,it,3] = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            #x[ic,ix,iy,iz,it,4] = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                        end

                    end
                end
            end
        end
    end

    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_mpi{NC,NDW},A::TA,x::WilsonFermion_4D_mpi{NC}) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)
    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                    end
                end
            end
        end
    end


    #=
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
    =#
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_mpi{NC,NDW},A::TA,x::WilsonFermion_4D_mpi{NC},iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven

                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                        end

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_mpi{NC,NDW},x::WilsonFermion_4D_mpi{NC},A::TA) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]

                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)

                            #=
                            e1 = x[ic,ix,iy,iz,it,1]
                            e2 = x[ic,ix,iy,iz,it,2]
                            e3 = x[ic,ix,iy,iz,it,3]
                            e4 = x[ic,ix,iy,iz,it,4]

                            xout[ic,ix,iy,iz,it,1] = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            xout[ic,ix,iy,iz,it,2] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            xout[ic,ix,iy,iz,it,3] = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            xout[ic,ix,iy,iz,it,4] = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                            =#

                    end
                end
            end
        end
    end
    
end

function LinearAlgebra.mul!(xout::WilsonFermion_4D_mpi{NC,NDW},x::WilsonFermion_4D_mpi{NC,NDW},A::TA,iseven) where {TA <: AbstractMatrix, NC,NDW}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds  for ic=1:NC
        for it=1:x.PN[4]
            for iz=1:x.PN[3]
                for iy=1:x.PN[2]
                    for ix=1:x.PN[1]
                        evenodd = ifelse((ix+iy+iz+it) % 2 == 0,true,false)
                        if evenodd == iseven
                            e1 = getvalue(x,ic,ix,iy,iz,it,1)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x,ic,ix,iy,iz,it,2)
                            e3 = getvalue(x,ic,ix,iy,iz,it,3)
                            e4 = getvalue(x,ic,ix,iy,iz,it,4)

                            v = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)
                            v = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                            setvalue!(xout,v,ic,ix,iy,iz,it,1)

                        end
                    end
                end
            end
        end
    end
    
end

#=
function set_wing_fermion!(a::WilsonFermion_4D_mpi{NC},boundarycondition) where NC 
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
    function mul_γ5x!(y::WilsonFermion_4D_mpi{NC,NDW},x::WilsonFermion_4D_mpi{NC}) where {NC,NDW}
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