import Gaugefields: staggered_U
import Gaugefields: comm, setvalue!
import Gaugefields: barrier
import Gaugefields.AbstractGaugefields_module: getvalue
import Gaugefields.AbstractGaugefields_module:
    Staggered_Gaugefields, Shifted_Gaugefields_4D_mpi_nowing, Gaugefields_4D_nowing_mpi

struct StaggeredFermion_4D_nowing_mpi{NC} <: AbstractFermionfields_4D{NC}
    f::Array{ComplexF64,6}
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    PEs::NTuple{4,Int64} #number of processes in each dimension
    PN::NTuple{4,Int64} #number of sites in each process
    mpiinit::Bool
    myrank::Int64
    nprocs::Int64
    myrank_xyzt::NTuple{4,Int64}
    mpi::Bool
    fshifted::Array{ComplexF64,6}
    tempmatrix::Array{ComplexF64,3}
    positions::Vector{Int64}
    send_ranks::Dict{Int64,Data_sent_fermion{NC}}
    win::MPI.Win
    win_i::MPI.Win
    win_1i::MPI.Win
    countvec::Vector{Int64}
    otherranks::Vector{Int64}
    win_other::MPI.Win
    your_ranks::Matrix{Int64}
    comm::MPI.Comm




    function StaggeredFermion_4D_nowing_mpi(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        PEs;
        comm=MPI.COMM_WORLD,
    ) where {T<:Integer}
        NG = 1
        NDW = 0
        NV = NX * NY * NZ * NT
        @assert NX % PEs[1] == 0 "NX % PEs[1] should be 0. Now NX = $NX and PEs = $PEs"
        @assert NY % PEs[2] == 0 "NY % PEs[2] should be 0. Now NY = $NY and PEs = $PEs"
        @assert NZ % PEs[3] == 0 "NZ % PEs[3] should be 0. Now NZ = $NZ and PEs = $PEs"
        @assert NT % PEs[4] == 0 "NT % PEs[4] should be 0. Now NT = $NT and PEs = $PEs"

        PN = (NX ÷ PEs[1], NY ÷ PEs[2], NZ ÷ PEs[3], NT ÷ PEs[4])

        nprocs = MPI.Comm_size(comm)
        @assert prod(PEs) == nprocs "num. of MPI process should be prod(PEs). Now nprocs = $nprocs and PEs = $PEs"
        myrank = MPI.Comm_rank(comm)

        myrank_xyzt = get_myrank_xyzt(myrank, PEs)




        #@assert NDW == 1 "only NDW = 1 is supported. Now NDW = $NDW"
        f = zeros(
            ComplexF64,
            NC,
            NG,
            PN[1] + 2NDW,
            PN[2] + 2NDW,
            PN[3] + 2NDW,
            PN[4] + 2NDW,
        ) #note: ic,ialpha,ix,iy,it,iz
        fshifted = zero(f)


        Dirac_operator = "Wilson"
        mpi = true
        mpiinit = true


        tempmatrix = zeros(ComplexF64, NC, NG, prod(PN))
        positions = zeros(Int64, prod(PN))
        send_ranks = Dict{Int64,Data_sent_fermion{NC}}()
        mpi = true
        win = MPI.Win_create(tempmatrix, comm)
        win_i = MPI.Win_create(positions, comm)
        countvec = zeros(Int64, 1)
        win_1i = MPI.Win_create(countvec, comm)

        otherranks = zeros(Int64, nprocs)
        otherranks .= 0
        win_other = MPI.Win_create(otherranks, comm)
        your_ranks = zeros(Int64, nprocs, nprocs)


        return new{NC}(
            f,
            NC,
            NX,
            NY,
            NZ,
            NT,
            NG,
            NDW,
            Dirac_operator,
            Tuple(PEs),
            PN,
            mpiinit,
            myrank,
            nprocs,
            myrank_xyzt,
            mpi,
            fshifted,
            tempmatrix,
            positions,
            send_ranks,
            win,
            win_i,
            win_1i,
            countvec,
            otherranks,
            win_other,
            your_ranks,
            comm,
        )
    end
end




function get_myrank(x::T) where {T<:StaggeredFermion_4D_nowing_mpi}
    return x.myrank
end



function get_nprocs(x::T) where {T<:StaggeredFermion_4D_nowing_mpi}
    return x.nprocs
end


function barrier(x::T) where {T<:StaggeredFermion_4D_nowing_mpi}
    MPI.Barrier(x.comm)
end


function Base.size(x::StaggeredFermion_4D_nowing_mpi{NC}) where {NC}
    return (x.NC, x.NX, x.NY, x.NZ, x.NT, x.NG)
    #return (x.NV,)
end

function Base.length(x::StaggeredFermion_4D_nowing_mpi{NC}) where {NC}
    return NC * x.NX * x.NY * x.NZ * x.NT * x.NG
end

function Base.similar(x::T) where {T<:StaggeredFermion_4D_nowing_mpi}
    return StaggeredFermion_4D_nowing_mpi(
        x.NC,
        x.NX,
        x.NY,
        x.NZ,
        x.NT,
        x.PEs,
        comm=x.comm,
    )
end

#=
function Base.similar(x::T) where {T<:StaggeredFermion_4D_nowing_mpi}
    return StaggeredFermion_4D_nowing_mpi(
        x.NC,
        x.NX,
        x.NY,
        x.NZ,
        x.NT,
        x.PEs,
        comm=x.comm,
    )
end
=#

function Base.setindex!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    error(
        "Each element can not be accessed by global index in $(typeof(x)). Use setvalue! function",
    )

end

function Base.getindex(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @warn "Each element can not be accessed by global index in $(typeof(x)) Use getvalue function"
    error(
        "Each element can not be accessed by global index in $(typeof(x)) Use getvalue function",
    )

    return getvalue(x, i1, i2, i3, i4, i5, i6)
    #error("Each element can not be accessed by global index in $(typeof(x)) Use getvalue function")

end

function setindex_global!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    v,
    ic,
    ix,
    iy,
    iz,
    it,
    ialpha,
) where {NC}
    i1 = ic
    i2 = ialpha
    i3 = ix
    i4 = iy
    i5 = iz
    i6 = it

    PN = x.PN
    PEs = x.PEs
    ii3 = i3 + ifelse(i3 < 1, x.NX, 0) + ifelse(i3 > x.NX, -x.NX, 0)
    ii4 = i4 + ifelse(i4 < 1, x.NY, 0) + ifelse(i4 > x.NY, -x.NY, 0)
    ii5 = i5 + ifelse(i5 < 1, x.NZ, 0) + ifelse(i5 > x.NZ, -x.NZ, 0)
    ii6 = i6 + ifelse(i6 < 1, x.NT, 0) + ifelse(i6 > x.NT, -x.NT, 0)
    #i = myrank_xyz*PN + i_local 
    myrank_x = ii3 ÷ PN[1]
    myrank_y = ii4 ÷ PN[2]
    myrank_z = ii5 ÷ PN[3]
    myrank_t = ii6 ÷ PN[4]
    myrank = (((myrank_t) * PEs[3] + myrank_z) * PEs[2] + myrank_y) * PEs[1] + myrank_x
    if myrank == x.myrank
        ilocal_3 = ((ii3 - 1) % PN[1]) + 1
        ilocal_4 = ((ii4 - 1) % PN[2]) + 1
        ilocal_5 = ((ii5 - 1) % PN[3]) + 1
        ilocal_6 = ((ii6 - 1) % PN[4]) + 1
        setvalue!(x, v, i1, i2, ilocal_3, ilocal_4, ilocal_5, ilocal_6)
    end
end

@inline function getvalue(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @inbounds return x.f[i1, i2, i3, i4, i5, i6]
end



@inline function getvalue(
    F::Adjoint_fermionfields{T},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:StaggeredFermion_4D_nowing_mpi}  #F'
    @inbounds return conj(getvalue(F.parent, i1, i2, i3, i4, i5, i6))
end


@inline function setvalue!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    v,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {NC}
    @inbounds x.f[i1, i2, i3, i4, i5, i6] = v
end

#=
function shift_fermion(F::StaggeredFermion_4D_nowing_mpi{NC}, ν::T) where {T<:Integer,NC}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return Shifted_fermionfields_4D_nowing_mpi(F, shift)
end
=#

function update_sent_data!(
    send_ranks,
    N,
    ix,
    iy,
    iz,
    it,
    ix_shifted,
    iy_shifted,
    iz_shifted,
    it_shifted,
    PEs,
    myrank_xyzt,
    xP,
    yP,
    zP,
    tP,
    x::StaggeredFermion_4D_nowing_mpi{NC},
    factor,
) where {NC}
    NG = x.NG
    tempmatrix_mini = view(x.tempmatrix, 1:NC, 1:NG, 1)


    px = myrank_xyzt[1] + xP
    while px >= PEs[1]
        px += -PEs[1]
    end
    while px < 0
        px += PEs[1]
    end

    py = myrank_xyzt[2] + yP
    while py >= PEs[2]
        py += -PEs[2]
    end
    while py < 0
        py += PEs[2]
    end

    pz = myrank_xyzt[3] + zP
    while pz >= PEs[3]
        pz += -PEs[3]
    end
    while pz < 0
        pz += PEs[3]
    end

    pt = myrank_xyzt[4] + tP
    while pt >= PEs[4]
        pt += -PEs[4]
    end
    while pt < 0
        pt += PEs[4]
    end


    myrank_xyzt_send = (px, py, pz, pt)

    myrank_send = get_myrank(myrank_xyzt_send, PEs)
    #println("send ",myrank_send)



    for jc = 1:NG
        @simd for ic = 1:NC
            #v = getvalue(U,ic,jc,ix_shifted_back,iy_shifted_back,iz_shifted_back,it_shifted_back)
            #v = getvalue(U,ic,jc,ix_shifted,iy_shifted,iz_shifted,it_shifted)
            v = factor * getvalue(x, ic, jc, ix, iy, iz, it)
            tempmatrix_mini[ic, jc] = v
        end
    end
    #disp = ((((it-1)*x.PN[3] + iz-1)*x.PN[2] + iy-1)*x.PN[1] + ix-1)*NC*NC
    #disp = ((((it_shifted-1)*x.PN[3] + iz_shifted-1)*x.PN[2] + iy_shifted-1)*x.PN[1] + ix_shifted-1)*NC*NC
    #println(myrank_send)
    disp =
        (((it_shifted - 1) * x.PN[3] + iz_shifted - 1) * x.PN[2] + iy_shifted - 1) *
        x.PN[1] + ix_shifted


    if haskey(send_ranks, myrank_send)
    else
        send_ranks[myrank_send] = Data_sent_fermion(N, NC)
    end
    send_ranks[myrank_send].count += 1
    send_ranks[myrank_send].data[:, :, send_ranks[myrank_send].count] .= tempmatrix_mini
    send_ranks[myrank_send].positions[send_ranks[myrank_send].count] = disp

end

function mpi_updates_fermion_1data!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    send_ranks,
) where {NC}
    if length(send_ranks) != 0
        NG = x.NG
        #=
        for rank=0:get_nprocs(U)
            if rank == get_myrank(U)
                println("myrank = ",myrank)
                for (key,value) in send_ranks
                    println(key,"\t",value.count)
                end
            end
            barrier(U)
        end
        =#
        tempmatrix = x.tempmatrix #zeros(ComplexF64,NC,NC,N)
        #tempmatrix = zeros(ComplexF64,NC,NC,N)
        positions = x.positions

        win = x.win
        #@time win = MPI.Win_create(tempmatrix,comm)
        #println(typeof(win))
        #Isend Irecv
        MPI.Win_fence(0, win)

        for (myrank_send, value) in send_ranks
            count = value.count
            MPI.Put(value.data[:, :, 1:count], myrank_send, win)
        end

        MPI.Win_fence(0, win)
        #MPI.free(win)

        win_i = x.win_i#MPI.Win_create(positions,comm)
        MPI.Win_fence(0, win_i)

        for (myrank_send, value) in send_ranks
            count = value.count
            MPI.Put(value.positions[1:count], myrank_send, win_i)
        end

        MPI.Win_fence(0, win_i)
        #MPI.free(win_i)

        countvec = x.countvec#zeros(Int64,1)
        win_c = x.win_1i
        #win_c = MPI.Win_create(countvec,comm)
        MPI.Win_fence(0, win_c)

        for (myrank_send, value) in send_ranks
            count = value.count
            MPI.Put(Int64[count], myrank_send, win_c)
        end

        MPI.Win_fence(0, win_c)
        #MPI.free(win_c)

        count = countvec[1]



        #=
        for rank=0:get_nprocs(U)
            if rank == get_myrank(U)
                println("myrank = ",myrank)
                for position in positions[1:count]
                    println(position)
                end
            end
            barrier(U)
        end
        =#

        for i = 1:count
            position = positions[i]
            for jc = 1:NG
                for ic = 1:NC
                    ii = ((position - 1) * NG + jc - 1) * NC + ic
                    x.fshifted[ii] = tempmatrix[ic, jc, i]
                end
            end
            #println(position)
        end

        #error("in shiftdU")
    end
end

function mpi_updates_fermion_moredata!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    send_ranks,
) where {NC}
    NG = x.NG
    otherranks = x.otherranks
    win_other = x.win_other

    MPI.Win_fence(0, win_other)
    myrank = get_myrank(x)
    nprocs = get_nprocs(x)
    for (myrank_send, value) in send_ranks
        count = value.count
        MPI.Put(Int64[count], myrank_send, myrank, win_other)
    end
    MPI.Win_fence(0, win_other)


    tempmatrix = x.tempmatrix #zeros(ComplexF64,NC,NC,N)
    #tempmatrix = zeros(ComplexF64,NC,NC,N)
    positions = x.positions

    win = x.win
    #@time win = MPI.Win_create(tempmatrix,comm)
    #println(typeof(win))
    #Isend Irecv

    win_i = x.win_i#MPI.Win_create(positions,comm)

    win_c = x.win_1i
    #win_c = MPI.Win_create(countvec,comm)


    countvec = x.countvec#zeros(Int64,1)

    your_ranks = x.your_ranks #zeros(Int64,nprocs,nprocs)
    your_ranks .= -1

    MPI.Win_fence(0, win_other)
    icount = 0
    for (myrank_send, value) in send_ranks
        icount += 1
        MPI.Get(view(your_ranks, 1:nprocs, icount), myrank_send, win_other)
    end
    MPI.Win_fence(0, win_other)




    MPI.Win_fence(0, win)
    MPI.Win_fence(0, win_i)
    MPI.Win_fence(0, win_c)

    icount = 0
    for (myrank_send, value) in send_ranks
        count = value.count
        icount += 1
        disp = 0
        for irank = 1:myrank
            if your_ranks[irank, icount] != -1
                disp += your_ranks[irank, icount]
            end
        end


        MPI.Put(value.positions[1:count], myrank_send, disp, win_i)
        MPI.Put(value.data[:, :, 1:count], myrank_send, disp * NC * NG, win)
    end


    MPI.Win_fence(0, win)
    MPI.Win_fence(0, win_i)
    MPI.Win_fence(0, win_c)

    your_ranks .= -1

    totaldatanum = sum(otherranks)


    for i = 1:totaldatanum
        position = positions[i]
        for jc = 1:NG
            for ic = 1:NC
                ii = ((position - 1) * NG + jc - 1) * NC + ic
                x.fshifted[ii] = tempmatrix[ic, jc, i]
            end
        end
        #println(position)
    end

    otherranks .= 0

end

function mpi_updates_fermion!(x::StaggeredFermion_4D_nowing_mpi{NC}, send_ranks) where {NC}
    if length(send_ranks) != 0

        val = MPI.Allreduce(length(send_ranks), +, x.comm) ÷ get_nprocs(x)

        #=
        for rank=0:get_nprocs(x)
            if rank == get_myrank(x)
                println("length = ",val,"\t")
                println("myrank = ",rank," length = $(length(send_ranks))")
            end
            barrier(x)
        end
        =#



        #if val == 1
        #    mpi_updates_fermion_1data!(x,send_ranks)
        #else
        mpi_updates_fermion_moredata!(x, send_ranks)
        #end

        return
    end
end

function shifted_fermion!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    boundarycondition,
    shift,
) where {NC}
    PEs = x.PEs
    PN = x.PN
    myrank = x.myrank
    myrank_xyzt = x.myrank_xyzt
    myrank_xyzt_send = x.myrank_xyzt
    bc = boundarycondition
    NG = x.NG
    #tempmatrix = zeros(ComplexF64,NC,NC)#view(x.tempmatrix,1:NC,1:NC,1) #zeros(ComplexF64,NC,NC)
    tempmatrix_mini = view(x.tempmatrix, 1:NC, 1:NG, 1)
    lat_size = size(x.fshifted)
    send_ranks = x.send_ranks
    empty!(send_ranks)
    # Dict{Int64,Data_sent}()
    N = prod(x.PN)

    for it = 1:x.PN[4]
        it_shifted = it - shift[4]
        it_global = myrank_xyzt[4] * x.PN[4] + it
        it_shifted_global = it_global - shift[4]
        inside_up = it_shifted_global > x.NT
        inside_down = it_shifted_global < 1
        factor_t = ifelse(inside_up || inside_down, bc[4], 1)
        #if myrank_xyzt[4] == 0
        while it_shifted_global < 1
            it_shifted += x.NT
            it_shifted_global += x.NT
        end
        #it_shifted += ifelse(it_shifted < 1,x.NT,0)
        #end  
        #if myrank_xyzt[4] == PEs[4]-1
        while it_shifted_global > x.NT
            it_shifted += -x.NT
            it_shifted_global += -x.NT
        end
        #it_shifted += ifelse(it_shifted > x.PN[4],-x.NT,0)
        #end
        if it_shifted <= 0
            tP = div(it_shifted, x.PN[4]) - 1
        else
            tP = div(it_shifted - 1, x.PN[4])
        end
        #if tP < 0 
        #    println("it_shifted $it_shifted tP = $tP myrank_xyzt $myrank_xyzt it = $it shift = $shift it_shifted_global $it_shifted_global")
        #end


        #it_shifted += ifelse(it_shifted < 1,x.PN[4],0)
        while it_shifted < 1
            it_shifted += x.PN[4]
        end
        while it_shifted > x.PN[4]
            it_shifted += -x.PN[4]
        end
        #it_shifted += ifelse(it_shifted > x.PN[4],-x.PN[4],0)


        for iz = 1:x.PN[3]
            iz_shifted = iz - shift[3]
            iz_global = myrank_xyzt[3] * x.PN[3] + iz
            iz_shifted_global = iz_global - shift[3]
            inside_up = iz_shifted_global > x.NZ
            inside_down = iz_shifted_global < 1
            factor_z = ifelse(inside_up || inside_down, bc[3], 1)
            #if myrank_xyzt[3] == 0
            while iz_shifted_global < 1
                iz_shifted += x.NZ
                iz_shifted_global += x.NZ
            end
            #iz_shifted += ifelse(iz_shifted < 1,x.NZ,0)
            #end
            #if myrank_xyzt[3] == PEs[3]-1
            while iz_shifted_global > x.NZ
                iz_shifted += -x.NZ
                iz_shifted_global += -x.NZ
            end

            #iz_shifted += ifelse(iz_shifted > x.PN[3],-x.NZ,0)
            #end

            if iz_shifted <= 0
                zP = div(iz_shifted, x.PN[3]) - 1
            else
                zP = div(iz_shifted - 1, x.PN[3])
            end



            while iz_shifted < 1
                iz_shifted += x.PN[3]
            end
            while iz_shifted > x.PN[3]
                iz_shifted += -x.PN[3]
            end
            #iz_shifted += ifelse(iz_shifted < 1,x.PN[3],0)
            #iz_shifted += ifelse(iz_shifted > x.PN[3],-x.PN[3],0)

            for iy = 1:x.PN[2]

                iy_shifted = iy - shift[2]
                iy_global = myrank_xyzt[2] * x.PN[2] + iy
                iy_shifted_global = iy_global - shift[2]
                inside_up = iy_shifted_global > x.NY
                inside_down = iy_shifted_global < 1
                factor_y = ifelse(inside_up || inside_down, bc[2], 1)
                #if myrank_xyzt[2] == 0
                while iy_shifted_global < 1
                    iy_shifted += x.NY
                    iy_shifted_global += x.NY
                end

                #iy_shifted += ifelse(iy_shifted < 1,x.NY,0)
                #end
                #if myrank_xyzt[2] == PEs[2]-1
                while iy_shifted_global > x.NY
                    iy_shifted += -x.NY
                    iy_shifted_global += -x.NY
                end
                #iy_shifted += ifelse(iy_shifted > x.PN[2],-x.NY,0)
                #end

                if iy_shifted <= 0
                    yP = div(iy_shifted, x.PN[2]) - 1
                else
                    yP = div(iy_shifted - 1, x.PN[2])
                end


                while iy_shifted < 1
                    iy_shifted += x.PN[2]
                end
                while iy_shifted > x.PN[2]
                    iy_shifted += -x.PN[2]
                end
                #iy_shifted += ifelse(iy_shifted < 1,x.PN[2],0)
                #iy_shifted += ifelse(iy_shifted > x.PN[2],-x.PN[2],0)

                for ix = 1:x.PN[1]
                    ix_shifted = ix - shift[1]
                    ix_global = myrank_xyzt[1] * x.PN[1] + ix
                    ix_shifted_global = ix_global - shift[1]
                    inside_up = ix_shifted_global > x.NX
                    inside_down = ix_shifted_global < 1
                    factor_x = ifelse(inside_up || inside_down, bc[1], 1)
                    #if myrank_xyzt[1] == 0
                    while ix_shifted_global < 1
                        ix_shifted += x.NX
                        ix_shifted_global += x.NX
                    end
                    #ix_shifted += ifelse(ix_shifted < 1,x.NX,0)
                    #end
                    #if myrank_xyzt[1] == PEs[1]-1
                    while ix_shifted_global > x.NX
                        ix_shifted += -x.NX
                        ix_shifted_global += -x.NX
                    end
                    #ix_shifted += ifelse(ix_shifted > x.PN[1],-x.NX,0)
                    #end


                    if ix_shifted <= 0
                        xP = div(ix_shifted, x.PN[1]) - 1
                    else
                        xP = div(ix_shifted - 1, x.PN[1])
                    end


                    while ix_shifted < 1
                        ix_shifted += x.PN[1]
                    end
                    while ix_shifted > x.PN[1]
                        ix_shifted += -x.PN[1]
                    end
                    #ix_shifted += ifelse(ix_shifted < 1,x.PN[1],0)
                    #ix_shifted += ifelse(ix_shifted > x.PN[1],-x.PN[1],0)
                    #xP = div(ix_shifted-1,x.PN[1])
                    #println((tP,zP,yP,xP),"\t $shift")
                    if tP == 0 && zP == 0 && yP == 0 && xP == 0
                        for jc = 1:NG
                            @simd for ic = 1:NC
                                #v = getvalue(U,ic,jc,ix_shifted,iy_shifted,iz_shifted,it_shifted)
                                #x.Ushifted[ic,jc,ix,iy,iz,it] = v
                                v =
                                    factor_x *
                                    factor_y *
                                    factor_z *
                                    factor_t *
                                    getvalue(x, ic, jc, ix, iy, iz, it)
                                x.fshifted[
                                    ic,
                                    jc,
                                    ix_shifted,
                                    iy_shifted,
                                    iz_shifted,
                                    it_shifted,
                                ] = v

                            end
                        end
                    else
                        update_sent_data!(
                            send_ranks,
                            N,
                            ix,
                            iy,
                            iz,
                            it,
                            ix_shifted,
                            iy_shifted,
                            iz_shifted,
                            it_shifted,
                            PEs,
                            myrank_xyzt,
                            xP,
                            yP,
                            zP,
                            tP,
                            x,
                            factor_x * factor_y * factor_z * factor_t,
                        )

                    end
                end
            end
        end
    end

    barrier(x)


    if length(send_ranks) != 0
        mpi_updates_fermion!(x, send_ranks)
    end


end

"""
c-------------------------------------------------c
c     Random number function for Gaussian  Noise
    with σ^2 = 1/2
c-------------------------------------------------c
    """
function gauss_distribution_fermion!(x::StaggeredFermion_4D_nowing_mpi{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    #n6 = size(x.f)[6]
    σ = sqrt(1 / 2)


    for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ialpha = 1:NG
                        @inbounds @simd for ic = 1:NC
                            v = σ * randn() + im * σ * randn()
                            setvalue!(x, v, ic, ialpha, ix, iy, iz, it)
                            #x[ic,ialpha,ix,iy,iz,it] = σ*randn()+im*σ*randn()
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
function gauss_distribution_fermion!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    randomfunc,
    σ,
) where {NC}

    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    #n6 = size(x.f)[6]
    #σ = sqrt(1/2)



    for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for mu = 1:NG
                        @inbounds @simd for ic = 1:NC

                            v1 = sqrt(-log(randomfunc() + 1e-10))
                            v2 = 2pi * randomfunc()

                            xr = v1 * cos(v2)
                            xi = v1 * sin(v2)

                            v = σ * xr + σ * im * xi

                            setvalue!(x, v, ic, mu, ix, iy, iz, it)

                            #x[ic,ix,iy,iz,it,mu] = σ*xr + σ*im*xi
                        end
                    end
                end
            end
        end
    end
    #error("v")

    set_wing_fermion!(x)

    return
end

function Z2_distribution_fermion!(x::StaggeredFermion_4D_nowing_mpi{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    #n6 = size(x.f)[6]
    #σ = sqrt(1/2)

    for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for mu = 1:NG
                        for ic = 1:NC
                            v = rand([-1, 1])
                            setvalue!(x, v, ic, mu, ix, iy, iz, it)
                            #x[ic,ix,iy,iz,it,mu] = rand([-1,1])
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
c     Random number function Z4  Noise
c     https://arxiv.org/pdf/1611.01193.pdf
c-------------------------------------------------c
    """
function Z4_distribution_fermi!(x::StaggeredFermion_4D_nowing_mpi{NC}) where {NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    n6 = size(x.f)[6]
    θ = 0.0
    N::Int32 = 4
    Ninv = Float64(1 / N)
    for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ialpha = 1:NG
                        @inbounds @simd for ic = 1:NC
                            θ = Float64(rand(0:N-1)) * π * Ninv # r \in [0,π/4,2π/4,3π/4]
                            v = cos(θ) + im * sin(θ)
                            setvalue!(x, v, ic, ialpha, ix, iy, iz, it)
                            #x[ic,ix,iy,iz,it,ialpha] = cos(θ)+im*sin(θ) 
                        end
                    end
                end
            end
        end
    end

    set_wing_fermion!(x)

    return
end

function gauss_distribution_fermion!(x::StaggeredFermion_4D_nowing_mpi{NC}, randomfunc) where {NC}
    σ = 1
    gauss_distribution_fermion!(x, randomfunc, σ)
end



#=
function set_wing_fermion!(
    a::StaggeredFermion_4D_nowing_mpi{NC},
    boundarycondition,
) where {NC}
    return
end
=#



function Dx!(
    xout::T,
    U::Array{G,1},
    x::T,
    temps::Array{T,1},
    boundarycondition,
) where {T<:StaggeredFermion_4D_nowing_mpi,G<:AbstractGaugefields}
    #temp = temps[4]
    temp1 = temps[1]
    temp2 = temps[2]

    #clear!(temp)
    set_wing_fermion!(x, boundarycondition)
    clear_fermion!(xout)
    for ν = 1:4
        xplus = shift_fermion(x, ν)
        Us = staggered_U(U[ν], ν)
        mul!(temp1, Us, xplus)


        xminus = shift_fermion(x, -ν)
        Uminus = shift_U(U[ν], -ν)
        Uminus_s = staggered_U(Uminus, ν)
        mul!(temp2, Uminus_s', xminus)

        add_fermion!(xout, 0.5, temp1, -0.5, temp2)

        #fermion_shift!(temp1,U,ν,x)
        #fermion_shift!(temp2,U,-ν,x)
        #add!(xout,0.5,temp1,-0.5,temp2)

    end


    set_wing_fermion!(xout, boundarycondition)

    return
end

function clear_fermion!(x::StaggeredFermion_4D_nowing_mpi{NC}, evensite) where {NC}
    ibush = ifelse(evensite, 0, 1)
    for it = 1:x.NT
        for iz = 1:x.NZ
            for iy = 1:x.NY
                xran = 1+(1+ibush+iy+iz+it)%2:2:x.NX
                for ix in xran
                    @simd for ic = 1:NC
                        x[ic, ix, iy, iz, it, 1] = 0
                    end
                end
            end
        end
    end
    return
end

function add_fermion!(
    c::StaggeredFermion_4D_nowing_mpi{NC},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
) where {NC,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b
    #n1,n2,n3,n4,n5,n6 = size(c.f)


    @inbounds for it = 1:c.PN[4]
        for iz = 1:c.PN[3]
            for iy = 1:c.PN[2]
                for ix = 1:c.PN[1]
                    for ialpha = 1:4
                        @simd for k1 = 1:NC
                            v =
                                getvalue(c, k1, ialpha, ix, iy, iz, it) +
                                α * getvalue(a, k1, ialpha, ix, iy, iz, it) +
                                β * getvalue(b, k1, ialpha, ix, iy, iz, it)
                            setvalue!(c, v, k1, ialpha, ix, iy, iz, it)
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


function add_fermion!(
    c::StaggeredFermion_4D_nowing_mpi{NC},
    α::Number,
    a::T1,
) where {NC,T1<:Abstractfermion}#c += alpha*a 
    #n1,n2,n3,n4,n5,n6 = size(c.f)

    @inbounds for it = 1:c.PN[4]
        for iz = 1:c.PN[3]
            for iy = 1:c.PN[2]
                for ix = 1:c.PN[1]
                    for ialpha = 1:4
                        @simd for k1 = 1:NC
                            v =
                                getvalue(c, k1, ialpha, ix, iy, iz, it) +
                                α * getvalue(a, k1, ialpha, ix, iy, iz, it)
                            setvalue!(c, v, k1, ialpha, ix, iy, iz, it)
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



function add_fermion!(
    c::StaggeredFermion_4D_nowing_mpi{NC},
    α::Number,
    a::T1,
    β::Number,
    b::T2,
    iseven,
) where {NC,T1<:Abstractfermion,T2<:Abstractfermion}#c += alpha*a + beta*b

    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:c.PN[4]
        for iz = 1:c.PN[3]
            for iy = 1:c.PN[2]
                for ix = 1:c.PN[1]
                    #for k2=1:NC    
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for ialpha = 1:4
                            @simd for k1 = 1:NC
                                v =
                                    getvalue(c, k1, ialpha, ix, iy, iz, it) +
                                    α * getvalue(a, k1, ialpha, ix, iy, iz, it) +
                                    β * getvalue(b, k1, ialpha, ix, iy, iz, it)
                                setvalue!(c, v, k1, ialpha, ix, iy, iz, it)
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

function add_fermion!(
    c::StaggeredFermion_4D_nowing_mpi{NC},
    α::Number,
    a::T1,
    iseven::Bool,
) where {NC,T1<:Abstractfermion}#c += alpha*a + beta*b


    NT = c.NT
    NZ = c.NZ
    NY = c.NY
    NX = c.NX
    @inbounds for it = 1:c.PN[4]
        for iz = 1:c.PN[3]
            for iy = 1:c.PN[2]
                for ix = 1:c.PN[1]
                    #for k2=1:NC     
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for ialpha = 1:4
                            @simd for k1 = 1:NC
                                v =
                                    getvalue(c, k1, ialpha, ix, iy, iz, it) +
                                    α * getvalue(a, k1, ialpha, ix, iy, iz, it)
                                setvalue!(c, v, k1, ialpha, ix, iy, iz, it)
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

function shift_fermion(F::StaggeredFermion_4D_nowing_mpi{NC}, ν::T) where {T<:Integer,NC}
    if ν == 1
        shift = (1, 0, 0, 0)
    elseif ν == 2
        shift = (0, 1, 0, 0)
    elseif ν == 3
        shift = (0, 0, 1, 0)
    elseif ν == 4
        shift = (0, 0, 0, 1)
    elseif ν == -1
        shift = (-1, 0, 0, 0)
    elseif ν == -2
        shift = (0, -1, 0, 0)
    elseif ν == -3
        shift = (0, 0, -1, 0)
    elseif ν == -4
        shift = (0, 0, 0, -1)
    end

    return Shifted_fermionfields_4D_nowing_mpi(F, shift)
end


function shift_fermion(
    F::TF,
    shift::NTuple{Dim,T},
) where {Dim,T<:Integer,TF<:StaggeredFermion_4D_nowing_mpi}
    return Shifted_fermionfields_4D_nowing_mpi(F, shift)
end



function set_wing_fermion!(
    a::StaggeredFermion_4D_nowing_mpi{NC},
    boundarycondition,
) where {NC}
    return
end

#=
function clear_fermion!(a::StaggeredFermion_4D_nowing_mpi{NC}, iseven) where {NC}
    n1, n6, n2, n3, n4, n5 = size(a.f)
    @inbounds for i5 = 1:n5
        it = i5
        for i4 = 1:n4
            iz = i4
            for i3 = 1:n3
                iy = i3
                for i2 = 1:n2
                    ix = i2
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for i6 = 1:n6
                            @simd for i1 = 1:NC
                                a.f[i1, i6, i2, i3, i4, i5] = 0
                            end
                        end
                    end
                end
            end
        end
    end
end

=#

function LinearAlgebra.dot(
    a::StaggeredFermion_4D_nowing_mpi{NC},
    b::StaggeredFermion_4D_nowing_mpi{NC},
) where {NC}
    NT = a.NT
    NZ = a.NZ
    NY = a.NY
    NX = a.NX
    NG = a.NG

    c = 0.0im
    @inbounds for it = 1:a.PN[4]
        for iz = 1:a.PN[3]
            for iy = 1:a.PN[2]
                for ix = 1:a.PN[1]
                    for α = 1:NG
                        @simd for ic = 1:NC
                            va = getvalue(a, ic, α, ix, iy, iz, it)
                            vb = getvalue(b, ic, α, ix, iy, iz, it)
                            c += conj(va) * vb
                            #c+= conj(a[ic,ix,iy,iz,it,α])*b[ic,ix,iy,iz,it,α]
                        end
                    end
                end
            end
        end
    end

    c = MPI.Allreduce(c, MPI.SUM, a.comm)
    return c
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_nowing_mpi{3},
    A::T,
    x::T3,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for it = 1:y.PN[4]
        for iz = 1:y.PN[3]
            for iy = 1:y.PN[2]
                for ix = 1:y.PN[1]
                    for ialpha = 1:NG
                        #println(ix)
                        x1 = getvalue(x, 1, ialpha, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                        x2 = getvalue(x, 2, ialpha, ix, iy, iz, it)
                        x3 = getvalue(x, 3, ialpha, ix, iy, iz, it)

                        v =
                            getvalue(A, 1, 1, ix, iy, iz, it) * x1 +
                            getvalue(A, 1, 2, ix, iy, iz, it) * x2 +
                            getvalue(A, 1, 3, ix, iy, iz, it) * x3
                        setvalue!(y, v, 1, ialpha, ix, iy, iz, it)
                        v =
                            getvalue(A, 2, 1, ix, iy, iz, it) * x1 +
                            getvalue(A, 2, 2, ix, iy, iz, it) * x2 +
                            getvalue(A, 2, 3, ix, iy, iz, it) * x3
                        setvalue!(y, v, 2, ialpha, ix, iy, iz, it)
                        v =
                            getvalue(A, 3, 1, ix, iy, iz, it) * x1 +
                            getvalue(A, 3, 2, ix, iy, iz, it) * x2 +
                            getvalue(A, 3, 3, ix, iy, iz, it) * x3
                        setvalue!(y, v, 3, ialpha, ix, iy, iz, it)
                        # =#
                    end
                end
            end
        end
    end
end



function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_nowing_mpi{2},
    A::T,
    x::T3,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for it = 1:y.PN[4]
        for iz = 1:y.PN[3]
            for iy = 1:y.PN[2]
                for ix = 1:y.PN[1]
                    for ialpha = 1:NG
                        #println(ix)
                        x1 = getvalue(x, 1, ialpha, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                        x2 = getvalue(x, 2, ialpha, ix, iy, iz, it)
                        #x3 = getvalue(x,3,ialpha,ix,iy,iz,it)

                        v =
                            getvalue(A, 1, 1, ix, iy, iz, it) * x1 +
                            getvalue(A, 1, 2, ix, iy, iz, it) * x2#+ 
                        #getvalue(A,1,3,ix,iy,iz,it)*x3
                        setvalue!(y, v, 1, ialpha, ix, iy, iz, it)
                        v =
                            getvalue(A, 2, 1, ix, iy, iz, it) * x1 +
                            getvalue(A, 2, 2, ix, iy, iz, it) * x2 #+ 
                        #getvalue(A,2,3,ix,iy,iz,it)*x3
                        setvalue!(y, v, 2, ialpha, ix, iy, iz, it)
                        # =#
                    end
                end
            end
        end
    end
end



function LinearAlgebra.mul!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ic = 1:NC
                        e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                        e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                        e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                        e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                        v = A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                        setvalue!(x, v, ic, 1, ix, iy, iz, it)
                        v = A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                        setvalue!(x, v, ic, 2, ix, iy, iz, it)
                        v = A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                        setvalue!(x, v, ic, 3, ix, iy, iz, it)
                        v = A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                        setvalue!(x, v, ic, 4, ix, iy, iz, it)

                        #x[ic,1,ix,iy,iz,it) = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                        #x[ic,2,ix,iy,iz,it] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                        #x[ic,3,ix,iy,iz,it) = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                        #x[ic,4,ix,iy,iz,it) = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    u::T1,
    x::Abstractfermion,
    y::Adjoint_fermionfields{<:StaggeredFermion_4D_nowing_mpi{NC}},
) where {T1<:AbstractGaugefields,NC}
    #_,NX,NY,NZ,NT,NG = size(y)
    NG = x.NG
    clear_U!(u)


    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ib = 1:NC
                        for ik = 1:NG
                            @simd for ia = 1:NC
                                v =
                                    getvalue(u, ia, ib, ix, iy, iz, it) +
                                    getvalue(x, ia, ik, ix, iy, iz, it) *
                                    getvalue(y, ib, ik, ix, iy, iz, it)
                                setvalue!(u, v, ia, ib, ix, iy, iz, it)
                                #u[ia,ib,ix,iy,iz,it] += x[ia,ix,iy,iz,it,ik]*y[ib,ix,iy,iz,it,ik]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(u)
end



"""
mul!(u,x,y) -> u_{ab} = x_a*y_b
"""
function LinearAlgebra.mul!(
    u::T1,
    x::StaggeredFermion_4D_nowing_mpi{NC},
    y::StaggeredFermion_4D_nowing_mpi{NC},
) where {T1<:AbstractGaugefields,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NG = x.NG
    clear_U!(u)


    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ik = 1:NG
                        for ib = 1:NC
                            @simd for ia = 1:NC
                                v =
                                    getvalue(u, ia, ib, ix, iy, iz, it) +
                                    getvalue(x, ia, ik, ix, iy, iz, it) *
                                    getvalue(y, ib, ik, ix, iy, iz, it)
                                setvalue!(u, v, ia, ib, ix, iy, iz, it)

                                #u[ia,ib,ix,iy,iz,it] += x[ia,ix,iy,iz,it,ik]*y[ib,ix,iy,iz,it,ik]
                            end
                        end
                    end
                end
            end
        end
    end
    set_wing_U!(u)
end

function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_nowing_mpi{3},
    x::T3,
    A::T,
) where {T<:Abstractfields,T3<:Abstractfermion}
    #@assert 3 == x.NC "dimension mismatch! NC in y is 3 but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for it = 1:y.PN[4]
        for iz = 1:y.PN[3]
            for iy = 1:y.PN[2]
                for ix = 1:y.PN[1]
                    for ialpha = 1:NG
                        x1 = getvalue(x, 1, ialpha, ix, iy, iz, it)
                        x2 = getvalue(x, 2, ialpha, ix, iy, iz, it)
                        x3 = getvalue(x, 3, ialpha, ix, iy, iz, it)
                        v =
                            x1 * getvalue(A, 1, 1, ix, iy, iz, it) +
                            x2 * getvalue(A, 2, 1, ix, iy, iz, it) +
                            x3 * getvalue(A, 3, 1, ix, iy, iz, it)
                        setvalue!(y, v, 1, ialpha, ix, iy, iz, it)
                        v =
                            x1 * getvalue(A, 1, 2, ix, iy, iz, it) +
                            x2 * getvalue(A, 2, 2, ix, iy, iz, it) +
                            x3 * getvalue(A, 3, 2, ix, iy, iz, it)
                        setvalue!(y, v, 2, ialpha, ix, iy, iz, it)
                        v =
                            x1 * getvalue(A, 1, 3, ix, iy, iz, it) +
                            x2 * getvalue(A, 2, 3, ix, iy, iz, it) +
                            x3 * getvalue(A, 3, 3, ix, iy, iz, it)
                        setvalue!(y, v, 3, ialpha, ix, iy, iz, it)
                    end
                end
            end
        end
    end
end


function LinearAlgebra.mul!(
    y::StaggeredFermion_4D_nowing_mpi{NC},
    A::T,
    x::T3,
) where {NC,T<:Number,T3<:Abstractfermion}
    @assert NC == x.NC "dimension mismatch! NC in y is $NC but NC in x is $(x.NC)"
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    NG = y.NG

    @inbounds for it = 1:y.PN[4]
        for iz = 1:y.PN[3]
            for iy = 1:y.PN[2]
                for ix = 1:y.PN[1]
                    for ialpha = 1:NG
                        for k1 = 1:NC
                            v = A * getvalue(x, k1, ialpha, ix, iy, iz, it)
                            setvalue!(y, v, k1, ialpha, ix, iy, iz, it) # A*getvalue(x,k1,ialpha,ix,iy,iz,it)
                        end
                    end
                end
            end
        end
    end
end

function LinearAlgebra.mul!(
    x::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
    iseven::Bool,
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for ic = 1:NC
                            e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                            e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                            e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                            e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                            v = A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                            setvalue!(x, v, ic, 1, ix, iy, iz, it)
                            v = A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                            setvalue!(x, v, ic, 2, ix, iy, iz, it)
                            v = A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                            setvalue!(x, v, ic, 3, ix, iy, iz, it)
                            v = A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                            setvalue!(x, v, ic, 4, ix, iy, iz, it)
                        end

                        #x[ic,1,ix,iy,iz,it) = A[1,1]*e1+A[1,2]*e2+A[1,3]*e3+A[1,4]*e4
                        #x[ic,2,ix,iy,iz,it] = A[2,1]*e1+A[2,2]*e2+A[2,3]*e3+A[2,4]*e4
                        #x[ic,3,ix,iy,iz,it) = A[3,1]*e1+A[3,2]*e2+A[3,3]*e3+A[3,4]*e4
                        #x[ic,4,ix,iy,iz,it) = A[4,1]*e1+A[4,2]*e2+A[4,3]*e3+A[4,4]*e4
                    end



                end
            end
        end
    end


end

function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
    x::StaggeredFermion_4D_nowing_mpi{NC},
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)
    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ic = 1:NC
                        e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,ix,iy,iz,it,1]
                        e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                        e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                        e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                        v = A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                        setvalue!(xout, v, ic, 1, ix, iy, iz, it)
                        v = A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                        setvalue!(xout, v, ic, 2, ix, iy, iz, it)
                        v = A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                        setvalue!(xout, v, ic, 3, ix, iy, iz, it)
                        v = A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                        setvalue!(xout, v, ic, 4, ix, iy, iz, it)
                    end
                end
            end
        end
    end


end

function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
    x::StaggeredFermion_4D_nowing_mpi{NC},
    iseven::Bool,
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for ic = 1:NC

                            e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,ix,iy,iz,it,1]
                            e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                            e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                            e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                            v = A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
                            setvalue!(xout, v, ic, ix, iy, iz, it, 1)
                            v = A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
                            setvalue!(xout, v, ic, 2, ix, iy, iz, it)
                            v = A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
                            setvalue!(xout, v, ic, 3, ix, iy, iz, it)
                            v = A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
                            setvalue!(xout, v, ic, 4, ix, iy, iz, it)
                        end
                    end


                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_nowing_mpi{NC},
    x::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    for ic = 1:NC
                        e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                        e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                        e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                        e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                        v = A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
                        setvalue!(xout, v, ic, 1, ix, iy, iz, it)
                        v = A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
                        setvalue!(xout, v, ic, 2, ix, iy, iz, it)
                        v = A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
                        setvalue!(xout, v, ic, 3, ix, iy, iz, it)
                        v = A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
                        setvalue!(xout, v, ic, 4, ix, iy, iz, it)

                        #=
                        e1 = x[ic,1,ix,iy,iz,it)
                        e2 = x[ic,2,ix,iy,iz,it]
                        e3 = x[ic,3,ix,iy,iz,it)
                        e4 = x[ic,4,ix,iy,iz,it)

                        xout[ic,1,ix,iy,iz,it) = A[1,1]*e1+A[2,1]*e2+A[3,1]*e3+A[4,1]*e4
                        xout[ic,2,ix,iy,iz,it] = A[1,2]*e1+A[2,2]*e2+A[3,2]*e3+A[4,2]*e4
                        xout[ic,3,ix,iy,iz,it) = A[1,3]*e1+A[2,3]*e2+A[3,3]*e3+A[4,3]*e4
                        xout[ic,4,ix,iy,iz,it) = A[1,4]*e1+A[2,4]*e2+A[3,4]*e3+A[4,4]*e4
                        =#

                    end
                end
            end
        end
    end

end

function LinearAlgebra.mul!(
    xout::StaggeredFermion_4D_nowing_mpi{NC},
    x::StaggeredFermion_4D_nowing_mpi{NC},
    A::TA,
    iseven,
) where {TA<:AbstractMatrix,NC}
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    #n6 = size(x.f)[6]
    #f = zeros(ComplexF64,4)
    #e = zeros(ComplexF64,4)

    @inbounds for it = 1:x.PN[4]
        for iz = 1:x.PN[3]
            for iy = 1:x.PN[2]
                for ix = 1:x.PN[1]
                    evenodd = ifelse((ix + iy + iz + it) % 2 == 0, true, false)
                    if evenodd == iseven
                        for ic = 1:NC
                            e1 = getvalue(x, ic, 1, ix, iy, iz, it)#  x[ic,1,ix,iy,iz,it)
                            e2 = getvalue(x, ic, 2, ix, iy, iz, it)
                            e3 = getvalue(x, ic, 3, ix, iy, iz, it)
                            e4 = getvalue(x, ic, 4, ix, iy, iz, it)

                            v = A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
                            setvalue!(xout, v, ic, 1, ix, iy, iz, it)
                            v = A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
                            setvalue!(xout, v, ic, 2, ix, iy, iz, it)
                            v = A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
                            setvalue!(xout, v, ic, 3, ix, iy, iz, it)
                            v = A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
                            setvalue!(xout, v, ic, 4, ix, iy, iz, it)

                        end
                    end
                end
            end
        end
    end

end

function Base.getindex(
    u::Staggered_Gaugefields{T,μ},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:Gaugefields_4D_nowing_mpi,μ}
    NT = u.parent.NT
    NZ = u.parent.NZ
    NY = u.parent.NY
    NX = u.parent.NX

    t = i6 - 1
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[4],1)
    z = i5 - 1
    z += ifelse(z < 0, NZ, 0)
    z += ifelse(z ≥ NZ, -NZ, 0)
    #boundary_factor_z = ifelse(z == NZ -1,BoundaryCondition[3],1)
    y = i4 - 1
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = i3 - 1
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    elseif μ == 4
        #η = (-1.0)^(x+y+z)
        η = ifelse((x + y + z) % 2 == 0, 1, -1)
    else
        error("η should be positive but η = $η")
    end

    #@inbounds return η * u.parent[i1, i2, i3, i4, i5, i6]
    @inbounds return η * getvalue(u.parent, i1, i2, i3, i4, i5, i6) # u.parent[i1,i2,i3,i4,i5,i6]
end

#function Base.getindex(u::Staggered_Gaugefields{Shifted_Gaugefields_4D_nowing{NC},μ},i1,i2,i3,i4,i5,i6)  where {μ,NC}
#    error("type $(typeof(u)) has no getindex method")
#end

function Base.getindex(
    u::Staggered_Gaugefields{Shifted_Gaugefields_4D_mpi_nowing{NC},μ},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {μ,NC}
    #function Base.getindex(u::Staggered_Gaugefields{T,μ},i1,i2,i3,i4,i5,i6) where {T <: Shifted_Gaugefields_4D,μ}
    NT = u.parent.NT
    NZ = u.parent.NZ
    NY = u.parent.NY
    NX = u.parent.NX

    t = i6 - 1 + u.parent.shift[4]
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[4],1)
    z = i5 - 1 + u.parent.shift[3]
    z += ifelse(z < 0, NZ, 0)
    z += ifelse(z ≥ NZ, -NZ, 0)
    #boundary_factor_z = ifelse(z == NZ -1,BoundaryCondition[3],1)
    y = i4 - 1 + u.parent.shift[2]
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = i3 - 1 + u.parent.shift[1]
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    elseif μ == 4
        #η = (-1.0)^(x+y+z)
        η = ifelse((x + y + z) % 2 == 0, 1, -1)
    else
        error("η should be positive but η = $η")
    end

    @inbounds return η * getvalue(u.parent, i1, i2, i3, i4, i5, i6)
end
