
import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate

import Gaugefields.AbstractGaugefields_module:
    Staggered_Gaugefields, Shifted_Gaugefields_4D_accelerator


struct StaggeredFermion_4D_accelerator{NC,TF,accdevise,TUv,TFshifted} <: AbstractFermionfields_4D{NC}
    f::TF
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NDW::Int64
    Dirac_operator::String
    fshifted::TFshifted
    blockinfo::Blockindices
    accelerator::String
    temp_volume::TUv
    nocopy::Bool


    function StaggeredFermion_4D_accelerator(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        blocks;
        accelerator="none", kwargs...) where {T<:Integer}

        L = [NX, NY, NZ, NT]
        NDW = 0

        if haskey(kwargs, :noshiftfield)
            nocopy = kwargs[:noshiftfield]
        else
            nocopy = false
        end


        blockinfo = Blockindices(L, blocks)#Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        Dirac_operator = "Staggered"

        fcpu = zeros(ComplexF64, NC, blocksize, rsize)
        temp_volumecpu = zeros(ComplexF64, blocksize, rsize)
        fshiftedcpu = zeros(ComplexF64, NC, blocksize, rsize)


        #println(accelerator)

        #f = CUDA.CuArray(fcpu)
        if accelerator == "cuda"
            error("accelerator = cuda is not supported. use JACC instead.")
            iscudadefined = @isdefined CUDA
            #println(iscudadefined)
            if iscudadefined
                if CUDA.has_cuda()
                    f = CUDA.CuArray(fcpu)
                    temp_volume = CUDA.CuArray(temp_volumecpu)
                    if nocopy
                        fshifted = nothing
                    else
                        fshifted = CUDA.CuArray(fshiftedcpu)
                    end
                    accdevise = :cuda
                else
                    @warn "accelerator=\"cuda\" is set but there is no CUDA devise. CPU will be used"
                    f = fcpu
                    temp_volume = temp_volumecpu
                    if nocopy
                        fshifted = nothing
                    else
                        fshifted = fshiftedcpu
                    end
                    accdevise = :none
                end

            else
                f = fcpu
                temp_volume = temp_volumecpu
                if nocopy
                    fshifted = nothing
                else
                    fshifted = fshiftedcpu
                end
                accdevise = :none
            end
        elseif accelerator == "JACC"
            #println("JACC is used")
            isjaccdefined = @isdefined JACC
            #println("isjaccdefined = $isjaccdefined")
            if isjaccdefined
                NN = NX * NY * NZ * NT
                fcpu = zeros(ComplexF64, NC, NN)
                temp_volumecpu = zeros(ComplexF64, NN)
                fshiftedcpu = zeros(ComplexF64, NC, NN)



                #Ucpu = zeros(dtype, NC, NC, NV)
                #temp_volume_cpu = zeros(dtype, NV)

                f = JACC.array(fcpu)
                temp_volume = JACC.array(temp_volumecpu)
                if nocopy
                    fshifted = nothing
                else
                    fshifted = JACC.array(fshiftedcpu)
                end

                accdevise = :jacc
                #println(typeof(U))
            else
                error("JACC should be used.")
                f = fcpu
                temp_volume = temp_volumecpu
                if nocopy
                    fshifted = nothing
                else
                    fshifted = fshiftedcpu
                end
                accdevise = :none
            end
        else
            error("only accelerator = JACC is supported")
            f = fcpu
            temp_volume = temp_volumecpu
            if nocopy
                fshifted = nothing
            else
                fshifted = fshiftedcpu
            end
            accdevise = :none
        end


        TF = typeof(f)
        TUv = typeof(temp_volume)
        TFshifted = typeof(fshifted)

        return new{NC,TF,accdevise,TUv,TFshifted}(f, NC, NX, NY, NZ, NT, NDW, Dirac_operator, fshifted,
            blockinfo, accelerator, temp_volume, nocopy)
    end


end

function get_myrank(x::T) where {T<:StaggeredFermion_4D_accelerator}
    return 0
end

function get_nprocs(x::T) where {T<:StaggeredFermion_4D_accelerator}
    return 0
end


function Initialize_StaggeredFermion(
    u::Gaugefields_4D_accelerator;
    nowing=false, kwargs...
)
    NX = u.NX
    NY = u.NY
    NZ = u.NZ
    NT = u.NT
    NC = u.NC
    x = StaggeredFermion_4D_accelerator(
        NC,
        NX,
        NY,
        NZ,
        NT,
        u.blockinfo.blocks;
        accelerator=u.accelerator, kwargs...)

    return x
end

function Base.similar(x::T) where {T<:StaggeredFermion_4D_accelerator}
    return StaggeredFermion_4D_accelerator(
        x.NC, x.NX, x.NY, x.NZ, x.NT, x.blockinfo.blocks; accelerator=x.accelerator, noshiftfield=x.nocopy)
end

function gauss_distribution_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:none}
) where {NC,TF}

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_gauss_distribution_staggeredfermion!(b, r, x.f,
                1, NC)
        end
    end

    return
end

function gauss_distribution_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:none},
    randomfunc,
    σ
) where {NC,TF}

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_gauss_distribution_staggeredfermion!(b, r, x.f,
                σ, NC)
        end
    end

    return
end


function clear_fermion!(a::StaggeredFermion_4D_accelerator{NC,TF,:none}) where {NC,TF}

    for r = 1:a.blockinfo.rsize
        for b = 1:a.blockinfo.blocksize
            kernel_clear_staggeredfermion!(b, r, a.f, NC)
        end
    end

end



function add_fermion!(
    c::StaggeredFermion_4D_accelerator{NC,TF,:none},
    α::Number,
    a::T1,
    β::Number,
    B::T1
) where {NC,T1<:StaggeredFermion_4D_accelerator,TF}#c += alpha*a 
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_staggeredfermion!(b, r, c.f, α, a.f, β, B.f, NC)
        end
    end

end


function add_fermion!(
    c::StaggeredFermion_4D_accelerator{NC,TF,:none},
    α::Number,
    a::T1,
) where {NC,T1<:StaggeredFermion_4D_accelerator,TF}#c += alpha*a 
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_staggeredfermion!(b, r, c.f, α, a.f, NC)
        end
    end

end

function Base.getindex(
    F::Adjoint_fermionfields{T},
    i1,
    i2,
    i3,
    i4,
    i5
) where {T<:StaggeredFermion_4D_accelerator}  #F'
    error("F[i1,i2,i3,i4,i5] is not supported in StaggeredFermion_4D_accelerator")
end


function set_wing_fermion!(F::StaggeredFermion_4D_accelerator)
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:StaggeredFermion_4D_accelerator}
end

function shift_fermion(F::StaggeredFermion_4D_accelerator, ν::T;
    boundarycondition=boundarycondition_default) where {T<:Integer}
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

    return Shifted_staggeredfermionfields_4D_accelerator(F, shift;
        boundarycondition)
end


struct Shifted_staggeredfermionfields_4D_accelerator{NC,T} <: Shifted_fermionfields{NC,4}
    parent::T
    #parent::T
    shift::NTuple{4,Int8}
    NC::Int64
    bc::NTuple{4,Int8}

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_staggeredfermionfields_4D_accelerator(
        F,
        shift;
        boundarycondition=boundarycondition_default_accelerator,
    )
        NC = F.NC
        bc = Tuple(boundarycondition)

        shifted_fermion!(F, boundarycondition, shift)
        return new{NC,typeof(F)}(F, shift, NC, bc)
    end
end

function shifted_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:none},
    boundarycondition,
    shift,
) where {NC,TF}

    bc = boundarycondition
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_shifted_staggeredfermion!(b, r, x.f, x.fshifted, x.blockinfo, bc, shift, NC, NX, NY, NZ, NT)
        end
    end
end

function shifted_fermion!(
    x::StaggeredFermion_4D_accelerator{NC,TF,:none,TUv,TFshifted},
    boundarycondition,
    shift,
) where {NC,TF,TUv,TFshifted<:Nothing}

end



function LinearAlgebra.dot(
    A::StaggeredFermion_4D_accelerator{NC,TF,:none},
    B::StaggeredFermion_4D_accelerator{NC,TF,:none},
) where {NC,TF}

    for r = 1:A.blockinfo.rsize
        for b = 1:A.blockinfo.blocksize
            kernel_staggereddot!(b, r, A.temp_volume, A.f, B.f, NC)
        end
    end
    s = reduce(+, A.temp_volume)
    return s
end

function substitute_fermion!(
    A::StaggeredFermion_4D_accelerator{NC,TF,:none},
    B::StaggeredFermion_4D_accelerator{NC,TF,:none},
) where {NC,TF}

    for r = 1:A.blockinfo.rsize
        for b = 1:A.blockinfo.blocksize
            kernel_substitute_staggeredfermion!(b, r, A.f, B.f, NC)
        end
    end

end


function substitute_fermion!(
    A::AbstractFermionfields_4D{NC},
    B::StaggeredFermion_4D_accelerator{NC,TF,:none},
) where {NC,TF}
    bcpu = Array(B.f)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            #for ig = 1:NG
            for ic = 1:NC
                A[ic, ix, iy, iz, it] = bcpu[ic, b, r]
            end
            #end
        end
    end

end

function substitute_fermion!(
    A::StaggeredFermion_4D_accelerator{NC,TF,:none},
    B::AbstractFermionfields_4D{NC},
) where {NC,TF}
    acpu = Array(A.f)

    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            #for ig = 1:NG
            for ic = 1:NC
                acpu[ic, b, r] = B[ic, ix, iy, iz, it]
            end
            #end
        end
    end
    A.f .= acpu
end




function Dx!(
    xout::T,
    U::Array{G,1},
    x::T,
    temps::Array{T,1},
    boundarycondition,
) where {T<:StaggeredFermion_4D_accelerator,G<:AbstractGaugefields}
    #temp = temps[4]
    temp1 = temps[1]
    temp2 = temps[2]

    #clear!(temp)
    set_wing_fermion!(x, boundarycondition)
    clear_fermion!(xout)
    for ν = 1:4
        xplus = shift_fermion(x, ν; boundarycondition)
        Us = staggered_U(U[ν], ν)
        mul!(temp1, Us, xplus)


        xminus = shift_fermion(x, -ν; boundarycondition)
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

function Base.getindex(
    u::Staggered_Gaugefields{T,μ},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:Gaugefields_4D_accelerator,μ}
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
    u::Staggered_Gaugefields{Shifted_Gaugefields_4D_accelerator{NC},μ},
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
