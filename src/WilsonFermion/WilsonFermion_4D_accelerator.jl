import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate

include("./kernelfunctions/kernel_wilson.jl")

struct WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted} <: WilsonFermion_4D{NC}
    f::TF
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    fshifted::TFshifted
    blockinfo::Blockindices
    accelerator::String
    temp_volume::TUv
    nocopy::Bool


    function WilsonFermion_4D_accelerator(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        blocks;
        accelerator="none", kwargs...) where {T<:Integer}

        L = [NX, NY, NZ, NT]
        NDW = 0
        NG = 4

        if haskey(kwargs, :noshiftfield)
            nocopy = kwargs[:noshiftfield]
        else
            nocopy = true
        end


        blockinfo = Blockindices(L, blocks)#Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        Dirac_operator = "Wilson"

        fcpu = zeros(ComplexF64, NC, NG, blocksize, rsize)
        temp_volumecpu = zeros(ComplexF64, blocksize, rsize)
        fshiftedcpu = zeros(ComplexF64, NC, NG, blocksize, rsize)




        #f = CUDA.CuArray(fcpu)
        if accelerator == "cuda"
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


        TF = typeof(f)
        TUv = typeof(temp_volume)
        TFshifted = typeof(fshifted)

        return new{NC,TF,NG,TUv,TFshifted}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator, fshifted,
            blockinfo, accelerator, temp_volume, nocopy)


    end


end

function get_myrank(x::T) where {T<:WilsonFermion_4D_accelerator}
    return 0
end

function get_nprocs(x::T) where {T<:WilsonFermion_4D_accelerator}
    return 0
end




function Initialize_WilsonFermion(
    u::Gaugefields_4D_accelerator;
    nowing=false, kwargs...
)
    NX = u.NX
    NY = u.NY
    NZ = u.NZ
    NT = u.NT
    NC = u.NC
    x = WilsonFermion_4D_accelerator(
        NC,
        NX,
        NY,
        NZ,
        NT,
        u.blockinfo.blocks;
        accelerator=u.accelerator, kwargs...)

    return x
end

function Base.similar(x::T) where {T<:WilsonFermion_4D_accelerator}
    return WilsonFermion_4D_accelerator(
        x.NC, x.NX, x.NY, x.NZ, x.NT, x.blockinfo.blocks; accelerator=x.accelerator, noshiftfield=x.nocopy)
end


function gauss_distribution_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG}
) where {NC,TF,NG}

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_gauss_distribution_fermion!(b, r, x.f,
                1, NC, NG)
        end
    end

    return
end

function clear_fermion!(a::WilsonFermion_4D_accelerator{NC,TF,NG}) where {NC,TF,NG}

    for r = 1:a.blockinfo.rsize
        for b = 1:a.blockinfo.blocksize
            kernel_clear_fermion!(b, r, a.f, NC, NG)
        end
    end

end

function add_fermion!(
    c::WilsonFermion_4D_accelerator{NC,TF,NG},
    α::Number,
    a::T1,
) where {NC,T1<:WilsonFermion_4D_accelerator,TF,NG}#c += alpha*a 
    for r = 1:c.blockinfo.rsize
        for b = 1:c.blockinfo.blocksize
            kernel_add_fermion!(b, r, c.f, α, a.f, NC, NG)
        end
    end

end

function Base.getindex(
    F::Adjoint_fermionfields{T},
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
) where {T<:WilsonFermion_4D_accelerator}  #F'
    error("F[i1,i2,i3,i4,i5,i6] is not supported in WilsonFermion_4D_accelerator")
end


function set_wing_fermion!(F::WilsonFermion_4D_accelerator)
end

function set_wing_fermion!(F::T, boundarycondition) where {T<:WilsonFermion_4D_accelerator}
end

function shift_fermion(F::WilsonFermion_4D_accelerator, ν::T) where {T<:Integer}
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

    return Shifted_fermionfields_4D_accelerator(F, shift)
end

const boundarycondition_default_accelerator = [1, 1, 1, -1]

struct Shifted_fermionfields_4D_accelerator{NC,T} <: Shifted_fermionfields{NC,4}
    parent::T
    #parent::T
    shift::NTuple{4,Int8}
    NC::Int64
    bc::NTuple{4,Int8}

    #function Shifted_Gaugefields(U::T,shift,Dim) where {T <: AbstractGaugefields}
    function Shifted_fermionfields_4D_accelerator(
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
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
    boundarycondition,
    shift,
) where {NC,TF,NG}

    bc = boundarycondition
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_shifted_fermion!(b, r, x.f, x.fshifted, x.blockinfo, bc, shift, NC, NX, NY, NZ, NT)
        end
    end
end

function shifted_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    boundarycondition,
    shift,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}

end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1plusγ5x!(b, r, y.f, x.f, NC)
        end
    end

end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1plusγ1x!(b, r, y.f, x.f, NC)
        end
    end

end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1plusγ2x!(b, r, y.f, x.f, NC)
        end
    end

end


function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1plusγ3x!(b, r, y.f, x.f, NC)
        end
    end

end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1plusγ4x!(b, r, y.f, x.f, NC)
        end
    end

end



function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1minusγ1x!(b, r, y.f, x.f, NC)
        end
    end

end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1minusγ2x!(b, r, y.f, x.f, NC)
        end
    end

end


function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1minusγ3x!(b, r, y.f, x.f, NC)
        end
    end

end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:x.blockinfo.rsize
        for b = 1:x.blockinfo.blocksize
            kernel_mul_1minusγ4x!(b, r, y.f, x.f, NC)
        end
    end

end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ5x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ5x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ1x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ1x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ2x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ2x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end



function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ3x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end


function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2
    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ3x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ4x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1plusγ4x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end




function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ1x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ1x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ2x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ2x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end



function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ3x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT
    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ3x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG}#(1+gamma_5)/2


    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ4x!(b, r, y.f, x.parent.fshifted, NC)
        end
    end

end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG,TUv,TFshifted},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF,NG,TUv,TFshifted<:Nothing}#(1+gamma_5)/2

    NX = y.NX
    NY = y.NY
    NZ = y.NZ
    NT = y.NT

    for r = 1:y.blockinfo.rsize
        for b = 1:y.blockinfo.blocksize
            kernel_mul_1minusγ4x_shifted!(b, r, y.f, x.parent.f, x.shift, x.parent.blockinfo, NC, x.bc, NX, NY, NZ, NT)
        end
    end

end


function LinearAlgebra.dot(
    A::WilsonFermion_4D_accelerator{NC,TF,NG},
    B::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}

    for r = 1:A.blockinfo.rsize
        for b = 1:A.blockinfo.blocksize
            kernel_dot!(b, r, A.temp_volume, A.f, B.f, NC)
        end
    end
    s = reduce(+, A.temp_volume)
    return s
end

function substitute_fermion!(
    A::WilsonFermion_4D_accelerator{NC,TF,NG},
    B::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}

    for r = 1:A.blockinfo.rsize
        for b = 1:A.blockinfo.blocksize
            kernel_substitute_fermion!(b, r, A.f, B.f, NC)
        end
    end

end


function substitute_fermion!(
    A::AbstractFermionfields_4D{NC},
    B::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF,NG}
    bcpu = Array(B.f)

    blockinfo = B.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            for ig = 1:NG
                for ic = 1:NC
                    A[ic, ix, iy, iz, it, ig] = bcpu[ic, ig, b, r]
                end
            end
        end
    end

end

function substitute_fermion!(
    A::WilsonFermion_4D_accelerator{NC,TF,NG},
    B::AbstractFermionfields_4D{NC},
) where {NC,TF,NG}
    acpu = Array(A.f)

    blockinfo = A.blockinfo
    for r = 1:blockinfo.rsize
        for b = 1:blockinfo.blocksize
            ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
            for ig = 1:NG
                for ic = 1:NC
                    acpu[ic, ig, b, r] = B[ic, ix, iy, iz, it, ig]
                end
            end
        end
    end
    A.f .= acpu
end


