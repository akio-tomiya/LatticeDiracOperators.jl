import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate

struct WilsonFermion_4D_accelerator{NC,TF,NG} <: WilsonFermion_4D{NC}
    f::TF
    NC::Int64
    NX::Int64
    NY::Int64
    NZ::Int64
    NT::Int64
    NG::Int64
    NDW::Int64
    Dirac_operator::String
    fshifted::TF
    blockinfo::Blockindices
    accelerator::String


    function WilsonFermion_4D_accelerator(
        NC::T,
        NX::T,
        NY::T,
        NZ::T,
        NT::T,
        blocks;
        accelerator="none") where {T<:Integer}

        L = [NX, NY, NZ, NT]
        NDW = 0
        NG = 4

        blockinfo = Blockindices(L, blocks)#Blockindices(Tuple(blocks),Tuple(blocks_s),Tuple(blocknumbers),Tuple(blocknumbers_s),blocksize,rsize)
        blocksize = blockinfo.blocksize
        rsize = blockinfo.rsize

        Dirac_operator = "Wilson"

        fcpu = zeros(ComplexF64, NC, NG, blocksize, rsize)


        #f = CUDA.CuArray(fcpu)
        if accelerator == "cuda"
            iscudadefined = @isdefined CUDA
            #println(iscudadefined)
            if iscudadefined
                if CUDA.has_cuda()
                    f = CUDA.CuArray(fcpu)
                    accdevise = :cuda
                else
                    @warn "accelerator=\"cuda\" is set but there is no CUDA devise. CPU will be used"
                    f = fcpu
                    accdevise = :none
                end
            else
                f = fcpu
                accdevise = :none
            end
        else
            f = fcpu
            accdevise = :none
        end

        fshifted = similar(f)
        TF = typeof(f)

        return new{NC,TF,NG}(f, NC, NX, NY, NZ, NT, NG, NDW, Dirac_operator, fshifted,
            blockinfo, accelerator)


    end


end

function Initialize_WilsonFermion(
    u::Gaugefields_4D_accelerator;
    nowing=false,
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
        accelerator=u.accelerator)

    return x
end

function Base.similar(x::T) where {T<:WilsonFermion_4D_accelerator}
    return WilsonFermion_4D_accelerator(
        x.NC, x.NX, x.NY, x.NZ, x.NT, x.blockinfo.blocks; accelerator=x.accelerator)
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

    for r = 1:a.blockinfo.rsize
        for b = 1:a.blockinfo.blocksize
            kernel_add_fermion!(b, r, c.f, α, a.f, NC, NG)
        end
    end

end


function set_wing_fermion!(F::WilsonFermion_4D_accelerator)
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


