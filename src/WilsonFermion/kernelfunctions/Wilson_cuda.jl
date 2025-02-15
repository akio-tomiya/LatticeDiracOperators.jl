include("./cudakernel_wilson.jl")

function gauss_distribution_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG}
) where {NC,TF <: CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_gauss_distribution_fermion!( x.f,
        1, NC, NG)
    end


    return
end

function clear_fermion!(a::WilsonFermion_4D_accelerator{NC,TF,NG}) where {NC,TF <:CUDA.CuArray,NG}
    CUDA.@sync begin
        CUDA.@cuda threads = a.blockinfo.blocksize blocks = a.blockinfo.rsize cudakernel_clear_fermion!(a.f, NC, NG)
    end
end

function add_fermion!(
    c::WilsonFermion_4D_accelerator{NC,TF,NG},
    α::Number,
    a::T1,
) where {NC,T1<:WilsonFermion_4D_accelerator,TF<:CUDA.CuArray,NG}#c += alpha*a 

    CUDA.@sync begin
        CUDA.@cuda threads = c.blockinfo.blocksize blocks = c.blockinfo.rsize cudakernel_add_fermion!(c.f, α, a.f, NC, NG)
    end

end


function shifted_fermion!(
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
    boundarycondition,
    shift,
) where {NC,TF<:CUDA.CuArray,NG}

    bc = Tuple(boundarycondition)
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_shifted_fermion!(x.f, x.fshifted, x.blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    end


end



function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1plusγ5x!(y.f, x.f, NC)
    end
end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1plusγ1x!(y.f, x.f, NC)
    end
end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1plusγ2x!(y.f, x.f, NC)
    end

end


function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1plusγ3x!(y.f, x.f, NC)
    end
end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1plusγ4x!(y.f, x.f, NC)
    end


end



function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1minusγ1x!(y.f, x.f, NC)
    end


end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1minusγ2x!(y.f, x.f, NC)
    end

end


function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1minusγ3x!(y.f, x.f, NC)
    end


end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_mul_1minusγ4x!(y.f, x.f, NC)
    end
end


function mul_1plusγ5x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1plusγ5x!(y.f, x.parent.fshifted, NC)
    end


end

function mul_1plusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1plusγ1x!(y.f, x.parent.fshifted, NC)
    end




end

function mul_1plusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1plusγ2x!(y.f, x.parent.fshifted, NC)
    end



end


function mul_1plusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1plusγ3x!(y.f, x.parent.fshifted, NC)
    end


end

function mul_1plusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1plusγ4x!(y.f, x.parent.fshifted, NC)
    end



end



function mul_1minusγ1x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1minusγ1x!(y.f, x.parent.fshifted, NC)
    end


end

function mul_1minusγ2x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1minusγ2x!(y.f, x.parent.fshifted, NC)
    end


end


function mul_1minusγ3x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2


    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1minusγ3x!(y.f, x.parent.fshifted, NC)
    end


end

function mul_1minusγ4x!(
    y::WilsonFermion_4D_accelerator{NC,TF,NG},
    x::Shifted_fermionfields_4D_accelerator,
) where {NC,TF<:CUDA.CuArray,NG}#(1+gamma_5)/2

    CUDA.@sync begin
        CUDA.@cuda threads = y.blockinfo.blocksize blocks = y.blockinfo.rsize cudakernel_mul_1minusγ4x!(y.f, x.parent.fshifted, NC)
    end
end


function LinearAlgebra.dot(
    A::WilsonFermion_4D_accelerator{NC,TF,NG},
    B::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = A.blockinfo.blocksize blocks = A.blockinfo.rsize cudakernel_dot!( A.temp_volume, A.f, B.f, NC)
    end
    s = CUDA.reduce(+, A.temp_volume)

    return s
end

function substitute_fermion!(
    A::WilsonFermion_4D_accelerator{NC,TF,NG},
    B::WilsonFermion_4D_accelerator{NC,TF,NG},
) where {NC,TF<:CUDA.CuArray,NG}

    CUDA.@sync begin
        CUDA.@cuda threads = A.blockinfo.blocksize blocks = A.blockinfo.rsize cudakernel_substitute_fermion!(A.f, B.f, NC)
    end


end

