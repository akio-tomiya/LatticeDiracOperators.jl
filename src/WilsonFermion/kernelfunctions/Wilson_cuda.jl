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

    bc = boundarycondition
    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT

    CUDA.@sync begin
        CUDA.@cuda threads = x.blockinfo.blocksize blocks = x.blockinfo.rsize cudakernel_shifted_fermion!(x.f, x.fshifted, x.blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    end


end
