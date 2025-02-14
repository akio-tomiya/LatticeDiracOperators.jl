function cudakernel_gauss_distribution_fermion!(x, σ, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_gauss_distribution_fermion!(b, r, x,σ, NC, NG)
end

function cudakernel_clear_fermion!(a, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_clear_fermion!(b, r, a, NC, NG)
end

function cudakernel_add_fermion!(c, α, a, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_fermion!(b,r,c, α, a, NC, NG)
end

function cudakernel_mul_yUdagx_NC3!( y, A, x, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yUdagx_NC3!(b,r, y, A, x, NG)
end

function cudakernel_mul_yUx_NC3!( y, A, x, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    cudakernel_mul_yUx_NC3!(b,r, y, A, x, NG)
end

function cudakernel_axpby!(α, X, β, Y, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_axpby!(b,r,α, X, β, Y, NC)
end

function cudakernel_mul_Ax!(xout, A, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_Ax!(b,r,xout, A, x, NC)
end

function cudakernel_mul_yAx_NC3!(y, A, x)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yAx_NC3!(b,r,y, A, x)
end

function cudakernel_mul_yAx_NC3!(y, A, x)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yAx_NC3!(b,r,y, A, x)
end

function cudakernel_mul_ysx_NC!(y, A, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_ysx_NC!(b,r,y, A, x, NC)
end

function cudakernel_mul_uxy_NC!(u, x, y, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxy_NC!(b,r,u, x, y, NC, NG)
end

function cudakernel_mul_uxydag_NC!(u, x, y, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxydag_NC!(b,r,u, x, y, NC, NG)
end

function cudakernel_mul_yxA_NC3!(y, x, A, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yxA_NC3!(b, r, y, x, A, NG)
end

function cudakernel_mul_yxdagAdag_NC3!(y, x, A, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yxdagAdag_NC3!(b,r,y, x, A, NG)
end

function cudakernel_mul_xA_NC!(xout, x, A, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_xA_NC!(b,r,xout, x, A, NC)
end

function cudakernel_shifted_fermion!(f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_shifted_fermion!(b,r,f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)
end