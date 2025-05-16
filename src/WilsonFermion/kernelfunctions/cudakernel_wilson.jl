function cudakernel_gauss_distribution_fermion!(x, σ, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_gauss_distribution_fermion!(b, r, x, σ, NC, NG)
end

function cudakernel_clear_fermion!(a, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_clear_fermion!(b, r, a, NC, NG)
end

function cudakernel_add_fermion!(c, α, a, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_add_fermion!(b, r, c, α, a, NC, NG)
end

function cudakernel_mul_yUdagx_NC3!(y, A, x, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yUdagx_NC3!(b, r, y, A, x, NG)
end

function cudakernel_mul_yUx_NC3!(y, A, x, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    cudakernel_mul_yUx_NC3!(b, r, y, A, x, NG)
end

function cudakernel_axpby!(α, X, β, Y, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_axpby!(b, r, α, X, β, Y, NC)
end

function cudakernel_axpby_NC3NG4!(α, X, β, Y)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_axpby_NC3NG4!(b, r, α, X, β, Y)
end

function cudakernel_mul_Ax!(xout, A, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_Ax!(b, r, xout, A, x, NC)
end

function cudakernel_mul_Ax_NC3NG4!(xout, A, x)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_Ax_NC3NG4!(b, r, xout, A, x)
end

function cudakernel_mul_xA_NC3NG4!(xout, x,A)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_xA_NC3!(b, r, xout, x,A)
end



function cudakernel_mul_yAx_NC3!(y, A, x)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yAx_NC3!(b, r, y, A, x)
end

function cudakernel_mul_yAx_NC3_shifted!(y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yAx_NC3_shifted!(b, r, y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
end

#function cudakernel_mul_yAx_NC3!(y, A, x)
#    b = Int64(CUDA.threadIdx().x)
#    r = Int64(CUDA.blockIdx().x)
#    kernel_mul_yAx_NC3!(b,r,y, A, x)
#end

function cudakernel_mul_ysx_NC!(y, A, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_ysx_NC!(b, r, y, A, x, NC)
end

function cudakernel_mul_ysx_NC3NG4!(y, A, x)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_ysx_NC3NG4!(b, r, y, A, x)
end


function cudakernel_mul_uxy_NC!(u, x, y, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxy_NC!(b, r, u, x, y, NC, NG)
end

function cudakernel_mul_uxy_NC3NG4!(u, x, y)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxy_NC3NG4!(b, r, u, x, y)
end

function cudakernel_mul_uxydag_NC!(u, x, y, NC, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxydag_NC!(b, r, u, x, y, NC, NG)
end

function cudakernel_mul_uxydag_NC3NG4!(u, x, y)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_uxydag_NC3NG4!(b, r, u, x, y)
end


function cudakernel_mul_yxA_NC3!(y, x, A, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yxA_NC3!(b, r, y, x, A, NG)
end

function cudakernel_mul_yxdagAdag_NC3!(y, x, A, NG)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yxdagAdag_NC3!(b, r, y, x, A, NG)
end

function cudakernel_mul_yxdagAdagshifted_NC3!(y, x, A, NG, shift, blockinfo, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_yxdagAdagshifted_NC3!(b, r, y, x, A, NG, shift, blockinfo, bc, NX, NY, NZ, NT)
end


function cudakernel_mul_xA_NC!(xout, x, A, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_xA_NC!(b, r, xout, x, A, NC)
end

function cudakernel_shifted_fermion!(f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)

    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_shifted_fermion!(b, r, f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    return
end

function cudakernel_mul_1plusγ5x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ5x!(b, r, y, x, NC)
end

function cudakernel_mul_1plusγ1x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ1x!(b, r, y, x, NC)
end

function cudakernel_mul_1plusγ2x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ2x!(b, r, y, x, NC)
end

function cudakernel_mul_1plusγ3x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ3x!(b, r, y, x, NC)
end

function cudakernel_mul_1plusγ4x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ4x!(b, r, y, x, NC)
end

function cudakernel_mul_1minusγ1x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ1x!(b, r, y, x, NC)
end

function cudakernel_mul_1minusγ2x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ2x!(b, r, y, x, NC)
end

function cudakernel_mul_1minusγ3x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ3x!(b, r, y, x, NC)
end

function cudakernel_mul_1minusγ4x!(y, x, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ4x!(b, r, y, x, NC)
end

function cudakernel_dot!(temp_volume, A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_dot!(b, r, temp_volume, A, B, NC)
end


function cudakernel_substitute_fermion!(A, B, NC)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_substitute_fermion!(b, r, A, B, NC)
end


function cudakernel_mul_1plusγ5x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ5x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1plusγ1x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ1x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1plusγ2x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ2x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1plusγ3x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ3x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1plusγ4x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1plusγ4x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1minusγ1x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ1x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1minusγ2x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ2x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1minusγ3x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ3x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end

function cudakernel_mul_1minusγ4x_shifted!(y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    b = Int64(CUDA.threadIdx().x)
    r = Int64(CUDA.blockIdx().x)
    kernel_mul_1minusγ4x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
end