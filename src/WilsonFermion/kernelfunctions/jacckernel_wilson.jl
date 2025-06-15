
import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate, index_to_coords, coords_to_index



import Gaugefields.AbstractGaugefields_module: shiftedindex

function jacckernel_gauss_distribution_fermion!(i, x, σ, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            v1 = sqrt(-log(rand() + 1e-10))
            v2 = 2pi * rand()

            xr = v1 * cos(v2)
            xi = v1 * sin(v2)
            x[ic, mu, i] = σ * xr + σ * im * xi
        end
    end

end

function jacckernel_clear_fermion!(i, a, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            a[ic, mu, i] = 0
        end
    end
end

#kernel_add_fermion!(i, c.f, α, a.f, NC, NG)
function jacckernel_add_fermion!(i, c, α, a, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            c[ic, mu, i] += α * a[ic, mu, i]
        end
    end
end

function jacckernel_add_fermion!(i, c, α, a, β, B, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            c[ic, mu, i] += α * a[ic, mu, i] + β * B[ic, mu, i]
        end
    end
end

function jacckernel_mul_yUx_NC3!(i, y, A, x, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, i]
        x2 = x[2, ialpha, i]
        x3 = x[3, ialpha, i]

        y[1, ialpha, i] =
            A[1, 1, i] * x1 +
            A[1, 2, i] * x2 +
            A[1, 3, i] * x3
        y[2, ialpha, i] =
            A[2, 1, i] * x1 +
            A[2, 2, i] * x2 +
            A[2, 3, i] * x3
        y[3, ialpha, i] =
            A[3, 1, i] * x1 +
            A[3, 2, i] * x2 +
            A[3, 3, i] * x3
    end
end


function jacckernel_mul_yUdagx_NC3!(i, y, A, x, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, i]
        x2 = x[2, ialpha, i]
        x3 = x[3, ialpha, i]

        y[1, ialpha, i] =
            conj(A[1, 1, i]) * x1 +
            conj(A[2, 1, i]) * x2 +
            conj(A[3, 1, i]) * x3
        y[2, ialpha, i] =
            conj(A[1, 2, i]) * x1 +
            conj(A[2, 2, i]) * x2 +
            conj(A[3, 2, i]) * x3
        y[3, ialpha, i] =
            conj(A[1, 3, i]) * x1 +
            conj(A[2, 3, i]) * x2 +
            conj(A[3, 3, i]) * x3
    end
end

function jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)
    #ix, iy, iz, it = fourdim_cordinate(i, blockinfo)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    it_shifted = it + shift[4]
    iz_shifted = iz + shift[3]
    iy_shifted = iy + shift[2]
    ix_shifted = ix + shift[1]
    factor_t = ifelse(it_shifted > NT || it_shifted < 1, bc[4], 1)
    factor_z = ifelse(iz_shifted > NZ || iz_shifted < 1, bc[3], 1)
    factor_y = ifelse(iy_shifted > NY || iy_shifted < 1, bc[2], 1)
    factor_x = ifelse(ix_shifted > NX || ix_shifted < 1, bc[1], 1)
    return factor_x * factor_y * factor_z * factor_t
end

function jacckernel_shifted_fermion!(i, f, fshifted, bc, shift, NC, NX, NY, NZ, NT)
    #ix, iy, iz, it = fourdim_cordinate(i, blockinfo)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)

    #inside_up = it_shifted > NT
    #inside_down = it_shifted < 1
    #ix,iy,iz,it = 1,1,1,1


    it_shifted = it + shift[4]
    iz_shifted = iz + shift[3]
    iy_shifted = iy + shift[2]
    ix_shifted = ix + shift[1]
    factor_t = ifelse(it_shifted > NT || it_shifted < 1, bc[4], 1)
    factor_z = ifelse(iz_shifted > NZ || iz_shifted < 1, bc[3], 1)
    factor_y = ifelse(iy_shifted > NY || iy_shifted < 1, bc[2], 1)
    factor_x = ifelse(ix_shifted > NX || ix_shifted < 1, bc[1], 1)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    @inbounds for ig = 1:4
        for ic = 1:NC
            fshifted[ic, ig, i] = factor_x *
                                  factor_y *
                                  factor_z *
                                  factor_t *
                                  f[ic, ig, i_shifted]
        end
    end

    return
end

function jacckernel_mul_1plusγ5x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        y[ic, 1, i] = 0#-1*x[ic,ix,iy,iz,it,1]
        y[ic, 2, i] = 0#-1*x[ic,ix,iy,iz,it,2]
        y[ic, 3, i] = x[ic, 3, i]
        y[ic, 4, i] = x[ic, 4, i]
    end

end

function jacckernel_mul_1minusγ5x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        y[ic, 1, i] = x[ic, 1, i]
        y[ic, 2, i] = x[ic, 2, i]
        y[ic, 3, i] = 0#x[ic, 3, i]
        y[ic, 4, i] = 0#x[ic, 4, i]
    end

end


function jacckernel_mul_1plusγ5x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        y[ic, 1, i] = 0#-1*x[ic,ix,iy,iz,it,1]
        y[ic, 2, i] = 0#-1*x[ic,ix,iy,iz,it,2]
        y[ic, 3, i] = x[ic, 3, i_shifted] * factor
        y[ic, 4, i] = x[ic, 4, i_shifted] * factor
    end

end


function jacckernel_mul_1plusγ1x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] - im * x[ic, 4, i]
        v2 = x[ic, 2, i] - im * x[ic, 3, i]
        v3 = x[ic, 3, i] + im * x[ic, 2, i]
        v4 = x[ic, 4, i] + im * x[ic, 1, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end


function jacckernel_mul_1plusγ1x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] - im * x[ic, 4, i_shifted] * factor
        v2 = x[ic, 2, i_shifted] - im * x[ic, 3, i_shifted] * factor
        v3 = x[ic, 3, i_shifted] + im * x[ic, 2, i_shifted] * factor
        v4 = x[ic, 4, i_shifted] + im * x[ic, 1, i_shifted] * factor
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end


function jacckernel_mul_1minusγ1x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] + im * x[ic, 4, i]
        v2 = x[ic, 2, i] + im * x[ic, 3, i]
        v3 = x[ic, 3, i] - im * x[ic, 2, i]
        v4 = x[ic, 4, i] - im * x[ic, 1, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1minusγ1x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] + im * x[ic, 4, i_shifted]
        v2 = x[ic, 2, i_shifted] + im * x[ic, 3, i_shifted]
        v3 = x[ic, 3, i_shifted] - im * x[ic, 2, i_shifted]
        v4 = x[ic, 4, i_shifted] - im * x[ic, 1, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end

function jacckernel_mul_1plusγ2x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] - x[ic, 4, i]
        v2 = x[ic, 2, i] + x[ic, 3, i]
        v3 = x[ic, 3, i] + x[ic, 2, i]
        v4 = x[ic, 4, i] - x[ic, 1, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end


function jacckernel_mul_1plusγ2x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)


    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] - x[ic, 4, i_shifted]
        v2 = x[ic, 2, i_shifted] + x[ic, 3, i_shifted]
        v3 = x[ic, 3, i_shifted] + x[ic, 2, i_shifted]
        v4 = x[ic, 4, i_shifted] - x[ic, 1, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end


function jacckernel_mul_1minusγ2x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] + x[ic, 4, i]
        v2 = x[ic, 2, i] - x[ic, 3, i]
        v3 = x[ic, 3, i] - x[ic, 2, i]
        v4 = x[ic, 4, i] + x[ic, 1, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1minusγ2x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] + x[ic, 4, i_shifted]
        v2 = x[ic, 2, i_shifted] - x[ic, 3, i_shifted]
        v3 = x[ic, 3, i_shifted] - x[ic, 2, i_shifted]
        v4 = x[ic, 4, i_shifted] + x[ic, 1, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end





function jacckernel_mul_1plusγ3x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] - im * x[ic, 3, i]
        v2 = x[ic, 2, i] + im * x[ic, 4, i]
        v3 = x[ic, 3, i] + im * x[ic, 1, i]
        v4 = x[ic, 4, i] - im * x[ic, 2, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1plusγ3x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] - im * x[ic, 3, i_shifted]
        v2 = x[ic, 2, i_shifted] + im * x[ic, 4, i_shifted]
        v3 = x[ic, 3, i_shifted] + im * x[ic, 1, i_shifted]
        v4 = x[ic, 4, i_shifted] - im * x[ic, 2, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end


function jacckernel_mul_1minusγ3x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] + im * x[ic, 3, i]
        v2 = x[ic, 2, i] - im * x[ic, 4, i]
        v3 = x[ic, 3, i] - im * x[ic, 1, i]
        v4 = x[ic, 4, i] + im * x[ic, 2, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1minusγ3x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] + im * x[ic, 3, i_shifted]
        v2 = x[ic, 2, i_shifted] - im * x[ic, 4, i_shifted]
        v3 = x[ic, 3, i_shifted] - im * x[ic, 1, i_shifted]
        v4 = x[ic, 4, i_shifted] + im * x[ic, 2, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end



function jacckernel_mul_1plusγ4x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] - x[ic, 3, i]
        v2 = x[ic, 2, i] - x[ic, 4, i]
        v3 = x[ic, 3, i] - x[ic, 1, i]
        v4 = x[ic, 4, i] - x[ic, 2, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1plusγ4x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)



    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] - x[ic, 3, i_shifted]
        v2 = x[ic, 2, i_shifted] - x[ic, 4, i_shifted]
        v3 = x[ic, 3, i_shifted] - x[ic, 1, i_shifted]
        v4 = x[ic, 4, i_shifted] - x[ic, 2, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end

function jacckernel_mul_1minusγ4x!(i, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i] + x[ic, 3, i]
        v2 = x[ic, 2, i] + x[ic, 4, i]
        v3 = x[ic, 3, i] + x[ic, 1, i]
        v4 = x[ic, 4, i] + x[ic, 2, i]
        y[ic, 1, i] = v1
        y[ic, 2, i] = v2
        y[ic, 3, i] = v3
        y[ic, 4, i] = v4
    end

end

function jacckernel_mul_1minusγ4x_shifted!(i, y, x, shift, NC, bc, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    @inbounds for ic = 1:NC
        v1 = x[ic, 1, i_shifted] + x[ic, 3, i_shifted]
        v2 = x[ic, 2, i_shifted] + x[ic, 4, i_shifted]
        v3 = x[ic, 3, i_shifted] + x[ic, 1, i_shifted]
        v4 = x[ic, 4, i_shifted] + x[ic, 2, i_shifted]
        y[ic, 1, i] = v1 * factor
        y[ic, 2, i] = v2 * factor
        y[ic, 3, i] = v3 * factor
        y[ic, 4, i] = v4 * factor
    end

end


function jacckernel_dot!(i, A, B, NC)
    res = zero(eltype(A))
    #temp_volume[i] = 0
    @inbounds for α = 1:4
        for ic = 1:NC
            res += conj(A[ic, α, i]) * B[ic, α, i]
        end
    end
    return res
end

function jacckernel_substitute_fermion!(i, A, B, NC)
    @inbounds for α = 1:4
        for ic = 1:NC
            A[ic, α, i] = B[ic, α, i]
        end
    end
end

function jacckernel_axpby!(i, α, X, β, Y, NC)
    @inbounds for ig = 1:4
        for ic = 1:NC
            Y[ic, ig, i] =
                α * X[ic, ig, i] +
                β * Y[ic, ig, i]
        end
    end
end

function jacckernel_axpby_NC3NG4!(i, α, X, β, Y)
    @inbounds for ig = 1:4
        for ic = 1:3
            Y[ic, ig, i] =
                α * X[ic, ig, i] +
                β * Y[ic, ig, i]
        end
    end
end


function jacckernel_mul_Ax!(i, xout, A, x, NC)
    @inbounds for ic = 1:NC
        e1 = x[ic, 1, i]
        e2 = x[ic, 2, i]
        e3 = x[ic, 3, i]
        e4 = x[ic, 4, i]

        xout[ic, 1, i] =
            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
        xout[ic, 2, i] =
            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
        xout[ic, 3, i] =
            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
        xout[ic, 4, i] =
            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
    end
end

function jacckernel_mul_Ax_NC3NG4!(i, xout, A, x)
    @inbounds for ic = 1:3
        e1 = x[ic, 1, i]
        e2 = x[ic, 2, i]
        e3 = x[ic, 3, i]
        e4 = x[ic, 4, i]

        xout[ic, 1, i] =
            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
        xout[ic, 2, i] =
            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
        xout[ic, 3, i] =
            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
        xout[ic, 4, i] =
            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
    end
end

function jacckernel_mul_yAx_NC3!(i, y, A, x)
    @inbounds for ialpha = 1:4
        x1 = x[1, ialpha, i]
        x2 = x[2, ialpha, i]
        x3 = x[3, ialpha, i]

        y[1, ialpha, i] =
            A[1, 1, i] * x1 +
            A[1, 2, i] * x2 +
            A[1, 3, i] * x3
        y[2, ialpha, i] =
            A[2, 1, i] * x1 +
            A[2, 2, i] * x2 +
            A[2, 3, i] * x3
        y[3, ialpha, i] =
            A[3, 1, i] * x1 +
            A[3, 2, i] * x2 +
            A[3, 3, i] * x3
        # =#
    end
end

function jacckernel_mul_yAx_NC3_shifted!(i, y, A, x, shift, bc, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)



    @inbounds for ialpha = 1:4
        x1 = x[1, ialpha, i_shifted] * factor
        x2 = x[2, ialpha, i_shifted] * factor
        x3 = x[3, ialpha, i_shifted] * factor

        y[1, ialpha, i] =
            A[1, 1, i] * x1 +
            A[1, 2, i] * x2 +
            A[1, 3, i] * x3
        y[2, ialpha, i] =
            A[2, 1, i] * x1 +
            A[2, 2, i] * x2 +
            A[2, 3, i] * x3
        y[3, ialpha, i] =
            A[3, 1, i] * x1 +
            A[3, 2, i] * x2 +
            A[3, 3, i] * x3
        # =#
    end
end


function jacckernel_mul_ysx_NC!(i, y, A, x, NC)
    @inbounds for ialpha = 1:4
        for k1 = 1:NC
            y[k1, ialpha, i] =
                A * x[k1, ialpha, i]
        end
    end
end

function jacckernel_mul_ysx_NC3NG4!(i, y, A, x)
    @inbounds for ialpha = 1:4
        for k1 = 1:3
            y[k1, ialpha, i] =
                A * x[k1, ialpha, i]
        end
    end
end


function jacckernel_mul_uxy_NC!(i, u, x, y, NC, NG)
    @inbounds for ik = 1:NG
        for ib = 1:NC
            for ia = 1:NC
                c = x[ia, ik, i] * y[ib, ik, i]
                u[ia, ib, i] += c
            end
        end
    end
end

function jacckernel_mul_uxy_NC3NG4!(i, u, x, y)
    @inbounds for ik = 1:4
        for ib = 1:3
            for ia = 1:3
                c = x[ia, ik, i] * y[ib, ik, i]
                u[ia, ib, i] += c
            end
        end
    end
end

function jacckernel_mul_uxydag_NC!(i, u, x, y, NC, NG)
    @inbounds for ik = 1:NG
        for ib = 1:NC
            for ia = 1:NC
                c = x[ia, ik, i] * conj(y[ib, ik, i])
                u[ia, ib, i] += c
            end
        end
    end
end

function jacckernel_mul_uxydag_NC3NG4!(i, u, x, y)
    @inbounds for ik = 1:4
        for ib = 1:3
            for ia = 1:3
                c = x[ia, ik, i] * conj(y[ib, ik, i])
                u[ia, ib, i] += c
            end
        end
    end
end


function jacckernel_mul_yxA_NC3!(i, y, x, A, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, i]
        x2 = x[2, ialpha, i]
        x3 = x[3, ialpha, i]
        y[1, ialpha, i] =
            x1 * A[1, 1, i] +
            x2 * A[2, 1, i] +
            x3 * A[3, 1, i]
        y[2, ialpha, i] =
            x1 * A[1, 2, i] +
            x2 * A[2, 2, i] +
            x3 * A[3, 2, i]
        y[3, ialpha, i] =
            x1 * A[1, 3, i] +
            x2 * A[2, 3, i] +
            x3 * A[3, 3, i]
    end
end

function jacckernel_mul_yxdagAdag_NC3!(i, y, x, A, NG)
    @inbounds for ialpha = 1:NG
        x1 = conj(x[1, ialpha, i])
        x2 = conj(x[2, ialpha, i])
        x3 = conj(x[3, ialpha, i])
        y[1, ialpha, i] =
            x1 * conj(A[1, 1, i]) +
            x2 * conj(A[1, 2, i]) +
            x3 * conj(A[1, 3, i])
        y[2, ialpha, i] =
            x1 * conj(A[2, 1, i]) +
            x2 * conj(A[2, 2, i]) +
            x3 * conj(A[2, 3, i])
        y[3, ialpha, i] =
            x1 * conj(A[3, 1, i]) +
            x2 * conj(A[3, 2, i]) +
            x3 * conj(A[3, 3, i])
    end
end

function jacckernel_mul_yxdagAdagshifted_NC3!(i, y, x, A, NG, shift, bc, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    #factor = kernel_calcfactor(i, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    ix_shifted = mod1(ix + shift[1], NX)
    iy_shifted = mod1(iy + shift[2], NY)
    iz_shifted = mod1(iz + shift[3], NZ)
    it_shifted = mod1(it + shift[4], NT)
    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)
    factor = jacckernel_calcfactor(i, shift, bc, NX, NY, NZ, NT)


    @inbounds for ialpha = 1:NG
        x1 = conj(x[1, ialpha, i_shifted]) * factor
        x2 = conj(x[2, ialpha, i_shifted]) * factor
        x3 = conj(x[3, ialpha, i_shifted]) * factor
        y[1, ialpha, i] =
            x1 * conj(A[1, 1, i]) +
            x2 * conj(A[1, 2, i]) +
            x3 * conj(A[1, 3, i])
        y[2, ialpha, i] =
            x1 * conj(A[2, 1, i]) +
            x2 * conj(A[2, 2, i]) +
            x3 * conj(A[2, 3, i])
        y[3, ialpha, i] =
            x1 * conj(A[3, 1, i]) +
            x2 * conj(A[3, 2, i]) +
            x3 * conj(A[3, 3, i])
    end
end

function jacckernel_mul_xA_NC!(i, xout, x, A, NC)
    @inbounds for ic = 1:NC
        e1 = x[ic, 1, i]
        e2 = x[ic, 2, i]
        e3 = x[ic, 3, i]
        e4 = x[ic, 4, i]

        xout[ic, 1, i] =
            A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
        xout[ic, 2, i] =
            A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
        xout[ic, 3, i] =
            A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
        xout[ic, 4, i] =
            A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
    end

end

function jacckernel_mul_xA_NC3!(i, xout, x, A)
    @inbounds for ic = 1:3
        e1 = x[ic, 1, i]
        e2 = x[ic, 2, i]
        e3 = x[ic, 3, i]
        e4 = x[ic, 4, i]

        xout[ic, 1, i] =
            A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
        xout[ic, 2, i] =
            A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
        xout[ic, 3, i] =
            A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
        xout[ic, 4, i] =
            A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
    end

end