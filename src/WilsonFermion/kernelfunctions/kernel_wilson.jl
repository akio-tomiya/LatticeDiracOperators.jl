
import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate



import Gaugefields.AbstractGaugefields_module: shiftedindex

function kernel_gauss_distribution_fermion!(b, r, x, σ, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            v1 = sqrt(-log(rand() + 1e-10))
            v2 = 2pi * rand()

            xr = v1 * cos(v2)
            xi = v1 * sin(v2)
            x[ic, mu, b, r] = σ * xr + σ * im * xi
        end
    end

end

function kernel_clear_fermion!(b, r, a, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            a[ic, mu, b, r] = 0
        end
    end
end

#kernel_add_fermion!(b, r, c.f, α, a.f, NC, NG)
function kernel_add_fermion!(b, r, c, α, a, NC, NG)
    @inbounds for mu = 1:NG
        for ic = 1:NC
            c[ic, mu, b, r] += α * a[ic, mu, b, r]
        end
    end
end

function kernel_mul_yUx_NC3!(b, r, y, A, x, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, b, r]
        x2 = x[2, ialpha, b, r]
        x3 = x[3, ialpha, b, r]

        y[1, ialpha, b, r] =
            A[1, 1, b, r] * x1 +
            A[1, 2, b, r] * x2 +
            A[1, 3, b, r] * x3
        y[2, ialpha, b, r] =
            A[2, 1, b, r] * x1 +
            A[2, 2, b, r] * x2 +
            A[2, 3, b, r] * x3
        y[3, ialpha, b, r] =
            A[3, 1, b, r] * x1 +
            A[3, 2, b, r] * x2 +
            A[3, 3, b, r] * x3
    end
end


function kernel_mul_yUdagx_NC3!(b, r, y, A, x, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, b, r]
        x2 = x[2, ialpha, b, r]
        x3 = x[3, ialpha, b, r]

        y[1, ialpha, b, r] =
            conj(A[1, 1, b, r]) * x1 +
            conj(A[2, 1, b, r]) * x2 +
            conj(A[3, 1, b, r]) * x3
        y[2, ialpha, b, r] =
            conj(A[1, 2, b, r]) * x1 +
            conj(A[2, 2, b, r]) * x2 +
            conj(A[3, 2, b, r]) * x3
        y[3, ialpha, b, r] =
            conj(A[1, 3, b, r]) * x1 +
            conj(A[2, 3, b, r]) * x2 +
            conj(A[3, 3, b, r]) * x3
    end
end

function kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)
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

function kernel_shifted_fermion!(b, r, f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

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


    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    @inbounds for ig = 1:4
        for ic = 1:NC
            fshifted[ic, ig, b, r] = factor_x *
                                     factor_y *
                                     factor_z *
                                     factor_t *
                                     f[ic, ig, bshifted, rshifted]
        end
    end

    return
end

function kernel_mul_1plusγ5x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        y[ic, 1, b, r] = 0#-1*x[ic,ix,iy,iz,it,1]
        y[ic, 2, b, r] = 0#-1*x[ic,ix,iy,iz,it,2]
        y[ic, 3, b, r] = x[ic, 3, b, r]
        y[ic, 4, b, r] = x[ic, 4, b, r]
    end

end


function kernel_mul_1plusγ5x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        y[ic, 1, b, r] = 0#-1*x[ic,ix,iy,iz,it,1]
        y[ic, 2, b, r] = 0#-1*x[ic,ix,iy,iz,it,2]
        y[ic, 3, b, r] = x[ic, 3, bshifted, rshifted] * factor
        y[ic, 4, b, r] = x[ic, 4, bshifted, rshifted] * factor
    end

end


function kernel_mul_1plusγ1x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] - im * x[ic, 4, b, r]
        v2 = x[ic, 2, b, r] - im * x[ic, 3, b, r]
        v3 = x[ic, 3, b, r] + im * x[ic, 2, b, r]
        v4 = x[ic, 4, b, r] + im * x[ic, 1, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end


function kernel_mul_1plusγ1x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] - im * x[ic, 4, bshifted, rshifted] * factor
        v2 = x[ic, 2, bshifted, rshifted] - im * x[ic, 3, bshifted, rshifted] * factor
        v3 = x[ic, 3, bshifted, rshifted] + im * x[ic, 2, bshifted, rshifted] * factor
        v4 = x[ic, 4, bshifted, rshifted] + im * x[ic, 1, bshifted, rshifted] * factor
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end


function kernel_mul_1minusγ1x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] + im * x[ic, 4, b, r]
        v2 = x[ic, 2, b, r] + im * x[ic, 3, b, r]
        v3 = x[ic, 3, b, r] - im * x[ic, 2, b, r]
        v4 = x[ic, 4, b, r] - im * x[ic, 1, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1minusγ1x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] + im * x[ic, 4, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] + im * x[ic, 3, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] - im * x[ic, 2, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] - im * x[ic, 1, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end

function kernel_mul_1plusγ2x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] - x[ic, 4, b, r]
        v2 = x[ic, 2, b, r] + x[ic, 3, b, r]
        v3 = x[ic, 3, b, r] + x[ic, 2, b, r]
        v4 = x[ic, 4, b, r] - x[ic, 1, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end


function kernel_mul_1plusγ2x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)


    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] - x[ic, 4, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] + x[ic, 3, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] + x[ic, 2, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] - x[ic, 1, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end


function kernel_mul_1minusγ2x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] + x[ic, 4, b, r]
        v2 = x[ic, 2, b, r] - x[ic, 3, b, r]
        v3 = x[ic, 3, b, r] - x[ic, 2, b, r]
        v4 = x[ic, 4, b, r] + x[ic, 1, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1minusγ2x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] + x[ic, 4, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] - x[ic, 3, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] - x[ic, 2, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] + x[ic, 1, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end





function kernel_mul_1plusγ3x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] - im * x[ic, 3, b, r]
        v2 = x[ic, 2, b, r] + im * x[ic, 4, b, r]
        v3 = x[ic, 3, b, r] + im * x[ic, 1, b, r]
        v4 = x[ic, 4, b, r] - im * x[ic, 2, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1plusγ3x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] - im * x[ic, 3, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] + im * x[ic, 4, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] + im * x[ic, 1, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] - im * x[ic, 2, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end


function kernel_mul_1minusγ3x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] + im * x[ic, 3, b, r]
        v2 = x[ic, 2, b, r] - im * x[ic, 4, b, r]
        v3 = x[ic, 3, b, r] - im * x[ic, 1, b, r]
        v4 = x[ic, 4, b, r] + im * x[ic, 2, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1minusγ3x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] + im * x[ic, 3, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] - im * x[ic, 4, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] - im * x[ic, 1, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] + im * x[ic, 2, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end



function kernel_mul_1plusγ4x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] - x[ic, 3, b, r]
        v2 = x[ic, 2, b, r] - x[ic, 4, b, r]
        v3 = x[ic, 3, b, r] - x[ic, 1, b, r]
        v4 = x[ic, 4, b, r] - x[ic, 2, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1plusγ4x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] - x[ic, 3, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] - x[ic, 4, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] - x[ic, 1, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] - x[ic, 2, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end

function kernel_mul_1minusγ4x!(b, r, y, x, NC)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, b, r] + x[ic, 3, b, r]
        v2 = x[ic, 2, b, r] + x[ic, 4, b, r]
        v3 = x[ic, 3, b, r] + x[ic, 1, b, r]
        v4 = x[ic, 4, b, r] + x[ic, 2, b, r]
        y[ic, 1, b, r] = v1
        y[ic, 2, b, r] = v2
        y[ic, 3, b, r] = v3
        y[ic, 4, b, r] = v4
    end

end

function kernel_mul_1minusγ4x_shifted!(b, r, y, x, shift, blockinfo, NC, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ic = 1:NC
        v1 = x[ic, 1, bshifted, rshifted] + x[ic, 3, bshifted, rshifted]
        v2 = x[ic, 2, bshifted, rshifted] + x[ic, 4, bshifted, rshifted]
        v3 = x[ic, 3, bshifted, rshifted] + x[ic, 1, bshifted, rshifted]
        v4 = x[ic, 4, bshifted, rshifted] + x[ic, 2, bshifted, rshifted]
        y[ic, 1, b, r] = v1 * factor
        y[ic, 2, b, r] = v2 * factor
        y[ic, 3, b, r] = v3 * factor
        y[ic, 4, b, r] = v4 * factor
    end

end


function kernel_dot!(b, r, temp_volume, A, B, NC)
    temp_volume[b, r] = 0
    @inbounds for α = 1:4
        for ic = 1:NC
            temp_volume[b, r] += conj(A[ic, α, b, r]) * B[ic, α, b, r]
        end
    end
end

function kernel_substitute_fermion!(b, r, A, B, NC)
    @inbounds for α = 1:4
        for ic = 1:NC
            A[ic, α, b, r] = B[ic, α, b, r]
        end
    end
end

function kernel_axpby!(b, r, α, X, β, Y, NC)
    @inbounds for ig = 1:4
        for ic = 1:NC
            Y[ic, ig, b, r] =
                α * X[ic, ig, b, r] +
                β * Y[ic, ig, b, r]
        end
    end
end

function kernel_mul_Ax!(b, r, xout, A, x, NC)
    @inbounds for ic = 1:NC
        e1 = x[ic, 1, b, r]
        e2 = x[ic, 2, b, r]
        e3 = x[ic, 3, b, r]
        e4 = x[ic, 4, b, r]

        xout[ic, 1, b, r] =
            A[1, 1] * e1 + A[1, 2] * e2 + A[1, 3] * e3 + A[1, 4] * e4
        xout[ic, 2, b, r] =
            A[2, 1] * e1 + A[2, 2] * e2 + A[2, 3] * e3 + A[2, 4] * e4
        xout[ic, 3, b, r] =
            A[3, 1] * e1 + A[3, 2] * e2 + A[3, 3] * e3 + A[3, 4] * e4
        xout[ic, 4, b, r] =
            A[4, 1] * e1 + A[4, 2] * e2 + A[4, 3] * e3 + A[4, 4] * e4
    end
end

function kernel_mul_yAx_NC3!(b, r, y, A, x)
    @inbounds for ialpha = 1:4
        x1 = x[1, ialpha, b, r]
        x2 = x[2, ialpha, b, r]
        x3 = x[3, ialpha, b, r]

        y[1, ialpha, b, r] =
            A[1, 1, b, r] * x1 +
            A[1, 2, b, r] * x2 +
            A[1, 3, b, r] * x3
        y[2, ialpha, b, r] =
            A[2, 1, b, r] * x1 +
            A[2, 2, b, r] * x2 +
            A[2, 3, b, r] * x3
        y[3, ialpha, b, r] =
            A[3, 1, b, r] * x1 +
            A[3, 2, b, r] * x2 +
            A[3, 3, b, r] * x3
        # =#
    end
end

function kernel_mul_yAx_NC3_shifted!(b, r, y, A, x, shift, blockinfo, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)


    @inbounds for ialpha = 1:4
        x1 = x[1, ialpha, bshifted, rshifted] * factor
        x2 = x[2, ialpha, bshifted, rshifted] * factor
        x3 = x[3, ialpha, bshifted, rshifted] * factor

        y[1, ialpha, b, r] =
            A[1, 1, b, r] * x1 +
            A[1, 2, b, r] * x2 +
            A[1, 3, b, r] * x3
        y[2, ialpha, b, r] =
            A[2, 1, b, r] * x1 +
            A[2, 2, b, r] * x2 +
            A[2, 3, b, r] * x3
        y[3, ialpha, b, r] =
            A[3, 1, b, r] * x1 +
            A[3, 2, b, r] * x2 +
            A[3, 3, b, r] * x3
        # =#
    end
end


function kernel_mul_ysx_NC!(b, r, y, A, x, NC)
    @inbounds for ialpha = 1:4
        for k1 = 1:NC
            y[k1, ialpha, b, r] =
                A * x[k1, ialpha, b, r]
        end
    end
end

function kernel_mul_uxy_NC!(b, r, u, x, y, NC, NG)
    @inbounds for ik = 1:NG
        for ib = 1:NC
            for ia = 1:NC
                c = x[ia, ik, b, r] * y[ib, ik, b, r]
                u[ia, ib, b, r] += c
            end
        end
    end
end

function kernel_mul_uxydag_NC!(b, r, u, x, y, NC, NG)
    @inbounds for ik = 1:NG
        for ib = 1:NC
            for ia = 1:NC
                c = x[ia, ik, b, r] * conj(y[ib, ik, b, r])
                u[ia, ib, b, r] += c
            end
        end
    end
end

function kernel_mul_yxA_NC3!(b, r, y, x, A, NG)
    @inbounds for ialpha = 1:NG
        x1 = x[1, ialpha, b, r]
        x2 = x[2, ialpha, b, r]
        x3 = x[3, ialpha, b, r]
        y[1, ialpha, b, r] =
            x1 * A[1, 1, b, r] +
            x2 * A[2, 1, b, r] +
            x3 * A[3, 1, b, r]
        y[2, ialpha, b, r] =
            x1 * A[1, 2, b, r] +
            x2 * A[2, 2, b, r] +
            x3 * A[3, 2, b, r]
        y[3, ialpha, b, r] =
            x1 * A[1, 3, b, r] +
            x2 * A[2, 3, b, r] +
            x3 * A[3, 3, b, r]
    end
end

function kernel_mul_yxdagAdag_NC3!(b, r, y, x, A, NG)
    @inbounds for ialpha = 1:NG
        x1 = conj(x[1, ialpha, b, r])
        x2 = conj(x[2, ialpha, b, r])
        x3 = conj(x[3, ialpha, b, r])
        y[1, ialpha, b, r] =
            x1 * conj(A[1, 1, b, r]) +
            x2 * conj(A[1, 2, b, r]) +
            x3 * conj(A[1, 3, b, r])
        y[2, ialpha, b, r] =
            x1 * conj(A[2, 1, b, r]) +
            x2 * conj(A[2, 2, b, r]) +
            x3 * conj(A[2, 3, b, r])
        y[3, ialpha, b, r] =
            x1 * conj(A[3, 1, b, r]) +
            x2 * conj(A[3, 2, b, r]) +
            x3 * conj(A[3, 3, b, r])
    end
end

function kernel_mul_yxdagAdagshifted_NC3!(b, r, y, x, A, NG, shift, blockinfo, bc, NX, NY, NZ, NT)
    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)
    factor = kernel_calcfactor(b, r, shift, blockinfo, bc, NX, NY, NZ, NT)

    @inbounds for ialpha = 1:NG
        x1 = conj(x[1, ialpha, bshifted, rshifted]) * factor
        x2 = conj(x[2, ialpha, bshifted, rshifted]) * factor
        x3 = conj(x[3, ialpha, bshifted, rshifted]) * factor
        y[1, ialpha, b, r] =
            x1 * conj(A[1, 1, b, r]) +
            x2 * conj(A[1, 2, b, r]) +
            x3 * conj(A[1, 3, b, r])
        y[2, ialpha, b, r] =
            x1 * conj(A[2, 1, b, r]) +
            x2 * conj(A[2, 2, b, r]) +
            x3 * conj(A[2, 3, b, r])
        y[3, ialpha, b, r] =
            x1 * conj(A[3, 1, b, r]) +
            x2 * conj(A[3, 2, b, r]) +
            x3 * conj(A[3, 3, b, r])
    end
end

function kernel_mul_xA_NC!(b, r, xout, x, A, NC)
    @inbounds for ic = 1:NC
        e1 = x[ic, 1, b, r]
        e2 = x[ic, 2, b, r]
        e3 = x[ic, 3, b, r]
        e4 = x[ic, 4, b, r]

        xout[ic, 1, b, r] =
            A[1, 1] * e1 + A[2, 1] * e2 + A[3, 1] * e3 + A[4, 1] * e4
        xout[ic, 2, b, r] =
            A[1, 2] * e1 + A[2, 2] * e2 + A[3, 2] * e3 + A[4, 2] * e4
        xout[ic, 3, b, r] =
            A[1, 3] * e1 + A[2, 3] * e2 + A[3, 3] * e3 + A[4, 3] * e4
        xout[ic, 4, b, r] =
            A[1, 4] * e1 + A[2, 4] * e2 + A[3, 4] * e3 + A[4, 4] * e4
    end

end