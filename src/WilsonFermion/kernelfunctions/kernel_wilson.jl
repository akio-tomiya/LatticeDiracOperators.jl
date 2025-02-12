include("linearalgebra_mul.jl")

function kernel_gauss_distribution_fermion!(b, r, x,
    σ, NC, NG)


    for mu = 1:NG
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
    for mu = 1:NG
        for ic = 1:NC
            a[ic, mu, b, r] = 0
        end
    end
end

function kernel_add_fermion!(b, r, c, α, a, NC, NG)
    for mu = 1:NG
        for ic = 1:NC
            c[ic, mu, b, r] += α * a[ic, mu, b, r]
        end
    end
end

function kernel_mul_yUx_NC3!(b, r, y, A, x, NG)
    for ialpha = 1:NG
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
    for ialpha = 1:NG
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

function kernel_shifted_fermion!(b, r, f, fshifted, blockinfo, bc, shift, NC, NX, NY, NZ, NT)
    ix, iy, iz, it = fourdim_cordinate(b, r, blockinfo)

    #inside_up = it_shifted > NT
    #inside_down = it_shifted < 1
    it_shifted += shift[4]
    iz_shifted += shift[3]
    iy_shifted += shift[2]
    ix_shifted += shift[1]
    factor_t = ifelse(it_shifted > NT || it_shifted < 1, bc[4], 1)
    factor_z = ifelse(iz_shifted > NZ || iz_shifted < 1, bc[3], 1)
    factor_y = ifelse(iy_shifted > NY || iy_shifted < 1, bc[2], 1)
    factor_x = ifelse(ix_shifted > NX || ix_shifted < 1, bc[1], 1)


    bshifted, rshifted = shiftedindex(b, r, shift, blockinfo)

    for ig = 1:4
        for ic = 1:NC
            fshifted[ic, ig, b, r] = factor_x *
                                     factor_y *
                                     factor_z *
                                     factor_t *
                                     f[ic, ig, bshifted, rshifted]
        end
    end
end

function kernel_mul_1plusγ5x!(b, r, y, x, NC)

    for ic = 1:NC
        y[ic, 1, b, r] = 0#-1*x[ic,ix,iy,iz,it,1]
        y[ic, 2, b, r] = 0#-1*x[ic,ix,iy,iz,it,2]
        y[ic, 3, b, r] = x[ic, 3, b, r]
        y[ic, 4, b, r] = x[ic, 4, b, r]
    end

end


function kernel_mul_1plusγ1x!(b, r, y, x, NC)

    for ic = 1:NC
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

function kernel_mul_1minusγ1x!(b, r, y, x, NC)

    for ic = 1:NC
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



function kernel_mul_1plusγ2x!(b, r, y, x, NC)

    for ic = 1:NC
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

function kernel_mul_1minusγ2x!(b, r, y, x, NC)

    for ic = 1:NC
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



function kernel_mul_1plusγ3x!(b, r, y, x, NC)

    for ic = 1:NC
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

function kernel_mul_1minusγ3x!(b, r, y, x, NC)

    for ic = 1:NC
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


function kernel_mul_1plusγ4x!(b, r, y, x, NC)

    for ic = 1:NC
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

function kernel_mul_1minusγ4x!(b, r, y, x, NC)

    for ic = 1:NC
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