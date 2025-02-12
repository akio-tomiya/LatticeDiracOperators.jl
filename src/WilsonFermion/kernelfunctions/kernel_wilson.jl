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