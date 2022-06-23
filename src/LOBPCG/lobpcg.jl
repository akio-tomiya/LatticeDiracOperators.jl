module LOBPCG
using LinearAlgebra

function eigenvalues_LOPBCG!(n, md, x, H, λ, eps, x_temp, itemax)
    gram_real!(n, md, x, x_temp)
    r = zero(x)
    p = zero(x)
    z = zeros(Float64, n, 3 * md)
    ztemp = zero(z)

    #for ii=1:md
    mul!(x, H, x_temp)
    #end
    xhx = x_temp' * x
    λ, v = eigen(xhx)
    mul!(x, x_temp, v)

    for ii = 1:md
        for k = 1:n
            r[k, ii] = x[k, ii] * λ[ii]
        end
    end

    #for ii=1:md
    mul!(x_temp, H, x)
    #end
    r .= x_temp .- r

    for ite = 1:itemax
        for ii = 1:md
            for k = 1:n
                z[k, ii] = x[k, ii]
                z[k, md+ii] = r[k, ii]
                x[k, 2*md+ii] = p[k, ii]
            end
        end
        ztemp .= z

        nz = ifelse(ite == 1, 2md, 3md)
        gram_real!(n, nz, x, x_temp)

    end
end

function gram_real!(m, n, mat_v, mat_v_out)
    mat_v_out .= mat_v
    viold = zeros(Float64, m)
    vi = zeros(Float64, m)
    for i = 1:n
        for k = 1:m
            viold[k] = mat_v_out[k, i]
        end
        for j = 1:(i-1)
            nai = 0
            for k = 1:m
                nai += mat_v_out[k, j] * viold[m]
            end
            for k = 1:m
                vi[k] = viold[k] - nai * mat_v_out[k, j]
            end
            viold .= vi
        end
        nor = sqrt(dot(viold, viold))
        for k = 1:m
            mat_v_out[k, j] = viold[k] / nor
        end
    end
    return
end
end
