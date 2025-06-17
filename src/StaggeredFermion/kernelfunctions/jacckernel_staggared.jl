
import Gaugefields.AbstractGaugefields_module:
    Gaugefields_4D_accelerator, Blockindices, Adjoint_Gaugefields, fourdim_cordinate, index_to_coords, coords_to_index



import Gaugefields.AbstractGaugefields_module: shiftedindex

function jacckernel_gauss_distribution_staggeredfermion!(i, x, σ, NC)
    #@inbounds for mu = 1:NG
    @inbounds for ic = 1:NC
        v1 = sqrt(-log(rand() + 1e-10))
        v2 = 2pi * rand()

        xr = v1 * cos(v2)
        xi = v1 * sin(v2)
        x[ic, i] = σ * xr + σ * im * xi
        #    end
    end

end

function jacckernel_clear_staggeredfermion!(i, a, NC)
    @inbounds for ic = 1:NC
        a[ic, i] = 0
    end
    #end
end

#kernel_add_fermion!(i, c.f, α, a.f, NC)
function jacckernel_add_staggeredfermion!(i, c, α, a, NC)
    @inbounds for ic = 1:NC
        c[ic, i] += α * a[ic, i]
    end
    #end
end

function jacckernel_add_staggeredfermion!(i, c, α, a, β, B, NC)
    @inbounds for ic = 1:NC
        c[ic, i] += α * a[ic, i] + β * B[ic, i]
    end
    #end
end

function jacckernel_mul_yUx_NC3_staggered!(i, y, A, x)

    x1 = x[1, i]
    x2 = x[2, i]
    x3 = x[3, i]

    y[1, i] =
        A[1, 1, i] * x1 +
        A[1, 2, i] * x2 +
        A[1, 3, i] * x3
    y[2, i] =
        A[2, 1, i] * x1 +
        A[2, 2, i] * x2 +
        A[2, 3, i] * x3
    y[3, i] =
        A[3, 1, i] * x1 +
        A[3, 2, i] * x2 +
        A[3, 3, i] * x3
    #end
end




function jacckernel_mul_yUdagx_NC3_staggered!(i, y, A, x)

    x1 = x[1, i]
    x2 = x[2, i]
    x3 = x[3, i]

    y[1, i] =
        conj(A[1, 1, i]) * x1 +
        conj(A[2, 1, i]) * x2 +
        conj(A[3, 1, i]) * x3
    y[2, i] =
        conj(A[1, 2, i]) * x1 +
        conj(A[2, 2, i]) * x2 +
        conj(A[3, 2, i]) * x3
    y[3, i] =
        conj(A[1, 3, i]) * x1 +
        conj(A[2, 3, i]) * x2 +
        conj(A[3, 3, i]) * x3
    #end
end


function jacckernel_calcfactor_staggered(i, shift, bc, NX, NY, NZ, NT)
    #ix, iy, iz, it = fourdim_cordinate(i, blockinfo)
    #bc = JACC.shared(bc_in)
    #shift = JACC.shared(shift_in)

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

@inline function jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)
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


    ix_shifted = mod1(ix_shifted, NX)
    iy_shifted = mod1(iy_shifted, NY)
    iz_shifted = mod1(iz_shifted, NZ)
    it_shifted = mod1(it_shifted, NT)


    i_shifted = coords_to_index(ix_shifted, iy_shifted, iz_shifted, it_shifted, NX, NY, NZ, NT)

    return factor_x * factor_y * factor_z * factor_t, i_shifted
end


function jacckernel_shifted_staggeredfermion!(i, f, fshifted, bc, shift, NC, NX, NY, NZ, NT)


    factor, i_shifted = jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)
    #bshifted, rshifted = shiftedindex(i, shift, blockinfo)

    for ic = 1:NC
        #fshifted[ic,   i] = factor_x *
        #                      factor_y *
        #                      factor_z *
        #                      factor_t *
        #                      f[ic,   i_shifted]
        fshifted[ic, i] = factor * f[ic, i_shifted]
    end
    #end

    return
end







function jacckernel_staggereddot!(i, A, B, NC)
    res = zero(eltype(A))
    #temp_volume[i] = 0

    for ic = 1:NC
        res += conj(A[ic, i]) * B[ic, i]
    end
    #end
    return res
end

@inbounds function jacckernel_substitute_staggeredfermion!(i, A, B, NC)
    #@inbounds #for α = 1:4
    for ic = 1:NC
        A[ic, i] = B[ic, i]
    end
    #end
end

@inbounds function jacckernel_axpby_staggered!(i, α, X, β, Y, NC)
    #for ig = 1:4
    for ic = 1:NC
        Y[ic, i] =
            α * X[ic, i] +
            β * Y[ic, i]
        #end
    end
end

@inbounds function jacckernel_axpby_NC3_staggered!(i, α, X, β, Y)
    #for ig = 1:4
    for ic = 1:3
        Y[ic, i] =
            α * X[ic, i] +
            β * Y[ic, i]
    end
    #end
end



@inbounds function jacckernel_mul_yAx_NC3_staggered!(i, y, A, x)
    #for ialpha = 1:4
    x1 = x[1, i]
    x2 = x[2, i]
    x3 = x[3, i]

    y[1, i] =
        A[1, 1, i] * x1 +
        A[1, 2, i] * x2 +
        A[1, 3, i] * x3
    y[2, i] =
        A[2, 1, i] * x1 +
        A[2, 2, i] * x2 +
        A[2, 3, i] * x3
    y[3, i] =
        A[3, 1, i] * x1 +
        A[3, 2, i] * x2 +
        A[3, 3, i] * x3
    # =#
    # end
end

@inbounds function jacckernel_mul_yAdagx_NC3_staggered!(i, y, A, x)
    #for ialpha = 1:4
    x1 = x[1, i]
    x2 = x[2, i]
    x3 = x[3, i]

    y[1, i] =
        conj(A[1, 1, i]) * x1 +
        conj(A[2, 1, i]) * x2 +
        conj(A[3, 1, i]) * x3
    y[2, i] =
        conj(A[1, 2, i]) * x1 +
        conj(A[2, 2, i]) * x2 +
        conj(A[3, 2, i]) * x3
    y[3, i] =
        conj(A[1, 3, i]) * x1 +
        conj(A[2, 3, i]) * x2 +
        conj(A[3, 3, i]) * x3
    # =#
    # end
end

@inbounds function jacckernel_mul_ystaggeredAx_NC3_staggered!(i,
    Y, A, X, NX, NY, NZ, NT, μ)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    t = it - 1
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[4],1)
    z = iz - 1
    z += ifelse(z < 0, NZ, 0)
    z += ifelse(z ≥ NZ, -NZ, 0)
    #boundary_factor_z = ifelse(z == NZ -1,BoundaryCondition[3],1)
    y = iy - 1
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = ix - 1
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    elseif μ == 4
        #η = (-1.0)^(x+y+z)
        η = ifelse((x + y + z) % 2 == 0, 1, -1)
    end

    #for ialpha = 1:4
    x1 = X[1, i] * η
    x2 = X[2, i] * η
    x3 = X[3, i] * η

    Y[1, i] =
        A[1, 1, i] * x1 +
        A[1, 2, i] * x2 +
        A[1, 3, i] * x3
    Y[2, i] =
        A[2, 1, i] * x1 +
        A[2, 2, i] * x2 +
        A[2, 3, i] * x3
    Y[3, i] =
        A[3, 1, i] * x1 +
        A[3, 2, i] * x2 +
        A[3, 3, i] * x3
    # =#
    # end
end


@inbounds function jacckernel_mul_ystaggeredAdagx_NC3_staggered!(i,
    Y, A, X, NX, NY, NZ, NT, μ)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    t = it - 1
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[4],1)
    z = iz - 1
    z += ifelse(z < 0, NZ, 0)
    z += ifelse(z ≥ NZ, -NZ, 0)
    #boundary_factor_z = ifelse(z == NZ -1,BoundaryCondition[3],1)
    y = iy - 1
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = ix - 1
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    elseif μ == 4
        #η = (-1.0)^(x+y+z)
        η = ifelse((x + y + z) % 2 == 0, 1, -1)
    end

    #for ialpha = 1:4
    x1 = X[1, i] * η
    x2 = X[2, i] * η
    x3 = X[3, i] * η

    Y[1, i] =
        conj(A[1, 1, i]) * x1 +
        conj(A[2, 1, i]) * x2 +
        conj(A[3, 1, i]) * x3
    Y[2, i] =
        conj(A[1, 2, i]) * x1 +
        conj(A[2, 2, i]) * x2 +
        conj(A[3, 2, i]) * x3
    Y[3, i] =
        conj(A[1, 3, i]) * x1 +
        conj(A[2, 3, i]) * x2 +
        conj(A[3, 3, i]) * x3
    # =#
    # end
end

@inbounds function jacckernel_mul_yxdagstaggeredAdagshifted_NC3_staggered!(i, Y, X, A, shift, bc, NX, NY, NZ, NT, μ)

    factor, i_shifted = jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)
    ix, iy, iz, it = index_to_coords(i, NX, NY, NZ, NT)
    t = it - 1
    t += ifelse(t < 0, NT, 0)
    t += ifelse(t ≥ NT, -NT, 0)
    #boundary_factor_t = ifelse(t == NT -1,BoundaryCondition[4],1)
    z = iz - 1
    z += ifelse(z < 0, NZ, 0)
    z += ifelse(z ≥ NZ, -NZ, 0)
    #boundary_factor_z = ifelse(z == NZ -1,BoundaryCondition[3],1)
    y = iy - 1
    y += ifelse(y < 0, NY, 0)
    y += ifelse(y ≥ NY, -NY, 0)
    #boundary_factor_y = ifelse(y == NY -1,BoundaryCondition[2],1)
    x = ix - 1
    x += ifelse(x < 0, NX, 0)
    x += ifelse(x ≥ NX, -NX, 0)
    #boundary_factor_x = ifelse(x == NX -1,BoundaryCondition[1],1)
    if μ == 1
        η = 1
    elseif μ == 2
        #η = (-1.0)^(x)
        η = ifelse(x % 2 == 0, 1, -1)
    elseif μ == 3
        #η = (-1.0)^(x+y)
        η = ifelse((x + y) % 2 == 0, 1, -1)
    elseif μ == 4
        #η = (-1.0)^(x+y+z)
        η = ifelse((x + y + z) % 2 == 0, 1, -1)
    end


    #for ialpha = 1:NG
    x1 = conj(X[1, i_shifted]) * factor * η
    x2 = conj(X[2, i_shifted]) * factor * η
    x3 = conj(X[3, i_shifted]) * factor * η
    Y[1, i] =
        x1 * conj(A[1, 1, i]) +
        x2 * conj(A[1, 2, i]) +
        x3 * conj(A[1, 3, i])
    Y[2, i] =
        x1 * conj(A[2, 1, i]) +
        x2 * conj(A[2, 2, i]) +
        x3 * conj(A[2, 3, i])
    Y[3, i] =
        x1 * conj(A[3, 1, i]) +
        x2 * conj(A[3, 2, i]) +
        x3 * conj(A[3, 3, i])
    #end
end


@inbounds function jacckernel_mul_yAx_NC3_shifted!(i, y, A, x, shift, bc, NX, NY, NZ, NT)

    factor, i_shifted = jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)




    #for ialpha = 1:4
    x1 = x[1, i_shifted] * factor
    x2 = x[2, i_shifted] * factor
    x3 = x[3, i_shifted] * factor

    y[1, i] =
        A[1, 1, i] * x1 +
        A[1, 2, i] * x2 +
        A[1, 3, i] * x3
    y[2, i] =
        A[2, 1, i] * x1 +
        A[2, 2, i] * x2 +
        A[2, 3, i] * x3
    y[3, i] =
        A[3, 1, i] * x1 +
        A[3, 2, i] * x2 +
        A[3, 3, i] * x3
    # =#
    #end
end


@inbounds function jacckernel_mul_ysx_NC_staggered!(i, y, A, x, NC)
    #for ialpha = 1:4
    for k1 = 1:NC
        y[k1, i] =
            A * x[k1, i]
    end
    #end
end

@inbounds function jacckernel_mul_ysx_NC3_staggered!(i, y, A, x)
    #for ialpha = 1:4
    for k1 = 1:3
        y[k1, i] =
            A * x[k1, i]
    end
    #end
end


@inbounds function jacckernel_mul_uxy_NC_staggered!(i, u, x, y, NC)
    #for ik = 1:NG
    for ib = 1:NC
        for ia = 1:NC
            c = x[ia, i] * y[ib, i]
            u[ia, ib, i] += c
        end
    end
    #end
end

@inbounds function jacckernel_mul_uxy_NC3_staggered!(i, u, x, y)
    #for ik = 1:4
    for ib = 1:3
        for ia = 1:3
            c = x[ia, i] * y[ib, i]
            u[ia, ib, i] += c
        end
    end
    #end
end

@inbounds function jacckernel_mul_uxydag_NC_staggered!(i, u, x, y, NC)
    #for ik = 1:NG
    for ib = 1:NC
        for ia = 1:NC
            c = x[ia, i] * conj(y[ib, i])
            u[ia, ib, i] += c
        end
    end
    #end
end

@inbounds function jacckernel_mul_uxydag_NC3_staggered!(i, u, x, y)
    #for ik = 1:4
    for ib = 1:3
        for ia = 1:3
            c = x[ia, i] * conj(y[ib, i])
            u[ia, ib, i] += c
        end
    end
    #end
end


@inbounds function jacckernel_mul_yxA_NC3_staggered!(i, y, x, A)
    #for ialpha = 1:NG
    x1 = x[1, i]
    x2 = x[2, i]
    x3 = x[3, i]
    y[1, i] =
        x1 * A[1, 1, i] +
        x2 * A[2, 1, i] +
        x3 * A[3, 1, i]
    y[2, i] =
        x1 * A[1, 2, i] +
        x2 * A[2, 2, i] +
        x3 * A[3, 2, i]
    y[3, i] =
        x1 * A[1, 3, i] +
        x2 * A[2, 3, i] +
        x3 * A[3, 3, i]
    #end
end

@inbounds function jacckernel_mul_yxdagAdag_NC3_staggered!(i, y, x, A)
    #for ialpha = 1:NG
    x1 = conj(x[1, i])
    x2 = conj(x[2, i])
    x3 = conj(x[3, i])
    y[1, i] =
        x1 * conj(A[1, 1, i]) +
        x2 * conj(A[1, 2, i]) +
        x3 * conj(A[1, 3, i])
    y[2, i] =
        x1 * conj(A[2, 1, i]) +
        x2 * conj(A[2, 2, i]) +
        x3 * conj(A[2, 3, i])
    y[3, i] =
        x1 * conj(A[3, 1, i]) +
        x2 * conj(A[3, 2, i]) +
        x3 * conj(A[3, 3, i])
    #end
end

@inbounds function jacckernel_mul_yxdagAdagshifted_NC3_staggered!(i, y, x, A, shift, bc, NX, NY, NZ, NT)

    factor, i_shifted = jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)



    #for ialpha = 1:NG
    x1 = conj(x[1, i_shifted]) * factor
    x2 = conj(x[2, i_shifted]) * factor
    x3 = conj(x[3, i_shifted]) * factor
    y[1, i] =
        x1 * conj(A[1, 1, i]) +
        x2 * conj(A[1, 2, i]) +
        x3 * conj(A[1, 3, i])
    y[2, i] =
        x1 * conj(A[2, 1, i]) +
        x2 * conj(A[2, 2, i]) +
        x3 * conj(A[2, 3, i])
    y[3, i] =
        x1 * conj(A[3, 1, i]) +
        x2 * conj(A[3, 2, i]) +
        x3 * conj(A[3, 3, i])
    #end
end


@inbounds function jacckernel_mul_yxAdagshifted_NC3_staggered!(i, y, x, A, shift, bc, NX, NY, NZ, NT)

    factor, i_shifted = jacckernel_calcfactor_and_index_staggered(i, shift, bc, NX, NY, NZ, NT)



    #for ialpha = 1:NG
    x1 = x[1, i_shifted] * factor
    x2 = x[2, i_shifted] * factor
    x3 = x[3, i_shifted] * factor
    y[1, i] =
        x1 * conj(A[1, 1, i]) +
        x2 * conj(A[1, 2, i]) +
        x3 * conj(A[1, 3, i])
    y[2, i] =
        x1 * conj(A[2, 1, i]) +
        x2 * conj(A[2, 2, i]) +
        x3 * conj(A[2, 3, i])
    y[3, i] =
        x1 * conj(A[3, 1, i]) +
        x2 * conj(A[3, 2, i]) +
        x3 * conj(A[3, 3, i])
    #end
end


