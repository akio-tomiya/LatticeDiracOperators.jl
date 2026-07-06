const CLOVER_DIRECTION_PAIRS = ((1,2),(1,3),(1,4),(2,3),(2,4),(3,4))

struct WilsonClover
    clover_coefficient::Float64
    internal_flags::Array{Bool,1}
    inn_table::Array{Int64,3}
    _ftmp_vectors::Array{Array{ComplexF64,3},1}
    _is1::Array{Int64,1}
    _is2::Array{Int64,1}
    Fmunu::Array{ComplexF64,4}
    sigma_munu::Array{ComplexF64,3}

    function WilsonClover(U::Array{<: AbstractGaugefields{NC,Dim},1},x,clover_coefficient) where {NC,Dim}
        _,_,NN... = size(U[1])
        NV = prod(NN)
        inn_table= zeros(Int64,NV,4,2)
        internal_flags = zeros(Bool,2)
        _ftmp_vectors = Array{Array{ComplexF64,3},1}(undef,6)
        for i=1:6
            _ftmp_vectors[i] = zeros(ComplexF64,NC,NV,4)
        end

        _is1 = zeros(Int64,NV)
        _is2 = zeros(Int64,NV)
        Fmunu = zeros(ComplexF64,NC,NC,NV,6)
        sigma_munu = clover_sigma_matrices()

        clover = new(clover_coefficient,internal_flags,inn_table,_ftmp_vectors,_is1,_is2,Fmunu,sigma_munu)
        update_clover!(clover,U)
        return clover
    end
end

function clover_sigma_matrices()
    gamma,_,_ = mk_gamma(1.0)
    sigma_munu = zeros(ComplexF64,4,4,6)
    for (ipair,(mu,nu)) in enumerate(CLOVER_DIRECTION_PAIRS)
        sigma_munu[:,:,ipair] .= 0.5 .* (gamma[:,:,mu] * gamma[:,:,nu] - gamma[:,:,nu] * gamma[:,:,mu])
    end
    return sigma_munu
end

@inline function _clover_shift(site::NTuple{4,Int},dir::Int,step::Int,dims::NTuple{4,Int})
    shifted = (site[1],site[2],site[3],site[4])
    value = mod(site[dir] - 1 + step,dims[dir]) + 1
    if dir == 1
        return (value,shifted[2],shifted[3],shifted[4])
    elseif dir == 2
        return (shifted[1],value,shifted[3],shifted[4])
    elseif dir == 3
        return (shifted[1],shifted[2],value,shifted[4])
    else
        return (shifted[1],shifted[2],shifted[3],value)
    end
end

@inline function _clover_linear_site(site::NTuple{4,Int},dims::NTuple{4,Int})
    ix,iy,iz,it = site
    NX,NY,NZ,_ = dims
    return (it-1)*NX*NY*NZ + (iz-1)*NX*NY + (iy-1)*NX + ix
end

function _clover_link_matrix(Udir,site::NTuple{4,Int})
    NC = Udir.NC
    ix,iy,iz,it = site
    mat = Matrix{ComplexF64}(undef,NC,NC)
    @inbounds for j=1:NC
        for i=1:NC
            mat[i,j] = Udir[i,j,ix,iy,iz,it]
        end
    end
    return mat
end

function _clover_plaq(U,site::NTuple{4,Int},mu::Int,nu::Int,dims::NTuple{4,Int})
    x = site
    xpmu = _clover_shift(x,mu,1,dims)
    xpnu = _clover_shift(x,nu,1,dims)
    xmmu = _clover_shift(x,mu,-1,dims)
    xmnu = _clover_shift(x,nu,-1,dims)
    xmmu_pnu = _clover_shift(xmmu,nu,1,dims)
    xmmu_mnu = _clover_shift(xmmu,nu,-1,dims)
    xpmu_mnu = _clover_shift(xpmu,nu,-1,dims)

    U_mu_x = _clover_link_matrix(U[mu],x)
    U_nu_x = _clover_link_matrix(U[nu],x)
    U_nu_xpmu = _clover_link_matrix(U[nu],xpmu)
    U_mu_xpnu = _clover_link_matrix(U[mu],xpnu)
    U_mu_xmmu = _clover_link_matrix(U[mu],xmmu)
    U_nu_xmmu = _clover_link_matrix(U[nu],xmmu)
    U_mu_xmnu = _clover_link_matrix(U[mu],xmnu)
    U_nu_xmnu = _clover_link_matrix(U[nu],xmnu)
    U_mu_xmmu_pnu = _clover_link_matrix(U[mu],xmmu_pnu)
    U_nu_xmmu_mnu = _clover_link_matrix(U[nu],xmmu_mnu)
    U_mu_xmmu_mnu = _clover_link_matrix(U[mu],xmmu_mnu)
    U_nu_xpmu_mnu = _clover_link_matrix(U[nu],xpmu_mnu)

    p1 = U_mu_x * U_nu_xpmu * U_mu_xpnu' * U_nu_x'
    p2 = U_nu_x * U_mu_xmmu_pnu' * U_nu_xmmu' * U_mu_xmmu
    p3 = U_mu_xmmu' * U_nu_xmmu_mnu' * U_mu_xmmu_mnu * U_nu_xmnu
    p4 = U_nu_xmnu' * U_mu_xmnu * U_nu_xpmu_mnu * U_mu_x'

    return p1 + p2 + p3 + p4
end

function _traceless_antihermitian_part(mat::AbstractMatrix{ComplexF64})
    NC = size(mat,1)
    fmat = 0.125 .* (mat - mat')
    trace_part = tr(fmat) / NC
    for i=1:NC
        fmat[i,i] -= trace_part
    end
    return fmat
end

function update_clover!(clover::WilsonClover,U::Array{<: AbstractGaugefields{NC,4},1}) where NC
    dims = (U[1].NX,U[1].NY,U[1].NZ,U[1].NT)
    for it=1:dims[4]
        for iz=1:dims[3]
            for iy=1:dims[2]
                for ix=1:dims[1]
                    site = (ix,iy,iz,it)
                    isite = _clover_linear_site(site,dims)
                    for (ipair,(mu,nu)) in enumerate(CLOVER_DIRECTION_PAIRS)
                        fmat = _traceless_antihermitian_part(_clover_plaq(U,site,mu,nu,dims))
                        @inbounds for j=1:NC
                            for i=1:NC
                                clover.Fmunu[i,j,isite,ipair] = fmat[i,j]
                            end
                        end
                    end
                end
            end
        end
    end
    return clover
end

function add_clover_term!(xout,A,x)
    clover = A.cloverterm
    clover === nothing && return xout
    coefficient = A.κ * clover.clover_coefficient
    iszero(coefficient) && return xout

    update_clover!(clover,A.U)

    NX = x.NX
    NY = x.NY
    NZ = x.NZ
    NT = x.NT
    NC = x.NC
    dims = (NX,NY,NZ,NT)

    for it=1:NT
        for iz=1:NZ
            for iy=1:NY
                for ix=1:NX
                    isite = _clover_linear_site((ix,iy,iz,it),dims)
                    for alpha=1:4
                        for color_out=1:NC
                            accum = zero(ComplexF64)
                            for ipair=1:6
                                for beta=1:4
                                    spin_factor = clover.sigma_munu[alpha,beta,ipair]
                                    if !iszero(spin_factor)
                                        for color_in=1:NC
                                            accum += clover.Fmunu[color_out,color_in,isite,ipair] *
                                                spin_factor * x[color_in,ix,iy,iz,it,beta]
                                        end
                                    end
                                end
                            end
                            xout[color_out,ix,iy,iz,it,alpha] += coefficient * accum
                        end
                    end
                end
            end
        end
    end
    return xout
end
