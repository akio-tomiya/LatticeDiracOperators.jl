# Four-dimensional even/odd reduction of the generalized domain-wall operator.
# D = [E B; C O], S = I - E^{-1} B O^{-1} C.  The fifth direction is local.
# Parity-restricted hopping implements B and C; raw D matvecs continue to
# act on both parities even when the solver uses EO.

function _validate_generalized_eo(x)
    all(iseven, (x.NX, x.NY, x.NZ, x.NT)) || throw(ArgumentError(
        "generalized domain-wall EO requires even global extents in all four physical directions"))
    x.f.phases[5] == 1 || throw(ArgumentError(
        "generalized domain-wall EO requires fifth-direction boundary phase 1"))
    return nothing
end

struct GeneralizedDomainwallEOCache{AT}
    lower::AT
    upper::AT
    lower_adjoint::AT
    upper_adjoint::AT
end

function _generalized_eo_cache(x, mass, M, a, b, c)
    L5 = x.L5
    # The on-site Wilson term is 4+M.  Spins 1:2 couple to s-1 and
    # spins 3:4 to s+1.  += also handles L5=1, where these entries coincide.
    lower = zeros(promote_type(ComplexF64, eltype(a)), L5, L5)
    upper = similar(lower)
    fill!(upper, 0)
    for s in 1:L5
        lower[s, s] = upper[s, s] = a[s] * (1 + (4 + M) * b[s])
        coupling = a[s] * ((4 + M) * c[s] - 1)
        lower[s, mod1(s - 1, L5)] += coupling * (s == 1 ? -mass : 1)
        upper[s, mod1(s + 1, L5)] += coupling * (s == L5 ? -mass : 1)
    end
    lower_inv, upper_inv = inv(lower), inv(upper)
    # Per-operator backend arrays avoid host arrays in JACC GPU kernels and
    # unbounded global caches.  These matrices do not depend on gauge links.
    matrices = map((lower_inv, upper_inv, Matrix(lower_inv'), Matrix(upper_inv'))) do matrix
        device = similar(x.f.A, eltype(x.f.A), size(matrix))
        copyto!(device, matrix)
        device
    end
    return GeneralizedDomainwallEOCache(matrices...)
end

function _with_generalized_eo_fields(f, pool, count)
    fields, tokens = get_temp(pool, count)
    try
        return f(fields)
    finally
        unused!(pool, tokens)
    end
end

function GeneralizedD5DW_clear_parity!(x::DomainwallFermion_5D_MPILattice, even::Bool;
    sethalo=true)
    LatticeMatrices.parallel_for_mutating!(x.f, prod(x.f.PN), kernel_GeneralizedD5DW_clear_parity!,
        x.f.A, x.f.indexer, x.f.coords, x.f.PN,
        Val(x.f.NC1), Val(x.f.NC2), Val(x.f.nw), even)
    sethalo && set_wing_fermion!(x)
    return x
end

GeneralizedD5DW_keep_parity!(x::DomainwallFermion_5D_MPILattice, even::Bool;
    sethalo=true) = GeneralizedD5DW_clear_parity!(x, !even; sethalo)

@inline function kernel_GeneralizedD5DW_clear_parity!(
    i, u, indexer, coords, PN, ::Val{NC}, ::Val{NG}, ::Val{nw}, even,
) where {NC,NG,nw}
    indices = delinearize(indexer, i, nw)
    parity = iseven(sum(ntuple(d -> indices[d] - nw + coords[d] * PN[d], 4)))
    if parity == even
        @inbounds for spin in 1:NG, color in 1:NC
            u[color, spin, indices...] = zero(eltype(u))
        end
    end
    return nothing
end

function _generalized_diag_solve!(y, A, b, even; adj=false, sethalo=true)
    A.use_eo || throw(ArgumentError("construct this operator with evenodd=true"))
    if y === b
        return _with_generalized_eo_fields(A._temporary_fermion_forCG, 1) do fields
            substitute_fermion!(fields[1], b)
            _generalized_diag_solve!(y, A, fields[1], even; adj, sethalo)
        end
    end
    cache = A.eo_cache
    lower = adj ? cache.lower_adjoint : cache.lower
    upper = adj ? cache.upper_adjoint : cache.upper
    LatticeMatrices.parallel_for_mutating!(y.f, prod(y.f.PN), kernel_GeneralizedD5DW_diag_solve_direct!,
        y.f.A, b.f.A, lower, upper, y.f.indexer, y.f.coords, y.f.PN,
        Val(y.f.NC1), Val(y.f.NC2), Val(y.f.nw), Val(A.L5), even)
    sethalo && set_wing_fermion!(y)
    return y
end

D5DW_diag_solve_block!(y, A::D5DW_GeneralizedDomainwall_operator_MPILattice, b,
    even::Bool; sethalo=true) = _generalized_diag_solve!(y, A, b, even; sethalo)
D5DW_diag_solve_adjoint_direct!(y, A::D5DW_GeneralizedDomainwall_operator_MPILattice,
    b, even::Bool; sethalo=true) = _generalized_diag_solve!(y, A, b, even; adj=true, sethalo)

@inline function kernel_GeneralizedD5DW_diag_solve_direct!(
    i, y, b, lower, upper, indexer, coords, PN,
    ::Val{NC}, ::Val{NG}, ::Val{nw}, ::Val{L5}, even,
) where {NC,NG,nw,L5}
    ix, iy, iz, it, i5 = delinearize(indexer, i, nw)
    parity = iseven(ix + iy + iz + it - 4nw +
        coords[1] * PN[1] + coords[2] * PN[2] + coords[3] * PN[3] + coords[4] * PN[4])
    @inbounds for spin in 1:NG, color in 1:NC
        value = zero(eltype(y))
        if parity == even
            matrix = ifelse(spin <= 2, lower, upper)
            for s in 1:L5
                value += matrix[i5 - nw, s] * b[color, spin, ix, iy, iz, it, s + nw]
            end
        end
        y[color, spin, ix, iy, iz, it, i5] = value
    end
    return nothing
end

@inline _generalized_eo_is_even(indices, coords, PN, nw) =
    iseven(sum(ntuple(d -> indices[d] - nw + coords[d] * PN[d], 4)))

@inline function _generalized_eo_zero_site!(y, indices, ::Val{NC}) where NC
    @inbounds for spin in 1:4, color in 1:NC
        y[color, spin, indices...] = zero(eltype(y))
    end
    return nothing
end

# Only four-dimensional neighbours contribute, so these helpers accept both
# input parities. As with the raw stencil, input and output must be distinct.
GeneralizedD5DW_offdiag_blockx_clean!(y, A, x, even::Bool; sethalo=true) =
    _generalized_offdiag!(y, A, x, even; sethalo)
GeneralizedD5DW_offdiag_adjoint_blockx_clean!(y, A, x, even::Bool; sethalo=true) =
    _generalized_offdiag!(y, A, x, even; adj=true, sethalo)

function _generalized_offdiag!(y, A, x, even; adj=false, sethalo=true)
    D = A.D
    LatticeMatrices._require_5d_halo(Val(x.f.nw))
    foreach(LatticeMatrices.ensure_halo!, D.U)
    U1, U2, U3, U4 = ntuple(mu -> D.U[mu].A, 4)
    if x.f.NC1 == 3
        _generalized_eo_hopping_halo!(x)
        if adj
            # Reuse each Wilson hopping result across fifth slices. The second
            # stage wraps physical fifth indices directly, needing no halo.
            _with_generalized_eo_fields(A._temporary_fermion_forCG, 1) do fields
                h = fields[1]
                LatticeMatrices.parallel_for_mutating!(h.f, prod(h.f.PN),
                    kernel_GeneralizedD5DW_adjoint_hopping3!,
                    h.f.A, U1, U2, U3, U4, x.f.A, h.f.indexer,
                    h.f.coords, h.f.PN, Val(h.f.nw), even)
                LatticeMatrices.parallel_for_mutating!(y.f, 12 * prod(y.f.PN),
                    kernel_GeneralizedD5DW_adjoint_hopping_combine3!,
                    y.f.A, h.f.A, D.mass, D.a, D.b, D.c, y.f.indexer,
                    y.f.coords, y.f.PN, Val(y.f.nw), Val(A.L5), even)
            end
        else
            LatticeMatrices.parallel_for_mutating!(y.f, prod(y.f.PN),
                kernel_GeneralizedD5DW_hopping3!, y.f.A, U1, U2, U3, U4,
                x.f.A, D.mass, D.a, D.b, D.c, y.f.indexer,
                y.f.coords, y.f.PN, Val(y.f.nw), Val(A.L5), even)
        end
    else
        # The generic-colour kernel has no separate hopping accumulator.
        # Project the source, then evaluate the full stencil only on the
        # opposite parity, where all on-site terms vanish.
        _with_generalized_eo_fields(A._temporary_fermion_forCG, 1) do fields
            source = fields[1]
            LatticeMatrices.parallel_for_mutating!(source.f, prod(source.f.PN),
                kernel_GeneralizedD5DW_project!, source.f.A, x.f.A,
                source.f.indexer, source.f.coords, source.f.PN,
                Val(source.f.NC1), Val(source.f.nw), !even)
            LatticeMatrices.ensure_halo!(source.f)
            LatticeMatrices.parallel_for_mutating!(y.f, prod(y.f.PN),
                kernel_GeneralizedD5DW_hopping_generic!, y.f.A,
                U1, U2, U3, U4, source.f.A, D.mass, D.wilson_params,
                D.a, D.b, D.c, y.f.indexer, y.f.coords, y.f.PN,
                Val(y.f.NC1), Val(y.f.nw), Val(A.L5), even, Val(adj))
        end
    end
    sethalo && set_wing_fermion!(y)
    return y
end

# The SU(3) kernels wrap the local fifth coordinate and read only one physical
# neighbour at a time. On one rank they need four-dimensional faces, without
# corners or fifth-direction ghosts. This partial update must NOT mark the
# full halo clean: a later public stencil may still need those other ghosts.
function _generalized_eo_hopping_halo!(x)
    LatticeMatrices.halo_is_dirty(x.f) || return nothing
    if get_nprocs(x) != 1
        LatticeMatrices.ensure_halo!(x.f)
        return nothing
    end
    kernel = LatticeMatrices._BoundMutatingKernel(
        kernel_GeneralizedD5DW_local_hopping_halo!,
        (x.f.A, x.f.indexer, x.f.PN, x.f.phases, Val(x.f.NC1), Val(x.f.nw)))
    JACC.parallel_for(prod(x.f.PN), kernel)
    return nothing
end

function kernel_GeneralizedD5DW_local_hopping_halo!(
    i, x, indexer, PN, phases, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(indexer, i, nw)
    @inbounds for mu in 1:4
        if indices[mu] == nw + 1
            ghost = ntuple(d -> d == mu ? nw + PN[mu] + 1 : indices[d], 5)
            for spin in 1:4, color in 1:NC
                x[color, spin, ghost...] = phases[mu] * x[color, spin, indices...]
            end
        elseif indices[mu] == nw + PN[mu]
            ghost = ntuple(d -> d == mu ? nw : indices[d], 5)
            phase = inv(phases[mu])
            for spin in 1:4, color in 1:NC
                x[color, spin, ghost...] = phase * x[color, spin, indices...]
            end
        end
    end
    return nothing
end

function kernel_GeneralizedD5DW_hopping3!(
    i, y, U1, U2, U3, U4, x, mass, a, b, c, indexer, coords, PN,
    ::Val{nw}, ::Val{L5}, even,
) where {nw,L5}
    indices = delinearize(indexer, i, nw)
    if _generalized_eo_is_even(indices, coords, PN, nw) != even
        return _generalized_eo_zero_site!(y, indices, Val(3))
    end
    s = indices[5] - nw
    ip, im = LatticeMatrices._domainwall_wrapped_fifth_indices(indices, Val(L5), Val(nw))
    qp = s == L5 ? -mass : one(mass)
    qm = s == 1 ? -mass : one(mass)
    @inbounds hopping = LatticeMatrices._domainwall_hopping_accumulator3(
        U1, U2, U3, U4, x, indices, im, ip, b[s], qm * c[s], qp * c[s], Val(-1))
    @inbounds for color in 1:3, spin in 1:4
        y[color, spin, indices...] = -a[s] * hopping[4 * (color - 1) + spin] / 2
    end
    return nothing
end

function kernel_GeneralizedD5DW_adjoint_hopping3!(
    i, y, U1, U2, U3, U4, x, indexer, coords, PN, ::Val{nw}, even,
) where nw
    indices = delinearize(indexer, i, nw)
    if _generalized_eo_is_even(indices, coords, PN, nw) != even
        return _generalized_eo_zero_site!(y, indices, Val(3))
    end
    hopping = LatticeMatrices._domainwall_wilson_hopping_accumulator3(
        U1, U2, U3, U4, x, indices, Val(1))
    @inbounds for color in 1:3, spin in 1:4
        y[color, spin, indices...] = -hopping[4 * (color - 1) + spin] / 2
    end
    return nothing
end

function kernel_GeneralizedD5DW_adjoint_hopping_combine3!(
    i, y, h, mass, a, b, c, indexer, coords, PN, ::Val{nw}, ::Val{L5}, even,
) where {nw,L5}
    color, spin, site = LatticeMatrices._domainwall_component_site(i)
    indices = delinearize(indexer, site, nw)
    if _generalized_eo_is_even(indices, coords, PN, nw) != even
        @inbounds y[color, spin, indices...] = zero(eltype(y))
        return nothing
    end
    s = indices[5] - nw
    ip, im = LatticeMatrices._domainwall_wrapped_fifth_indices(indices, Val(L5), Val(nw))
    shifted = spin <= 2 ? ip : im
    t = shifted[5] - nw
    boundary = (spin <= 2 ? s == L5 : s == 1) ? -mass : one(mass)
    @inbounds y[color, spin, indices...] = a[s] * b[s] * h[color, spin, indices...] +
        boundary * a[t] * c[t] * h[color, spin, shifted...]
    return nothing
end

function kernel_GeneralizedD5DW_project!(
    i, y, x, indexer, coords, PN, ::Val{NC}, ::Val{nw}, even,
) where {NC,nw}
    indices = delinearize(indexer, i, nw)
    keep = _generalized_eo_is_even(indices, coords, PN, nw) == even
    @inbounds for spin in 1:4, color in 1:NC
        y[color, spin, indices...] = keep ? x[color, spin, indices...] : zero(eltype(y))
    end
    return nothing
end

function kernel_GeneralizedD5DW_hopping_generic!(
    i, y, U1, U2, U3, U4, x, mass, wilson_params, a, b, c,
    indexer, coords, PN, ::Val{NC}, ::Val{nw}, ::Val{L5}, even, ::Val{adj},
) where {NC,nw,L5,adj}
    indices = delinearize(indexer, i, nw)
    if _generalized_eo_is_even(indices, coords, PN, nw) != even
        return _generalized_eo_zero_site!(y, indices, Val(NC))
    end
    kernel = adj ? LatticeMatrices.kernel_adjoint_D5DW_GeneralizedDomainwallOperator5D! :
        LatticeMatrices.kernel_D5DW_GeneralizedDomainwallOperator5D!
    kernel(i, y, U1, U2, U3, U4, mass, wilson_params, x,
        a, b, c, Val(NC), Val(nw), indexer, Val(L5))
    return nothing
end

function GeneralizedD5DW_blockx!(y, A, x, input_even::Bool, output_even::Bool)
    return _with_generalized_eo_fields(A._temporary_fermion_forCG, 1) do fields
        substitute_fermion!(fields[1], x)
        GeneralizedD5DW_keep_parity!(fields[1], input_even)
        mul!(y, A, fields[1])
        GeneralizedD5DW_keep_parity!(y, output_even)
    end
end

struct D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice{Dim,T} <: Dirac_operator{Dim}
    parent::T
    function D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(
        A::D5DW_GeneralizedDomainwall_operator_MPILattice{Dim},
    ) where {Dim}
        A.use_eo || throw(ArgumentError("construct this operator with evenodd=true"))
        return new{Dim,typeof(A)}(A)
    end
end
struct Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice{T} <: Adjoint_Dirac_operator
    parent::T
end
struct DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice{T} <: DdagD_operator
    dirac::T
end
Base.adjoint(A::D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice) =
    Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
Base.adjoint(A::Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice) = A.parent
get_temporaryvectors_forCG(A::D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice) =
    A.parent._temporary_fermion_forCG
get_temporaryvectors_forCG(A::Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice) =
    get_temporaryvectors_forCG(A.parent)
get_temporaryvectors_forCG(A::DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice) =
    get_temporaryvectors_forCG(A.dirac)

function _generalized_schur!(y, A, x; adj=false, sethalo=true)
    return _with_generalized_eo_fields(A._temporary_fermion_forCG, 2) do fields
        t1, t2 = fields
        if adj
            # S† = I - C† O^{-†} B† E^{-†}.
            D5DW_diag_solve_adjoint_direct!(t1, A, x, true; sethalo=false)
            GeneralizedD5DW_offdiag_adjoint_blockx_clean!(t2, A, t1, false; sethalo=false)
            D5DW_diag_solve_adjoint_direct!(t1, A, t2, false; sethalo=false)
            GeneralizedD5DW_offdiag_adjoint_blockx_clean!(t2, A, t1, true; sethalo=false)
        else
            GeneralizedD5DW_offdiag_blockx_clean!(t1, A, x, false; sethalo=false)
            D5DW_diag_solve_block!(t2, A, t1, false; sethalo=false)
            GeneralizedD5DW_offdiag_blockx_clean!(t1, A, t2, true; sethalo=false)
            D5DW_diag_solve_block!(t2, A, t1, true; sethalo=false)
        end
        # All stencil reads finish before this pointwise update, including
        # when y === x. No initial projection or final copy is necessary.
        LatticeMatrices.parallel_for_mutating!(y.f, prod(y.f.PN),
            kernel_GeneralizedD5DW_schur_finish!, y.f.A, x.f.A, t2.f.A,
            y.f.indexer, y.f.coords, y.f.PN, Val(y.f.NC1), Val(y.f.nw))
        sethalo && set_wing_fermion!(y)
        y
    end
end

function kernel_GeneralizedD5DW_schur_finish!(
    i, y, x, correction, indexer, coords, PN, ::Val{NC}, ::Val{nw},
) where {NC,nw}
    indices = delinearize(indexer, i, nw)
    even = _generalized_eo_is_even(indices, coords, PN, nw)
    @inbounds for spin in 1:4, color in 1:NC
        y[color, spin, indices...] = even ?
            x[color, spin, indices...] - correction[color, spin, indices...] : zero(eltype(y))
    end
    return nothing
end

LinearAlgebra.mul!(y::DomainwallFermion_5D_MPILattice,
    S::D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice,
    x::DomainwallFermion_5D_MPILattice) = _generalized_schur!(y, S.parent, x)
LinearAlgebra.mul!(y::DomainwallFermion_5D_MPILattice,
    S::Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice,
    x::DomainwallFermion_5D_MPILattice) = _generalized_schur!(y, S.parent.parent, x; adj=true)
function LinearAlgebra.mul!(y::DomainwallFermion_5D_MPILattice,
    Q::DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice,
    x::DomainwallFermion_5D_MPILattice)
    return _generalized_schur_normal!(y, Q, x)
end

function _generalized_schur_normal!(y, Q, x; sethalo=true)
    return _with_generalized_eo_fields(get_temporaryvectors_forCG(Q), 1) do fields
        # Support both S†S and SS†. Each hopping step ensures only the halo
        # it consumes; the first output needs no eager synchronization.
        _solver_mul!(fields[1], Q.dirac, x)
        _solver_mul!(y, Q.dirac', fields[1])
        sethalo && set_wing_fermion!(y)
        y
    end
end

const GeneralizedDomainwallEOKrylovOperator = Union{
    D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice,
    Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice,
    DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice,
}

# Krylov vector algebra reads only physical sites. Keep its output halo dirty
# until a stencil consumes it; public field operations retain their contract.
_solver_copy!(::GeneralizedDomainwallEOKrylovOperator, y, x) =
    LatticeMatrices.substitute!(y.f, x.f)
_solver_axpby!(::GeneralizedDomainwallEOKrylovOperator, a, x, b, y) =
    LinearAlgebra.axpby!(a, x.f, b, y.f)
_solver_finish!(::GeneralizedDomainwallEOKrylovOperator, x) =
    LatticeMatrices.ensure_halo!(x.f)
_solver_mul!(y, S::D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice, x) =
    _generalized_schur!(y, S.parent, x; sethalo=false)
_solver_mul!(y, S::Adjoint_D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice, x) =
    _generalized_schur!(y, S.parent.parent, x; adj=true, sethalo=false)
_solver_mul!(y, Q::DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice, x) =
    _generalized_schur_normal!(y, Q, x; sethalo=false)

# Generalized domain-wall CG owns its leases even if convergence fails. eps follows
# the existing solvers' convention: an absolute squared residual tolerance.
function cg(x, Q::Union{
    DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice,
    GeneralizedD5DWdagD5DW_Wilson_operator{<:D5DW_GeneralizedDomainwall_operator_MPILattice},
}, b;
    eps=1e-10, maxsteps=1000, verbose=Verbose_print(2))
    return _with_generalized_eo_fields(get_temporaryvectors_forCG(Q), 3) do fields
        r, p, q = fields
        _solver_mul!(q, Q, x)
        _solver_copy!(Q, r, b)
        _solver_axpby!(Q, -1, q, 1, r)
        _solver_copy!(Q, p, r)
        initial = residual = real(dot(r, r))
        for iteration in 0:maxsteps
            if residual < eps
                branch = iteration == 0 ? :initial_residual : :updated_residual
                _solver_finish!(Q, x)
                return SolverDiagnostics(:cg, iteration, residual, initial, eps, maxsteps, 0, branch)
            end
            iteration == maxsteps && break
            _solver_mul!(q, Q, p)
            denominator = real(dot(p, q))
            isfinite(denominator) && denominator > 0 || error("Generalized domain-wall CG breakdown: p†Qp=$denominator")
            alpha = residual / denominator
            _solver_axpby!(Q, alpha, p, 1, x)
            _solver_axpby!(Q, -alpha, q, 1, r)
            next_residual = real(dot(r, r))
            _solver_axpby!(Q, 1, r, next_residual / residual, p)
            residual = next_residual
            println_verbose_level3(verbose, "Generalized domain-wall CG step $(iteration + 1): residual²=$residual")
        end
        error("Generalized domain-wall CG did not converge in $maxsteps steps (residual²=$residual, target=$eps)")
    end
end

function solve_DdagD_even!(y, A::D5DW_GeneralizedDomainwall_operator_MPILattice, b;
    eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)
    return _with_generalized_eo_fields(A._temporary_fermion_forCG, 1) do fields
        rhs = fields[1]
        substitute_fermion!(rhs, b)
        GeneralizedD5DW_keep_parity!(rhs, true)
        clear_fermion!(y)
        S = D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
        result = cg(y, DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice(S), rhs;
            eps, maxsteps, verbose)
        GeneralizedD5DW_keep_parity!(y, true)
        result
    end
end

function _solve_generalized_eo!(x, A, b; adj=false, solver=Symbol(A.method_CG),
    eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)
    solver in (:bicg, :bicgstab, :cg) || throw(ArgumentError("unsupported EO solver: $solver"))
    return _with_generalized_eo_fields(A._temporary_fermion_forCG, 5) do fields
        t_odd, t_even, rhs_even, rhs_odd, source = fields
        # Keep the full source so that solve_DinvX!(b, A, b) is also safe.
        substitute_fermion!(source, b)
        diag! = adj ? D5DW_diag_solve_adjoint_direct! : D5DW_diag_solve_block!
        hop! = adj ? GeneralizedD5DW_offdiag_adjoint_blockx_clean! : GeneralizedD5DW_offdiag_blockx_clean!
        diag!(t_odd, A, source, false)
        hop!(t_even, A, t_odd, true; sethalo=false)
        substitute_fermion!(rhs_even, source)
        GeneralizedD5DW_keep_parity!(rhs_even, true; sethalo=false)
        add!(rhs_even, -1, t_even)
        if !adj
            diag!(t_even, A, rhs_even, true)
            substitute_fermion!(rhs_even, t_even)
        end
        S = D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)
        op = adj ? S' : S
        clear_fermion!(x)
        result = if solver == :bicg
            bicg(x, op, rhs_even; eps, maxsteps, verbose)
        elseif solver == :bicgstab
            bicgstab(x, op, rhs_even; eps, maxsteps, verbose)
        elseif adj
            # (S†)^{-1} rhs = S (S† S)^{-1} rhs.
            clear_fermion!(t_even)
            diagnostics = cg(t_even, DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice(S),
                rhs_even; eps, maxsteps, verbose)
            mul!(x, S, t_even)
            diagnostics
        else
            mul!(t_even, S', rhs_even)
            cg(x, DdagD_GeneralizedDomainwall_operator_evenodd_MPILattice(S),
                t_even; eps, maxsteps, verbose)
        end
        if adj
            # S† is right preconditioned: its unknown is E† x_even.
            diag!(t_even, A, x, true)
            substitute_fermion!(x, t_even)
        end
        hop!(t_odd, A, x, false; sethalo=false)
        substitute_fermion!(rhs_odd, source)
        GeneralizedD5DW_keep_parity!(rhs_odd, false; sethalo=false)
        add!(rhs_odd, -1, t_odd)
        diag!(t_odd, A, rhs_odd, false; sethalo=false)
        add!(x, 1, t_odd)
        set_wing_fermion!(x)
        result
    end
end

solve_DinvX_eo!(x, A::D5DW_GeneralizedDomainwall_operator_MPILattice, b; kwargs...) =
    _solve_generalized_eo!(x, A, b; kwargs...)
GeneralizedD5DW_solve_adjoint_eo!(x, A::D5DW_GeneralizedDomainwall_operator_MPILattice,
    b; kwargs...) = _solve_generalized_eo!(x, A, b; adj=true, kwargs...)

function _solve_generalized_full_cg!(y, A, b; adj=false)
    return _with_generalized_eo_fields(A._temporary_fermi, 2) do fields
        rhs, solution = fields
        if adj
            # CGNE for A† y=b: solve A†A z=b, then y=A z.
            substitute_fermion!(rhs, b)
        else
            # CGNR for A y=b: solve A†A y=A†b.
            mul!(rhs, A', b)
        end
        clear_fermion!(solution)
        Q = GeneralizedD5DWdagD5DW_Wilson_operator(A)
        diagnostics = cg(solution, Q, rhs;
            eps=A.eps_CG, maxsteps=A.MaxCGstep, verbose=A.verbose_print)
        if adj
            mul!(y, A, solution)
        else
            substitute_fermion!(y, solution)
        end
        set_wing_fermion!(y)
        diagnostics
    end
end

function solve_DinvX!(y::DomainwallFermion_5D_MPILattice,
    A::D5DW_GeneralizedDomainwall_operator_MPILattice, x::DomainwallFermion_5D_MPILattice)
    A.use_eo && return solve_DinvX_eo!(y, A, x)
    A.method_CG == "cg" && return _solve_generalized_full_cg!(y, A, x)
    return invoke(solve_DinvX!, Tuple{AbstractFermionfields,Dirac_operator,AbstractFermionfields}, y, A, x)
end
function solve_DinvX!(y::DomainwallFermion_5D_MPILattice,
    A::Adjoint_D5DW_GeneralizedDomainwall_operator_MPILattice, x::DomainwallFermion_5D_MPILattice)
    A.parent.use_eo && return GeneralizedD5DW_solve_adjoint_eo!(y, A.parent, x)
    A.parent.method_CG == "cg" && return _solve_generalized_full_cg!(y, A.parent, x; adj=true)
    return invoke(solve_DinvX!, Tuple{AbstractFermionfields,Adjoint_Dirac_operator,AbstractFermionfields}, y, A, x)
end

_generalized_domainwall_eo_enabled(D) =
    D.D5DW isa D5DW_GeneralizedDomainwall_operator_MPILattice && D.D5DW.use_eo

_generalized_domainwall_direct_solve(D) =
    D.D5DW isa D5DW_GeneralizedDomainwall_operator_MPILattice &&
    (D.D5DW.use_eo || D.D5DW.method_CG == "cg")

# The outer operator is still W = D(m) D(PV)^{-1} on the full field. Only
# its internal inversions change; the Schur pseudofermion action is separate.
function LinearAlgebra.mul!(y::DomainwallFermion_5D_MPILattice,
    W::GeneralizedDomainwall_Dirac_operator, x::DomainwallFermion_5D_MPILattice)
    if !_generalized_domainwall_direct_solve(W)
        return invoke(mul!, Tuple{AbstractFermionfields,GeneralizedDomainwall_Dirac_operator,
            AbstractFermionfields}, y, W, x)
    end
    return _with_generalized_eo_fields(W.D5DW_PV._temporary_fermi, 1) do fields
        solve_DinvX!(fields[1], W.D5DW_PV, x)
        mul!(y, W.D5DW, fields[1])
    end
end
function LinearAlgebra.mul!(y::DomainwallFermion_5D_MPILattice,
    Wdag::Adjoint_GeneralizedDomainwall_operator, x::DomainwallFermion_5D_MPILattice)
    W = Wdag.parent
    if !_generalized_domainwall_direct_solve(W)
        return invoke(mul!, Tuple{AbstractFermionfields,Adjoint_GeneralizedDomainwall_operator,
            AbstractFermionfields}, y, Wdag, x)
    end
    return _with_generalized_eo_fields(W.D5DW_PV._temporary_fermi, 1) do fields
        mul!(fields[1], W.D5DW', x)
        solve_DinvX!(y, W.D5DW_PV', fields[1])
        y
    end
end
function solve_DinvX!(y::DomainwallFermion_5D_MPILattice,
    W::GeneralizedDomainwall_Dirac_operator, x::DomainwallFermion_5D_MPILattice)
    if !_generalized_domainwall_direct_solve(W)
        return invoke(solve_DinvX!, Tuple{AbstractFermionfields,Dirac_operator,AbstractFermionfields}, y, W, x)
    end
    return _with_generalized_eo_fields(W.D5DW._temporary_fermi, 1) do fields
        diagnostics = solve_DinvX!(fields[1], W.D5DW, x)
        mul!(y, W.D5DW_PV, fields[1])
        diagnostics
    end
end
function solve_DinvX!(y::DomainwallFermion_5D_MPILattice,
    Wdag::Adjoint_GeneralizedDomainwall_operator, x::DomainwallFermion_5D_MPILattice)
    W = Wdag.parent
    if !_generalized_domainwall_direct_solve(W)
        return invoke(solve_DinvX!, Tuple{AbstractFermionfields,Adjoint_Dirac_operator,AbstractFermionfields},
            y, Wdag, x)
    end
    return _with_generalized_eo_fields(W.D5DW._temporary_fermi, 1) do fields
        mul!(fields[1], W.D5DW_PV', x)
        solve_DinvX!(y, W.D5DW', fields[1])
    end
end
