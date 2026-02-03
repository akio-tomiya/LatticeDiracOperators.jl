#=
#=

function LatticeDiracOperators.dSFdU!(dfdU, GD::T, φ; numtemp=5) where {T<:General_Dirac_operator}
    D = GD.diracop
    U = D.U
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    #dfdU, itdfdUtemp = get_block(D.temps, 4)
    dfdU1 = dfdU[1]
    dfdU2 = dfdU[2]
    dfdU3 = dfdU[3]
    dfdU4 = dfdU[4]

    DdagD = DgagD_General_Dirac_operator(GD)
    #DdagD = DdagDOp(D)
    phitemp1, itphitemp1 = get_block(D.phitemps)
    η = phitemp1

    solve_DinvX!(η, DdagD, φ)
    #solve!(η, DdagD, φ) #η = (DdagD)^-1 φ
    println("solved")
    set_wing_fermion!(η)
    phitemp2, itphitemp2 = get_block(D.phitemps)
    χ = phitemp2
    mul!(χ, D, η)

    #phitemp1, itphitemp1 = get_block(D.phitemps)
    func(U1, U2, U3, U4, χ, η, apply, phitemp, temp) = g(χ, U1, U2, U3, U4, η, D.p, apply, phitemp, temp)

    temp, ittemp = get_block(D.temps, numtemp)
    phitemp, itphitemp = get_block(D.phitemps, numtemp)
    dtemp, itdtemp = get_block(D.temps, numtemp)
    dphitemp, itdphitemp = get_block(D.phitemps, numtemp)

    Enzyme_derivative!(
        func,
        U1,
        U2,
        U3,
        U4,
        dfdU1,
        dfdU2,
        dfdU3,
        dfdU4,
        nodiff(χ), nodiff(η), nodiff(D.apply); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

    #for μ = 1:4
    #    mul!(dfdU[μ], -2)
    #end

    unused!(D.temps, ittemp)
    unused!(D.temps, itdtemp)
    unused!(D.phitemps, itphitemp)
    unused!(D.phitemps, itdphitemp)
    unused!(D.phitemps, itphitemp1)
    unused!(D.phitemps, itphitemp2)



end

=#

# === EnzymeRules for WilsonFermion/Gaugefields wrappers ===

@inline function _lm_primal(x)
    if x isa Base.RefValue
        return _lm_primal(x[])
    end
    if hasfield(typeof(x), :f)
        return getfield(x, :f)
    elseif hasfield(typeof(x), :U)
        return getfield(x, :U)
    elseif hasfield(typeof(x), :val)
        return _lm_primal(getfield(x, :val))
    elseif hasproperty(x, :val)
        return _lm_primal(getproperty(x, :val))
    else
        return x
    end
end

@inline _lm_primal(x::Base.RefValue) = _lm_primal(x[])
@inline _lm_primal(x::WilsonFermion_4D_MPILattice) = x.f
@inline _lm_primal(x::Gaugefields_4D_MPILattice) = x.U

@inline function _shadow(x::Base.RefValue)
    xval = x[]
    xval isa Type && return nothing
    return _shadow(xval)
end
@inline _shadow(x::LatticeMatrix) = x
@inline _shadow(x::Shifted_Lattice) = x
@inline _shadow(x::Adjoint_Lattice) = x
@inline _shadow(x::WilsonFermion_4D_MPILattice) = x.f
@inline _shadow(x::Gaugefields_4D_MPILattice) = x.U

@inline function _shadow(x)
    if x isa Base.RefValue
        return _shadow(x[])
    end
    if hasfield(typeof(x), :f)
        return getfield(x, :f)
    elseif hasfield(typeof(x), :U)
        return getfield(x, :U)
    elseif hasfield(typeof(x), :val)
        return _shadow(getfield(x, :val))
    elseif hasproperty(x, :val)
        return _shadow(getproperty(x, :val))
    else
        return nothing
    end
end

@inline _lm_primal(x::ER.Annotation) = _lm_primal(x.val)
@inline _lm_primal(x::ER.Const) = _lm_primal(x.val)
@inline _shadow(x::ER.Annotation) = _shadow(x.val)
@inline _shadow(x::ER.Const) = _shadow(x.val)
@inline _lm_primal(x::LatticeMatrices.NoDiffArg) = _lm_primal(x.val)
@inline _lm_primal(x::LatticeMatrices.DiffArg) = _lm_primal(x.val)
@inline _shadow(x::LatticeMatrices.NoDiffArg) = _shadow(x.val)
@inline _shadow(x::LatticeMatrices.DiffArg) = _shadow(x.val)

@inline _shadow_data(x) = _shadow(x)
@inline _shadow_data(x::Base.RefValue) = _shadow_data(x[])
@inline _shadow_data(x::Shifted_Lattice) = _shadow_data(getfield(x, :data))
@inline _shadow_data(x::Adjoint_Lattice) = _shadow_data(getfield(x, :data))

@inline function _shadow_out(dCout, C::Annotation)
    if dCout isa Active
        return _shadow(dCout.val)
    elseif dCout isa Base.RefValue
        return dCout[]
    elseif dCout === nothing
        return _shadow(C.dval)
    else
        return dCout
    end
end

@inline _should_zero_dC(dCout) = dCout !== nothing

@inline function _zero_shadow!(C::LatticeMatrix)
    JACC.parallel_for(
        prod(C.PN), kernel_clear_4D!, C.A, C.indexer, Val(C.NC1), Val(C.NC2), Val(C.nw)
    )
    return nothing
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(clear_fermion!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {RT}
    clear_matrix!(_lm_primal(C.val))
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(clear_fermion!)},
    dCout, _tape,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing,)
    _zero_shadow!(dC_struct)
    return (nothing,)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_fermion!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    α::S,
    A::ER.Annotation,
) where {RT,S}
    αval = hasproperty(α, :val) ? α.val : α
    add_matrix!(_lm_primal(C.val), _lm_primal(A.val), αval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_fermion!)},
    dCout, _tape,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    α::S,
    A::ER.Annotation,
) where {S}
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _shadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        JACC.parallel_for(
            prod(dC_struct.PN),
            kernel_add_4D!,
            dAval, dCval, dC_struct.indexer,
            Val(dC_struct.NC1), Val(dC_struct.NC2),
            conj(αval), Val(dC_struct.nw)
        )
    end
    return (nothing, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_fermion!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    α::S,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    β::S2,
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {RT,S,S2}
    αval = hasproperty(α, :val) ? α.val : α
    βval = hasproperty(β, :val) ? β.val : β
    add_matrix!(_lm_primal(C.val), _lm_primal(A.val), _lm_primal(B.val), αval, βval)
    return ER.AugmentedReturn(nothing, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(add_fermion!)},
    dCout, _tape,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    α::S,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    β::S2,
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {S,S2}
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _shadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dB_struct = _shadow(B.dval)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    if dAval !== nothing
        αval = hasproperty(α, :val) ? α.val : α
        JACC.parallel_for(
            prod(dC_struct.PN),
            kernel_add_4D!,
            dAval, dCval, dC_struct.indexer,
            Val(dC_struct.NC1), Val(dC_struct.NC2),
            conj(αval), Val(dC_struct.nw)
        )
    end
    if dBval !== nothing
        βval = hasproperty(β, :val) ? β.val : β
        JACC.parallel_for(
            prod(dC_struct.PN),
            kernel_add_4D!,
            dBval, dCval, dC_struct.indexer,
            Val(dC_struct.NC1), Val(dC_struct.NC2),
            conj(βval), Val(dC_struct.nw)
        )
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation{<:AbstractMatrix},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {RT}
    mul!(_lm_primal(C.val), A.val, _lm_primal(B.val))
    return ER.AugmentedReturn(nothing, nothing, A.val)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dCout, tapeA,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation{<:AbstractMatrix},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dB_struct = _shadow(B.dval)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing
    dBval === nothing && return (nothing, nothing, nothing)

    Aval = (tapeA === nothing) ? A.val : tapeA
    At = JACC.array(Aval)

    C_lm = _lm_primal(C.val)
    NC1 = Val(C_lm.NC1)
    NC2 = Val(C_lm.NC2)
    NC3 = Val(size(Aval, 1))
    nw = Val(C_lm.nw)
    idxr = C_lm.indexer
    Nsites = prod(C_lm.PN)

    JACC.parallel_for(
        Nsites, kernel_Dmatrix_mulAdagBadd_matrix!, dBval, At, dCval, NC1, NC2, NC3, nw, idxr
    )

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:AbstractMatrix},
) where {RT}
    mul!(_lm_primal(C.val), _lm_primal(A.val), B.val)
    return ER.AugmentedReturn(nothing, nothing, B.val)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.mul!)},
    dCout, tapeB,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:AbstractMatrix},
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _shadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dAval === nothing && return (nothing, nothing, nothing)

    Bval = (tapeB === nothing) ? B.val : tapeB
    Bt = JACC.array(Bval)

    A_lm = _lm_primal(A.val)
    NC1 = Val(A_lm.NC1)
    NC2 = Val(A_lm.NC2)
    NC3 = Val(size(Bval, 2))
    nw = Val(A_lm.nw)
    idxr = A_lm.indexer
    Nsites = prod(A_lm.PN)

    JACC.parallel_for(
        Nsites, kernel_Dmatrix_mulACadd_matrix!, dAval, dCval, Bt, NC1, NC2, NC3, nw, idxr
    )

    _should_zero_dC(dCout) && _zero_shadow!(dC_struct)
    return (nothing, nothing, nothing)
end

@inline function kernel_Dmatrix_mul_dA_from_dC_Bdag_shift!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = conj(B[kc, jc, indices_p...])
            for ic = 1:NC1
                dA[ic, kc, indices...] += dC[ic, jc, indices...] * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shift!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shift) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_p = shiftindices(indices, shift)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_p...] += acc
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mul_dA_from_dC_Bdag_shiftAshiftB_scatter!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = conj(B[kc, jc, indices_B...])
            for ic = 1:NC1
                dA[ic, kc, indices_A...] += dC[ic, jc, indices...] * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mul_dAdag_from_dC_B_shiftAshiftB_scatter!(
    i, dA, dC, B,
    ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB
) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for kc = 1:NC3
        for jc = 1:NC2
            b = B[kc, jc, indices_B...]
            for ic = 1:NC1
                dA[kc, ic, indices_A...] += conj(dC[ic, jc, indices...]) * b
            end
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc, indices_A...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_B...] += acc
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB_adagA!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer, shiftA, shiftB) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    indices_A = shiftindices(indices, shiftA)
    indices_B = shiftindices(indices, shiftB)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[kc, ic, indices_A...]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices_B...] += acc
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mulACadd_matrix!(i, dA, dC, B, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for kc = 1:NC3
        for ic = 1:NC1
            acc = zero(eltype(dA))
            for jc = 1:NC2
                acc += dC[ic, jc, indices...] * B[jc, kc]
            end
            dA[ic, kc, indices...] += acc
        end
    end
    return nothing
end

@inline function kernel_Dmatrix_mulAdagBadd_matrix!(i, dB, A, dC, ::Val{NC1}, ::Val{NC2}, ::Val{NC3}, ::Val{nw}, dindexer) where {NC1,NC2,NC3,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for jc = 1:NC2
        for kc = 1:NC3
            acc = zero(eltype(dB))
            for ic = 1:NC1
                acc += conj(A[ic, kc]) * dC[ic, jc, indices...]
            end
            dB[kc, jc, indices...] += acc
        end
    end
    return nothing
end

@inline function kernel_dot_WF!(i, A, B, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    s = zero(eltype(A))
    @inbounds for ialpha = 1:NC2
        for ic = 1:NC1
            s += conj(A[ic, ialpha, indices...]) * B[ic, ialpha, indices...]
        end
    end
    return s
end

@inline function kernel_dot_grad_A!(i, dA, B, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dout) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ialpha = 1:NC2
        for ic = 1:NC1
            dA[ic, ialpha, indices...] += conj(dout) * B[ic, ialpha, indices...]
        end
    end
    return nothing
end

@inline function kernel_dot_grad_B!(i, dB, A, dindexer, ::Val{NC1}, ::Val{NC2}, ::Val{nw}, dout) where {NC1,NC2,nw}
    indices = delinearize(dindexer, i, nw)
    @inbounds for ialpha = 1:NC2
        for ic = 1:NC1
            dB[ic, ialpha, indices...] += dout * A[ic, ialpha, indices...]
        end
    end
    return nothing
end

@inline function _get_dout(dCout)
    if dCout isa Active
        return dCout.val
    elseif dCout isa Base.RefValue
        return dCout[]
    elseif dCout === nothing
        return nothing
    else
        return dCout
    end
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    ::Type{RT},
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {RT}
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    s = dot(A_lm, B_lm)
    return ER.AugmentedReturn(s, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    ::Type{RT},
    A::ER.Const{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
) where {RT}
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    s = dot(A_lm, B_lm)
    return ER.AugmentedReturn(s, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    ::Type{RT},
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Const{<:WilsonFermion_4D_MPILattice},
) where {RT}
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    s = dot(A_lm, B_lm)
    return ER.AugmentedReturn(s, nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    dCout, _tape,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    dout = _get_dout(dCout)
    dout === nothing && return (nothing, nothing)

    A_lm = _lm_primal(A.val)
    A_lm = (A_lm isa LatticeMatrix) ? A_lm : _lm_primal(A_lm)
    B_lm = _lm_primal(B.val)
    B_lm = (B_lm isa LatticeMatrix) ? B_lm : _lm_primal(B_lm)
    dA_struct = _shadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dB_struct = _shadow(B.dval)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing

    Nsites = prod(A_lm.PN)
    if dAval !== nothing
        JACC.parallel_for(
            Nsites, kernel_dot_grad_A!,
            dAval, B_lm.A, A_lm.indexer, Val(A_lm.NC1), Val(A_lm.NC2), Val(A_lm.nw), dout
        )
    end
    if dBval !== nothing
        JACC.parallel_for(
            Nsites, kernel_dot_grad_B!,
            dBval, A_lm.A, A_lm.indexer, Val(A_lm.NC1), Val(A_lm.NC2), Val(A_lm.nw), dout
        )
    end
    return (nothing, nothing)
end

function _reverse_dot_constA_annB(cfg::ER.RevConfig,
    dCout, _tape,
    A::ER.Const{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    dout = _get_dout(dCout)
    dout === nothing && return (nothing, nothing)

    B_lm = _lm_primal(B.val)
    B_lm = (B_lm isa LatticeMatrix) ? B_lm : _lm_primal(B_lm)
    dB_struct = _shadow(B.dval)
    dBval = (dB_struct isa LatticeMatrix) ? dB_struct.A : nothing
    dBval === nothing && return (nothing, nothing)

    Nsites = prod(B_lm.PN)
    JACC.parallel_for(
        Nsites, kernel_dot_grad_B!,
        dBval, _lm_primal(A.val).A, B_lm.indexer, Val(B_lm.NC1), Val(B_lm.NC2), Val(B_lm.nw), dout
    )
    return (nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    dCout, _tape,
    A::ER.Const{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    return _reverse_dot_constA_annB(cfg, dCout, _tape, A, B)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    dCout::ER.Active, _tape,
    A::ER.Const{<:WilsonFermion_4D_MPILattice},
    B::ER.Annotation{<:WilsonFermion_4D_MPILattice},
)
    return _reverse_dot_constA_annB(cfg, dCout, _tape, A, B)
end

function _reverse_dot_annA_constB(cfg::ER.RevConfig,
    dCout, _tape,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Const{<:WilsonFermion_4D_MPILattice},
)
    dout = _get_dout(dCout)
    dout === nothing && return (nothing, nothing)

    A_lm = _lm_primal(A.val)
    A_lm = (A_lm isa LatticeMatrix) ? A_lm : _lm_primal(A_lm)
    dA_struct = _shadow(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing
    dAval === nothing && return (nothing, nothing)

    Nsites = prod(A_lm.PN)
    JACC.parallel_for(
        Nsites, kernel_dot_grad_A!,
        dAval, _lm_primal(B.val).A, A_lm.indexer, Val(A_lm.NC1), Val(A_lm.NC2), Val(A_lm.nw), dout
    )
    return (nothing, nothing)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    dCout, _tape,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Const{<:WilsonFermion_4D_MPILattice},
)
    return _reverse_dot_annA_constB(cfg, dCout, _tape, A, B)
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(LinearAlgebra.dot)},
    dCout::ER.Active, _tape,
    A::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    B::ER.Const{<:WilsonFermion_4D_MPILattice},
)
    return _reverse_dot_annA_constB(cfg, dCout, _tape, A, B)
end


function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation,
    B::ER.Annotation,
    shift::RT2,
) where {RT,RT2}
    shift_val = hasproperty(shift, :val) ? shift.val : shift
    C_lm = _lm_primal(C.val)
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    mul_AshiftB!(C_lm, A_lm, B_lm, shift_val)

    tapeA_obj, itA = get_block(A_lm.temps)
    tapeA_obj .= A_lm.A
    tapeA = (tapeA_obj, itA)

    tapeB_obj, itB = get_block(B_lm.temps)
    tapeB_obj .= B_lm.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, shift_val))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_AshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation,
    B,
    shift::RT,
) where {RT}
    do_dB = false
    if hasproperty(B, :dval)
        s = _shadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    return _rev_mul_AshiftB!(cfg, dCout, tape, C, A, B, shift; do_dB=do_dB)
end

function _rev_mul_AshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shift;
    do_dB::Bool,
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _shadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _shadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shift = tape
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    Aval = (tapeA === nothing) ? A_lm.A : tapeA[1]
    Bval = (tapeB === nothing) ? B_lm.A : tapeB[1]

    C_lm = _lm_primal(C.val)
    NC1 = Val(C_lm.NC1)
    NC2 = Val(C_lm.NC2)
    NC3 = Val(A_lm.NC2)
    nw = Val(C_lm.nw)
    idxr = C_lm.indexer
    Nsites = prod(C_lm.PN)
    sh = tape_shift

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shift!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, sh
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shift!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, sh
        )
        fold_halo_to_core_grad!(dB_struct)
    end

    if tapeA !== nothing
        unused!(A_lm.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B_lm.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing, nothing)
end

function ER.augmented_primal(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    ::Type{RT},
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation,
    B::ER.Annotation,
    shiftA::RT2,
    shiftB::RT3,
) where {RT,RT2,RT3}
    shiftA_val = hasproperty(shiftA, :val) ? shiftA.val : shiftA
    shiftB_val = hasproperty(shiftB, :val) ? shiftB.val : shiftB

    C_lm = _lm_primal(C.val)
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    mul_shiftAshiftB!(C_lm, A_lm, B_lm, shiftA_val, shiftB_val)

    if A_lm isa Adjoint_Lattice
        tapeA_obj, itA = get_block(A_lm.data.temps)
        tapeA_obj .= A_lm.data.A
        tapeA = (tapeA_obj, itA)
    else
        tapeA_obj, itA = get_block(A_lm.temps)
        tapeA_obj .= A_lm.A
        tapeA = (tapeA_obj, itA)
    end

    tapeB_obj, itB = get_block(B_lm.temps)
    tapeB_obj .= B_lm.A
    tapeB = (tapeB_obj, itB)

    return ER.AugmentedReturn(nothing, nothing, (tapeA, tapeB, shiftA_val, shiftB_val))
end

function ER.reverse(cfg::ER.RevConfig,
    ::ER.Const{typeof(mul_shiftAshiftB!)},
    dCout, tape,
    C::ER.Annotation{<:WilsonFermion_4D_MPILattice},
    A::ER.Annotation,
    B,
    shiftA::RT,
    shiftB::RTB,
) where {RT,RTB}
    do_dB = false
    if hasproperty(B, :dval)
        s = _shadow(getproperty(B, :dval))
        do_dB = (s isa LatticeMatrix)
    end
    A_lm = _lm_primal(A.val)
    if A_lm isa Adjoint_Lattice
        return _rev_mul_shiftAdagshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
    end
    return _rev_mul_shiftAshiftB!(cfg, dCout, tape, C, A, B, shiftA, shiftB; do_dB=do_dB)
end

function _rev_mul_shiftAshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shiftA, shiftB;
    do_dB::Bool,
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = hasproperty(A, :dval) ? _shadow(A.dval) : nothing
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _shadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shiftA, tape_shiftB = tape
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    Aval = (tapeA === nothing) ? A_lm.A : tapeA[1]
    Bval = (tapeB === nothing) ? B_lm.A : tapeB[1]

    C_lm = _lm_primal(C.val)
    NC1 = Val(C_lm.NC1)
    NC2 = Val(C_lm.NC2)
    NC3 = Val(A_lm.NC2)
    nw = Val(C_lm.nw)
    idxr = C_lm.indexer
    Nsites = prod(C_lm.PN)
    shA = tape_shiftA
    shB = tape_shiftB

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dA_from_dC_Bdag_shiftAshiftB_scatter!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        fold_halo_to_core_grad!(dB_struct)
    end

    if tapeA !== nothing
        unused!(A_lm.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B_lm.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing, nothing, nothing)
end

function _rev_mul_shiftAdagshiftB!(
    cfg::ER.RevConfig,
    dCout, tape,
    C, A, B, shiftA, shiftB;
    do_dB::Bool,
)
    dC_struct = _shadow_out(dCout, C)
    dC_struct isa LatticeMatrix || (dC_struct = _shadow(C.dval))
    dC_struct === nothing && return (nothing, nothing, nothing, nothing, nothing)
    dCval = dC_struct.A

    dA_struct = _shadow_data(A.dval)
    dAval = (dA_struct isa LatticeMatrix) ? dA_struct.A : nothing

    dB_struct = do_dB ? _shadow(B.dval) : nothing
    dBval = (do_dB && (dB_struct isa LatticeMatrix)) ? dB_struct.A : nothing

    tapeA, tapeB, tape_shiftA, tape_shiftB = tape
    A_lm = _lm_primal(A.val)
    B_lm = _lm_primal(B.val)
    Aval = (tapeA === nothing) ? A_lm.data.A : tapeA[1]
    Bval = (tapeB === nothing) ? B_lm.A : tapeB[1]

    C_lm = _lm_primal(C.val)
    NC1 = Val(C_lm.NC1)
    NC2 = Val(C_lm.NC2)
    NC3 = Val(A_lm.data.NC1)
    nw = Val(C_lm.nw)
    idxr = C_lm.indexer
    Nsites = prod(C_lm.PN)
    shA = tape_shiftA
    shB = tape_shiftB

    if dAval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mul_dAdag_from_dC_B_shiftAshiftB_scatter!,
            dAval, dCval, Bval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        fold_halo_to_core_grad!(dA_struct)
    end

    if dBval !== nothing
        JACC.parallel_for(
            Nsites,
            kernel_Dmatrix_mulAdagBadd_scatter_shiftAshiftB_adagA!,
            dBval, Aval, dCval,
            NC1, NC2, NC3, nw, idxr, shA, shB
        )
        fold_halo_to_core_grad!(dB_struct)
    end

    if tapeA !== nothing
        unused!(A_lm.data.temps, tapeA[2])
    end
    if tapeB !== nothing
        unused!(B_lm.temps, tapeB[2])
    end

    if _should_zero_dC(dCout)
        _zero_shadow!(dC_struct)
    end
    return (nothing, nothing, nothing, nothing, nothing)
end


#=
function LatticeDiracOperators.dSFdU!(U::Vector{TG}, dfdU::Vector{TG}, apply_D, apply_Ddag, φ::L1) where {TG<:Fields_4D_MPILattice,L1<:GeneralFermion}
    U1 = U[1]
    U2 = U[2]
    U3 = U[3]
    U4 = U[4]

    dfdU1 = dfdU[1]
    dfdU2 = dfdU[2]
    dfdU3 = dfdU[3]
    dfdU4 = dfdU[4]

    DdagD = DdagDgeneral(apply_D, apply_Ddag)

    return

    DdagD = DgagD_General_Dirac_operator(GD)
    #DdagD = DdagDOp(D)
    phitemp1, itphitemp1 = get_block(D.phitemps)
    η = phitemp1

    solve_DinvX!(η, DdagD, φ)
    #solve!(η, DdagD, φ) #η = (DdagD)^-1 φ
    println("solved")
    set_wing_fermion!(η)
    phitemp2, itphitemp2 = get_block(D.phitemps)
    χ = phitemp2
    mul!(χ, D, η)

    #phitemp1, itphitemp1 = get_block(D.phitemps)
    func(U1, U2, U3, U4, χ, η, apply, phitemp, temp) = g(χ, U1, U2, U3, U4, η, D.p, apply, phitemp, temp)

    temp, ittemp = get_block(D.temps, numtemp)
    phitemp, itphitemp = get_block(D.phitemps, numtemp)
    dtemp, itdtemp = get_block(D.temps, numtemp)
    dphitemp, itdphitemp = get_block(D.phitemps, numtemp)

    Enzyme_derivative!(
        func,
        U1,
        U2,
        U3,
        U4,
        dfdU1,
        dfdU2,
        dfdU3,
        dfdU4,
        nodiff(χ), nodiff(η), nodiff(D.apply); temp=temp, dtemp=dtemp, phitemp=phitemp, dphitemp=dphitemp)

    #for μ = 1:4
    #    mul!(dfdU[μ], -2)
    #end

    unused!(D.temps, ittemp)
    unused!(D.temps, itdtemp)
    unused!(D.phitemps, itphitemp)
    unused!(D.phitemps, itdphitemp)
    unused!(D.phitemps, itphitemp1)
    unused!(D.phitemps, itphitemp2)



end
=#

=#