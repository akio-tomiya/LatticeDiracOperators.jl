# S(m) is the normalized even Schur operator.  Gauge-independent on-site
# determinants cancel out of the MD force, so the pseudofermion action is
# phi† S(PV) [S(m)† S(m)]^{-1} S(PV)† phi on the even subspace.

function _evaluate_generalized_eo_action(action, U, phi)
    A = action.diracoperator.D5DW(U)
    P = action.diracoperator.D5DW_PV(U)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 2) do fields
        rhs, solution = fields
        mul!(rhs, D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(P)', phi)
        solve_DdagD_even!(solution, A, rhs)
        real(dot(rhs, solution))
    end
end

function _sample_generalized_eo_pseudofermions!(phi, U, action, noise)
    A = action.diracoperator.D5DW(U)
    P = action.diracoperator.D5DW_PV(U)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 2) do fields
        rhs, solution = fields
        # phi = S(PV)^{-†} S(m)† eta_even.
        mul!(rhs, D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A)', noise)
        solve_DdagD_even!(solution, P, rhs)
        mul!(phi, D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(P), solution)
        GeneralizedD5DW_keep_parity!(phi, true)
        nothing
    end
end

function _calc_generalized_eo_force!(force, action, U, phi)
    A = action.diracoperator.D5DW(U)
    P = action.diracoperator.D5DW_PV(U)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 3) do fields
        phi_even, rhs, solution = fields
        substitute_fermion!(phi_even, phi)
        GeneralizedD5DW_keep_parity!(phi_even, true)
        mul!(rhs, D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(P)', phi_even)
        solve_DdagD_even!(solution, A, rhs)
        clear_U!(force)
        calc_UdSfdU_eo_fromX!(force, phi_even, action, U, solution, A, P)
        set_wing_U!(force)
        nothing
    end
end

function calc_UdSfdU_eo_fromX!(force, phi_even, action, U, solution, A, P; coeff=1)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 1) do fields
        image = fields[1]
        mul!(image, D5DW_GeneralizedDomainwall_operator_evenodd_MPILattice(A), solution)
        add_schur_force_pair!(force, phi_even, solution, action, U, P; coeff)
        add_schur_force_pair!(force, image, solution, action, U, A; coeff=-coeff)
        nothing
    end
end

function add_schur_force_pair!(force, left, right, action, U, D; coeff=1)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 4) do fields
        left_bar, v_odd, w_odd, tmp = fields
        D5DW_diag_solve_adjoint_direct!(left_bar, D, left, true)
        GeneralizedD5DW_offdiag_blockx_clean!(tmp, D, right, false; sethalo=false)
        D5DW_diag_solve_block!(v_odd, D, tmp, false)
        GeneralizedD5DW_offdiag_adjoint_blockx_clean!(tmp, D, left_bar, false; sethalo=false)
        D5DW_diag_solve_adjoint_direct!(w_odd, D, tmp, false)
        # delta S = -E^{-1} [(delta B) O^{-1} C + B O^{-1} (delta C)].
        add_generalized_hopping_force_pair!(force, left_bar, v_odd, action, U, D; coeff=-coeff)
        add_generalized_hopping_force_pair!(force, w_odd, right, action, U, D; coeff=-coeff)
        nothing
    end
end

function add_generalized_hopping_force_pair!(force, left, right, action, U, D; coeff=1)
    return _with_generalized_eo_fields(action._temporary_fermionfields, 4) do fields
        X, scratch, t0, t1 = fields
        _with_generalized_eo_fields(action._temporary_gaugefields, 1) do gauges
            tg = gauges[1]
            apply_F!(X, D.L5, D.mass, right, scratch)
            clear_fermion!(scratch)
            combine_generalized_domainwall_force!(X, right, scratch, D.D.a, D.D.b, D.D.c)
            for mu in 1:4
                Xplus = shift_fermion(X, mu)
                mul!(t0, U[mu], Xplus)
                mul_1minusγμx!(t1, t0, mu)
                mul!(t0, 0.5, t1)
                muladd_U!(force[mu], coeff, tg, t0, left', t1)

                Yplus = shift_fermion(left, mu)
                mul!(t0, Yplus', U[mu]')
                mul_x1plusγμ!(t1, t0, mu)
                mul!(t0, 0.5, t1)
                muladd_U!(force[mu], -coeff, tg, X, t0, t1)
            end
            nothing
        end
    end
end
