function calc_UdSfdU_fromX!(
    UdSfdU::Vector{TG},
    Y::TF,
    fermi_action::StaggeredFermiAction{
        Dim,Dirac,fermion,gauge,Nf
    },
    U::Vector{TG},
    X::TF;
    coeff=1,
) where {
    Dim,
    Dirac<:HISQ_Dirac_operator_MPILattice,
    fermion,
    gauge,
    Nf,
    TG<:Gaugefields_4D_MPILattice,
    TF<:StaggeredFermion_4D_MPILattice,
}
    X.f.nw >= 3 || throw(ArgumentError(
        "HISQ fermion-force evaluation requires halo width at least 3"))

    W = fermi_action.diracoperator(U)
    mul!(Y, W, X)

    gauge_work, gauge_token = get_temp(
        fermi_action._temporary_gaugefields, 5)

    raw_gradient = gauge_work[1:4]
    converted_gradient = gauge_work[5]
    try
        clear_U!(raw_gradient)
        hisq_link_pullback!(
            ntuple(mu -> raw_gradient[mu].U, Val(4)),
            W.cache,
            ntuple(mu -> U[mu].U, Val(4)),
            Y.f,
            X.f;
            coefficient=coeff,
        )

        for mu in 1:Dim
            mul!(converted_gradient, U[mu], raw_gradient[mu]')
            add_U!(UdSfdU[mu], converted_gradient)
        end
    finally
        unused!(fermi_action._temporary_gaugefields, gauge_token)
    end
    return nothing
end
