module WilsonCloverTests

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random
using Test

import Gaugefields: Initialize_4DGaugefields
import LatticeDiracOperators.Dirac_operators: apply_γ5!

const LDO = LatticeDiracOperators.Dirac_operators
const KAPPA = 0.08
const FORCE_EPSILON = 1.0e-6

function make_gauge(NC; L=2, condition="hot", seed=2026071601)
    U = Initialize_4DGaugefields(NC, 0, L, L, L, L; condition="cold")
    condition == "cold" && return U
    condition == "hot" || error("unsupported condition: $condition")

    tangent = initialize_TA_Gaugefields(U)
    clear_U!(tangent)
    phase = 0.001 * (seed % 1000)
    for mu in eachindex(U), it in 1:L, iz in 1:L, iy in 1:L, ix in 1:L
        for igen in 1:(NC^2 - 1)
            tangent[mu][igen, ix, iy, iz, it] =
                0.18 * sin(phase + 0.31igen + 0.47mu + 0.19ix + 0.23iy + 0.29iz + 0.37it)
        end
    end
    for mu in eachindex(U)
        expU, temp1, temp2 = (similar(U[mu]) for _ in 1:3)
        exptU!(expU, 1.0, tangent[mu], [temp1, temp2])
        substitute_U!(U[mu], expU)
    end
    set_wing_U!(U)
    return U
end

function make_spinor(U; seed=2026071602)
    Random.seed!(seed + U[1].NC)
    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    gauss_distribution_fermion!(x)
    return x
end

function wilson_parameters(; csw=nothing)
    parameters = Dict{String,Any}(
        "Dirac_operator" => isnothing(csw) ? "Wilson" : "WilsonClover",
        "κ" => KAPPA,
        "faster version" => false,
        "eps_CG" => 1.0e-12,
        "MaxCGstep" => 10000,
        "verbose_level" => 0,
    )
    if !isnothing(csw)
        parameters["cSW"] = csw
    end
    return parameters
end

function apply_operator(D, x)
    out = similar(x)
    mul!(out, D, x)
    return out
end

field_vector(x) = ComplexF64[x[i] for i = 1:length(x)]

function relative_field_error(x, y)
    xv = field_vector(x)
    yv = field_vector(y)
    return norm(xv - yv) / max(norm(yv), 1.0)
end

function perturb_link(U, mu, igen, site, epsilon)
    Uwork = [similar(link) for link in U]
    substitute_U!(Uwork, U)
    tangent = initialize_TA_Gaugefields(U)
    clear_U!(tangent)
    tangent[mu][igen, site...] = 1.0
    expU = similar(U[mu])
    temp1 = similar(U[mu])
    temp2 = similar(U[mu])
    newlink = similar(U[mu])
    exptU!(expU, epsilon, tangent[mu], [temp1, temp2])
    mul!(newlink, expU, U[mu])
    substitute_U!(Uwork[mu], newlink)
    set_wing_U!(Uwork)
    return Uwork
end

function fixed_xy_finite_difference(action, U, X, Y, mu, igen, site)
    Uplus = perturb_link(U, mu, igen, site, FORCE_EPSILON)
    Uminus = perturb_link(U, mu, igen, site, -FORCE_EPSILON)
    DplusX = apply_operator(action.diracoperator(Uplus), X)
    DminusX = apply_operator(action.diracoperator(Uminus), X)
    return -2real((dot(Y, DplusX) - dot(Y, DminusX)) / (2FORCE_EPSILON))
end

function test_operator(NC)
    U = make_gauge(NC)
    x = make_spinor(U)
    Dwilson = Dirac_operator(U, x, wilson_parameters())
    Dzero = Dirac_operator(U, x, wilson_parameters(csw=0.0))
    Dclover = Dirac_operator(U, x, wilson_parameters(csw=0.7))

    @test relative_field_error(apply_operator(Dzero, x), apply_operator(Dwilson, x)) < 1.0e-12
    @test relative_field_error(apply_operator(Dzero', x), apply_operator(Dwilson', x)) < 1.0e-12

    Nwilson = DdagD_operator(U, x, wilson_parameters())
    Nzero = DdagD_operator(U, x, wilson_parameters(csw=0.0))
    @test relative_field_error(apply_operator(Nzero, x), apply_operator(Nwilson, x)) < 1.0e-12

    y = make_spinor(U; seed=2026071603)
    lhs = dot(y, apply_operator(Dclover, x))
    rhs = dot(apply_operator(Dclover', y), x)
    @test lhs ≈ rhs rtol=1.0e-12 atol=1.0e-12

    gamma5_x = deepcopy(x)
    apply_γ5!(gamma5_x)
    gamma5_D_gamma5_x = apply_operator(Dclover, gamma5_x)
    apply_γ5!(gamma5_D_gamma5_x)
    @test relative_field_error(gamma5_D_gamma5_x, apply_operator(Dclover', x)) < 1.0e-12

    maximum_antihermitian_error = 0.0
    maximum_trace_error = 0.0
    for ipair in axes(Dclover.cloverterm.Fmunu, 4)
        for isite in axes(Dclover.cloverterm.Fmunu, 3)
            F = Dclover.cloverterm.Fmunu[:, :, isite, ipair]
            maximum_antihermitian_error = max(maximum_antihermitian_error, norm(F + F'))
            maximum_trace_error = max(maximum_trace_error, abs(tr(F)))
        end
    end
    @test norm(Dclover.cloverterm.Fmunu) > 1.0e-12
    @test maximum_antihermitian_error < 1.0e-12
    @test maximum_trace_error < 1.0e-12

    Dquarter = Dirac_operator(U, x, wilson_parameters(csw=0.25))
    Dhalf = Dirac_operator(U, x, wilson_parameters(csw=0.50))
    y0 = field_vector(apply_operator(Dzero, x))
    yquarter = field_vector(apply_operator(Dquarter, x))
    yhalf = field_vector(apply_operator(Dhalf, x))
    @test norm((yhalf - y0) - 2 .* (yquarter - y0)) / max(norm(yhalf - y0), 1.0) < 1.0e-12

    Ucold = make_gauge(NC; condition="cold")
    Dcold = Dirac_operator(Ucold, x, wilson_parameters(csw=1.0))
    Wcold = Dirac_operator(Ucold, x, wilson_parameters())
    @test norm(Dcold.cloverterm.Fmunu) < 1.0e-12
    @test relative_field_error(apply_operator(Dcold, x), apply_operator(Wcold, x)) < 1.0e-12

    parameters = wilson_parameters(csw=0.4)
    Dirac_operator(U, x, parameters)
    @test !haskey(parameters, "hasclover")
end

function test_force(NC; L=1, all_generators=true)
    U = make_gauge(NC; L=L, seed=2026071610)
    X = make_spinor(U; seed=2026071611)
    D = Dirac_operator(U, X, wilson_parameters(csw=0.7))
    action = FermiAction(D, Dict{String,Any}())

    Y = similar(X)
    p = initialize_TA_Gaugefields(U)
    clear_U!(p)
    LDO.calc_p_UdSfdU_fromX!(p, Y, action, U, X; coeff=1.0)

    raw_force = [similar(link) for link in U]
    clear_U!(raw_force)
    Yraw = similar(X)
    LDO.calc_UdSfdU_fromX!(raw_force, Yraw, action, U, X; coeff=1.0)
    projected_force = initialize_TA_Gaugefields(U)
    clear_U!(projected_force)
    for mu in eachindex(U)
        Traceless_antihermitian_add!(projected_force[mu], 1.0, raw_force[mu])
    end

    generators = LDO._sun_generator_matrices_for_clover_force(NC)
    generator_indices = all_generators ? eachindex(generators) : 1:1
    sites = L == 1 ? ((1, 1, 1, 1),) : ((1, 1, 1, 1), (2, 1, 2, 1))
    directions = L == 1 ? (1,) : eachindex(U)
    for site in sites, mu in directions, igen in generator_indices
        finite_difference = fixed_xy_finite_difference(action, U, X, Y, mu, igen, site)
        @test p[mu][igen, site...] ≈ finite_difference rtol=2.0e-6 atol=2.0e-7
        @test projected_force[mu][igen, site...] ≈
              p[mu][igen, site...] rtol=1.0e-12 atol=1.0e-12
    end

    if NC == 2 && L == 1
        site = only(sites)
        phi = similar(X)
        normal_operator = LDO.DdagD_Wilson_operator(D)
        mul!(phi, normal_operator, X)
        solved_force = initialize_TA_Gaugefields(U)
        clear_U!(solved_force)
        calc_p_UdSfdU!(solved_force, action, U, phi, 1.0)
        for direction in eachindex(U), igen in eachindex(generators)
            @test solved_force[direction][igen, site...] ≈
                  p[direction][igen, site...] rtol=2.0e-6 atol=2.0e-7
        end
    end
end

function test_csw_zero_force(NC)
    U = make_gauge(NC; L=1, seed=2026071620)
    X = make_spinor(U; seed=2026071621)
    Dwilson = Dirac_operator(U, X, wilson_parameters())
    Dzero = Dirac_operator(U, X, wilson_parameters(csw=0.0))
    wilson_action = FermiAction(Dwilson, Dict{String,Any}())
    zero_action = FermiAction(Dzero, Dict{String,Any}())
    pwilson = initialize_TA_Gaugefields(U)
    pzero = initialize_TA_Gaugefields(U)
    clear_U!(pwilson)
    clear_U!(pzero)
    Ywilson = similar(X)
    Yzero = similar(X)
    LDO.calc_p_UdSfdU_fromX!(pwilson, Ywilson, wilson_action, U, X; coeff=1.0)
    LDO.calc_p_UdSfdU_fromX!(pzero, Yzero, zero_action, U, X; coeff=1.0)
    generators = LDO._sun_generator_matrices_for_clover_force(NC)
    site = (1, 1, 1, 1)
    for direction in eachindex(U), igen in eachindex(generators)
        @test pzero[direction][igen, site...] ≈
              pwilson[direction][igen, site...] rtol=1.0e-12 atol=1.0e-12
    end
end

site_array(y, NC, L) = ComplexF64[y[c, ix, iy, iz, it, alpha]
                                  for c = 1:NC, ix = 1:L, iy = 1:L, iz = 1:L, it = 1:L, alpha = 1:4]

# The standard clover term -κ c_SW Σ_{μ<ν} σ_μν F_μν x (Bridge++, LatticeMatrices.jl),
# with σ_μν = [γ_μ, γ_ν]/2 and F_μν the traceless part of (Q_μν - Q_μν†)/8, Q_μν the
# sum of the four counterclockwise leaves at x, written out here independently of the
# package's clover code.
function reference_clover_term(U, x, kappa, csw)
    NC = U[1].NC
    L = U[1].NX
    gamma, _, _ = LDO.mk_gamma(1.0)
    link(mu, s) = ComplexF64[U[mu][i, j, s...] for i = 1:NC, j = 1:NC]
    shift(s, mu, k) = ntuple(d -> d == mu ? mod1(s[d] + k, L) : s[d], 4)
    out = zeros(ComplexF64, NC, L, L, L, L, 4)
    for it = 1:L, iz = 1:L, iy = 1:L, ix = 1:L
        s = (ix, iy, iz, it)
        psi = [ComplexF64[x[c, s..., beta] for c = 1:NC] for beta = 1:4]
        for mu = 1:4, nu = mu+1:4
            smu, snu = shift(s, mu, -1), shift(s, nu, -1)
            Q = link(mu, s) * link(nu, shift(s, mu, 1)) * link(mu, shift(s, nu, 1))' * link(nu, s)' +
                link(nu, s) * link(mu, shift(smu, nu, 1))' * link(nu, smu)' * link(mu, smu) +
                link(mu, smu)' * link(nu, shift(smu, nu, -1))' * link(mu, shift(smu, nu, -1)) * link(nu, snu) +
                link(nu, snu)' * link(mu, snu) * link(nu, shift(snu, mu, 1)) * link(mu, s)'
            F = (Q - Q') / 8
            F -= (tr(F) / NC) * I
            sigma = (gamma[:, :, mu] * gamma[:, :, nu] - gamma[:, :, nu] * gamma[:, :, mu]) / 2
            for alpha = 1:4, beta = 1:4
                iszero(sigma[alpha, beta]) && continue
                out[:, s..., alpha] .+= (-kappa * csw * sigma[alpha, beta]) .* (F * psi[beta])
            end
        end
    end
    return out
end

function test_clover_sign(NC; csw=0.7)
    L = 3
    U = make_gauge(NC; L=L, seed=2026110601)
    x = make_spinor(U; seed=2026110602)
    Dwilson = Dirac_operator(U, x, wilson_parameters())
    Dclover = Dirac_operator(U, x, wilson_parameters(csw=csw))
    clover_only = site_array(apply_operator(Dclover, x), NC, L) .- site_array(apply_operator(Dwilson, x), NC, L)
    reference = reference_clover_term(U, x, KAPPA, csw)
    @test norm(reference) > 1.0e-6
    @test norm(clover_only - reference) / norm(reference) < 1.0e-12
end

@testset "Wilson-clover operator" begin
    for NC in (2, 3)
        test_operator(NC)
    end
end

@testset "Wilson-clover sign convention" begin
    for NC in (2, 3)
        test_clover_sign(NC)
    end
end

@testset "Wilson-clover analytic force" begin
    for NC in (2, 3)
        test_force(NC; L=1, all_generators=true)
        test_csw_zero_force(NC)
    end
    for NC in (2, 3)
        test_force(NC; L=2, all_generators=true)
    end
end

end
