module WilsonCloverOperatorCheckpoints

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Random
using SparseArrays
using Test

import Gaugefields: Initialize_4DGaugefields
import LatticeDiracOperators.Dirac_operators: apply_γ5!

const STRICT = get(ENV, "WILSON_CLOVER_STRICT", "1") != "0"
const DEFAULT_KAPPA = 0.1
const DEFAULT_RTOL = 1.0e-10

function make_gauge(; NC = 3, L = 2, condition = "cold")
    return Initialize_4DGaugefields(NC, 1, L, L, L, L; condition = condition)
end

function make_spinor(U; seed = 20260706)
    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    Random.seed!(seed)
    gauss_distribution_fermion!(x)
    return x
end

function wilson_params(; kappa = DEFAULT_KAPPA)
    return Dict{String,Any}(
        "Dirac_operator" => "Wilson",
        "κ" => kappa,
        "verbose_level" => 1,
    )
end

function clover_params(; cSW, kappa = DEFAULT_KAPPA)
    params = wilson_params(; kappa = kappa)
    params["hasclover"] = true
    params["cSW"] = cSW
    params["clover_coefficient"] = cSW
    return params
end

function named_clover_params(; cSW, kappa = DEFAULT_KAPPA)
    return Dict{String,Any}(
        "Dirac_operator" => "WilsonClover",
        "κ" => kappa,
        "verbose_level" => 1,
        "cSW" => cSW,
    )
end

function coefficient_alias_params(; cSW, kappa = DEFAULT_KAPPA)
    params = wilson_params(; kappa = kappa)
    params["hasclover"] = true
    params["clover_coefficient"] = cSW
    return params
end

function try_construct_clover(U, x; cSW)
    try
        return Dirac_operator(U, x, clover_params(; cSW = cSW)), nothing
    catch err
        return nothing, err
    end
end

function field_vector(x)
    return ComplexF64[x[i] for i in 1:length(x)]
end

function l2norm_field(x)
    return norm(field_vector(x))
end

function l2diff_field(x, y)
    @assert length(x) == length(y)
    return norm(field_vector(x) - field_vector(y))
end

function relative_field_error(x, y)
    return l2diff_field(x, y) / max(l2norm_field(y), 1.0)
end

function relative_vector_field_error(v, y)
    return norm(v - field_vector(y)) / max(norm(v), 1.0)
end

function apply_operator(D, x)
    y = similar(x)
    mul!(y, D, x)
    return y
end

function apply_gamma5_copy(x)
    y = deepcopy(x)
    apply_γ5!(y)
    return y
end

function require_clover_constructor()
    U = make_gauge(; NC = 3, L = 2, condition = "cold")
    x = make_spinor(U)
    D, err = try_construct_clover(U, x; cSW = 0.0)

    @testset "Wilson-clover constructor availability" begin
        if D === nothing
            if STRICT
                @error "Wilson-clover constructor failed in strict mode" exception = (err, catch_backtrace())
                @test D !== nothing
            else
                @info "Wilson-clover constructor is not implemented yet; non-strict mode stops here" err
                @test D === nothing
            end
        else
            @test D !== nothing
        end
    end

    return D !== nothing
end

function test_csw_zero_regression(; NC = 3, condition = "hot")
    U = make_gauge(; NC = NC, L = 2, condition = condition)
    x = make_spinor(U; seed = 20260706 + NC)

    Dwilson = Dirac_operator(U, x, wilson_params())
    Dclover, err = try_construct_clover(U, x; cSW = 0.0)
    @test Dclover !== nothing
    Dclover === nothing && error(err)

    yw = apply_operator(Dwilson, x)
    yc = apply_operator(Dclover, x)
    @test relative_field_error(yc, yw) < DEFAULT_RTOL

    ywd = apply_operator(Dwilson', x)
    ycd = apply_operator(Dclover', x)
    @test relative_field_error(ycd, ywd) < DEFAULT_RTOL

    DwDw = DdagD_operator(U, x, wilson_params())
    DcDc = DdagD_operator(U, x, clover_params(; cSW = 0.0))
    yww = apply_operator(DwDw, x)
    ycc = apply_operator(DcDc, x)
    @test relative_field_error(ycc, yww) < DEFAULT_RTOL
end

function test_identity_gauge_removes_clover(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "cold")
    x = make_spinor(U; seed = 20260716 + NC)

    Dwilson = Dirac_operator(U, x, wilson_params())
    Dclover, err = try_construct_clover(U, x; cSW = 1.0)
    @test Dclover !== nothing
    Dclover === nothing && error(err)

    yw = apply_operator(Dwilson, x)
    yc = apply_operator(Dclover, x)
    @test relative_field_error(yc, yw) < DEFAULT_RTOL
end

function test_clover_field_strength_diagnostics(; NC = 3)
    Ucold = make_gauge(; NC = NC, L = 2, condition = "cold")
    xcold = make_spinor(Ucold; seed = 20260718 + NC)
    Dcold, errcold = try_construct_clover(Ucold, xcold; cSW = 1.0)
    @test Dcold !== nothing
    Dcold === nothing && error(errcold)
    @test norm(Dcold.cloverterm.Fmunu) < DEFAULT_RTOL

    Uhot = make_gauge(; NC = NC, L = 2, condition = "hot")
    xhot = make_spinor(Uhot; seed = 20260719 + NC)
    Dhot, errhot = try_construct_clover(Uhot, xhot; cSW = 1.0)
    @test Dhot !== nothing
    Dhot === nothing && error(errhot)

    max_antihermitian_error = 0.0
    max_trace_error = 0.0
    for ipair in axes(Dhot.cloverterm.Fmunu, 4)
        for isite in axes(Dhot.cloverterm.Fmunu, 3)
            F = Dhot.cloverterm.Fmunu[:, :, isite, ipair]
            max_antihermitian_error = max(max_antihermitian_error, norm(F + F'))
            max_trace_error = max(max_trace_error, abs(tr(F)))
        end
    end

    @test norm(Dhot.cloverterm.Fmunu) > 1.0e-12
    @test max_antihermitian_error < DEFAULT_RTOL
    @test max_trace_error < DEFAULT_RTOL
end

function test_hot_gauge_has_nonzero_clover_contribution(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260721 + NC)

    D0, err0 = try_construct_clover(U, x; cSW = 0.0)
    D1, err1 = try_construct_clover(U, x; cSW = 1.0)
    @test D0 !== nothing
    @test D1 !== nothing
    D0 === nothing && error(err0)
    D1 === nothing && error(err1)

    y0 = apply_operator(D0, x)
    y1 = apply_operator(D1, x)
    @test relative_field_error(y1, y0) > 1.0e-12
end

function test_gamma5_hermiticity(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260726 + NC)

    D, err = try_construct_clover(U, x; cSW = 0.7)
    @test D !== nothing
    D === nothing && error(err)

    gx = apply_gamma5_copy(x)
    lhs = apply_operator(D, gx)
    apply_γ5!(lhs)

    rhs = apply_operator(D', x)
    @test relative_field_error(lhs, rhs) < 5.0e-10
end

function test_dag_d_positivity(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260736 + NC)

    D, err = try_construct_clover(U, x; cSW = 0.5)
    @test D !== nothing
    D === nothing && error(err)

    Dx = apply_operator(D, x)
    DdagDx = apply_operator(D', Dx)
    qform = dot(x, DdagDx)
    dxnorm2 = dot(Dx, Dx)

    @test abs(imag(qform)) < 5.0e-10 * max(abs(real(qform)), 1.0)
    @test real(qform) >= -5.0e-10
    @test abs(real(qform) - real(dxnorm2)) / max(abs(real(dxnorm2)), 1.0) < 5.0e-10
end

function test_sparse_matrix_roundtrip(; NC = 2)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260746 + NC)

    D, err = try_construct_clover(U, x; cSW = 0.5)
    @test D !== nothing
    D === nothing && error(err)

    mat = construct_sparsematrix(D)
    y = apply_operator(D, x)
    yvec = mat * field_vector(x)
    @test relative_vector_field_error(yvec, y) < DEFAULT_RTOL
end

function test_csw_linearity(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260756 + NC)

    D0, err0 = try_construct_clover(U, x; cSW = 0.0)
    D1, err1 = try_construct_clover(U, x; cSW = 0.25)
    D2, err2 = try_construct_clover(U, x; cSW = 0.50)
    @test D0 !== nothing
    @test D1 !== nothing
    @test D2 !== nothing
    D0 === nothing && error(err0)
    D1 === nothing && error(err1)
    D2 === nothing && error(err2)

    y0 = field_vector(apply_operator(D0, x))
    y1 = field_vector(apply_operator(D1, x))
    y2 = field_vector(apply_operator(D2, x))

    c1 = y1 - y0
    c2 = y2 - y0
    @test norm(c2 - 2.0 .* c1) / max(norm(c2), 1.0) < 5.0e-10
end

function test_named_operator_and_parameter_aliases(; NC = 3)
    U = make_gauge(; NC = NC, L = 2, condition = "hot")
    x = make_spinor(U; seed = 20260766 + NC)

    Dhasclover = Dirac_operator(U, x, clover_params(; cSW = 0.4))
    Dnamed = Dirac_operator(U, x, named_clover_params(; cSW = 0.4))
    Dcoefficient = Dirac_operator(U, x, coefficient_alias_params(; cSW = 0.4))

    yhasclover = apply_operator(Dhasclover, x)
    ynamed = apply_operator(Dnamed, x)
    ycoefficient = apply_operator(Dcoefficient, x)

    @test relative_field_error(ynamed, yhasclover) < DEFAULT_RTOL
    @test relative_field_error(ycoefficient, yhasclover) < DEFAULT_RTOL

    Nhasclover = DdagD_operator(U, x, clover_params(; cSW = 0.4))
    Nnamed = DdagD_operator(U, x, named_clover_params(; cSW = 0.4))
    @test relative_field_error(apply_operator(Nnamed, x), apply_operator(Nhasclover, x)) < DEFAULT_RTOL
end

function test_gauge_rebind_rebuilds_clover(; NC = 3)
    Uhot = make_gauge(; NC = NC, L = 2, condition = "hot")
    Ucold = make_gauge(; NC = NC, L = 2, condition = "cold")
    x = make_spinor(Uhot; seed = 20260776 + NC)

    Dhot = Dirac_operator(Uhot, x, clover_params(; cSW = 0.6))
    Drebuilt = Dhot(Ucold)
    Ddirect = Dirac_operator(Ucold, x, clover_params(; cSW = 0.6))

    @test relative_field_error(apply_operator(Drebuilt, x), apply_operator(Ddirect, x)) < DEFAULT_RTOL
end

function run()
    if !require_clover_constructor()
        return
    end

    @testset "Wilson-clover cSW=0 regression" begin
        for NC in (2, 3)
            test_csw_zero_regression(; NC = NC, condition = "hot")
        end
    end

    @testset "Wilson-clover identity gauge" begin
        for NC in (2, 3)
            test_identity_gauge_removes_clover(; NC = NC)
        end
    end

    @testset "Wilson-clover field-strength diagnostics" begin
        test_clover_field_strength_diagnostics(; NC = 3)
    end

    @testset "Wilson-clover nonzero hot-gauge contribution" begin
        test_hot_gauge_has_nonzero_clover_contribution(; NC = 3)
    end

    @testset "Wilson-clover gamma5 hermiticity" begin
        test_gamma5_hermiticity(; NC = 3)
    end

    @testset "Wilson-clover DdagD positivity" begin
        test_dag_d_positivity(; NC = 3)
    end

    @testset "Wilson-clover sparse-matrix roundtrip" begin
        test_sparse_matrix_roundtrip(; NC = 2)
    end

    @testset "Wilson-clover cSW linearity" begin
        test_csw_linearity(; NC = 3)
    end

    @testset "Wilson-clover API aliases" begin
        test_named_operator_and_parameter_aliases(; NC = 3)
    end

    @testset "Wilson-clover gauge rebind" begin
        test_gauge_rebind_rebuilds_clover(; NC = 3)
    end
end

end # module

WilsonCloverOperatorCheckpoints.run()
