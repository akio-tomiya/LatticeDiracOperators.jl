module ReadmeExamples

using Gaugefields
using LatticeDiracOperators
using LinearAlgebra
using Optimisers
using Random
using Test
import Gaugefields.Abstractsmearing_module:
    get_parameters, zero_grad!, set_parameters!

finite_norm(field) = hasproperty(field, :U) ?
    all(isfinite, getproperty(field, :U)) : isfinite(real(dot(field, field)))

function contains_parse_error(expr)
    expr isa Expr || return false
    expr.head in (:error, :incomplete) && return true
    return any(contains_parse_error, expr.args)
end

@testset "README Julia code blocks parse" begin
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    blocks = [match.captures[1] for match in eachmatch(r"```julia\s*\n(.*?)```"s, readme)]
    @test !isempty(blocks)
    for (index, block) in enumerate(blocks)
        expr = Meta.parseall(block; filename="README.md Julia block $index")
        @test !contains_parse_error(expr)
    end
end

@testset "README basic Wilson examples" begin
    Random.seed!(101)
    U = Initialize_4DGaugefields(3, 1, 4, 4, 4, 4; condition="cold")
    x = Initialize_pseudofermion_fields(U[1], "Wilson")
    y = similar(x)
    gauss_distribution_fermion!(x)

    params = Dict(
        "Dirac_operator" => "Wilson",
        "κ" => 0.141139,
        "eps_CG" => 1e-8,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, params)
    mul!(y, D, x)
    @test finite_norm(y)

    solve_DinvX!(y, D, x)
    @test finite_norm(y)
    solve_DinvX!(y, D', x)
    @test finite_norm(y)

    DdagD = DdagD_operator(U, x, params)
    solve_DinvX!(y, DdagD, x)
    @test finite_norm(y)

    action = FermiAction(D, Dict())
    @test isfinite(evaluate_FermiAction(action, U, x))
    force = calc_UdSfdU(action, U, x)
    @test length(force) == 4
    @test all(finite_norm, force)
end

@testset "README staggered examples" begin
    Random.seed!(102)
    U = Initialize_4DGaugefields(3, 1, 4, 4, 4, 4; condition="cold")
    x = Initialize_pseudofermion_fields(U[1], "staggered")
    y = similar(x)
    gauss_distribution_fermion!(x)

    params = Dict(
        "Dirac_operator" => "staggered",
        "mass" => 0.1,
        "eps_CG" => 1e-8,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, params)
    mul!(y, D, x)
    @test finite_norm(y)
    solve_DinvX!(y, D, x)
    @test finite_norm(y)

    action = FermiAction(D, Dict("Nf" => 2))
    @test isfinite(evaluate_FermiAction(action, U, x))
    force = calc_UdSfdU(action, U, x)
    @test length(force) == 4
    @test all(finite_norm, force)
end

@testset "README domain-wall examples" begin
    Random.seed!(103)
    U = Initialize_4DGaugefields(3, 1, 4, 4, 4, 4; condition="cold")
    L5 = 4
    x = Initialize_pseudofermion_fields(U[1], "Domainwall"; L5)
    y = similar(x)
    gauss_distribution_fermion!(x)
    @test isfinite(real(x.w[1][1, 1, 1, 1, 1, 1]))

    params = Dict(
        "Dirac_operator" => "Domainwall",
        "mass" => 0.1,
        "L5" => L5,
        "eps_CG" => 1e-10,
        "MaxCGstep" => 3_000,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, params)
    solve_DinvX!(y, D, x)
    @test finite_norm(y)
    z = similar(x)
    mul!(z, D, y)
    @test finite_norm(z)

    action = FermiAction(D, Dict())
    @test isfinite(evaluate_FermiAction(action, U, x))
    force = calc_UdSfdU(action, U, x)
    @test length(force) == 4
    @test all(finite_norm, force)
end

@testset "README STOUT and SLHMC examples" begin
    Random.seed!(104)
    U = Initialize_Gaugefields(2, 0, 4, 4, 4, 4; condition="hot")
    network = CovNeuralnet(U)
    layer_names = [
        "plaquette", "polyakov_x", "polyakov_y", "polyakov_z", "polyakov_t"]
    push!(network, STOUT_Layer(layer_names, fill(1e-3, 5), U))
    push!(network, STOUT_Layer(layer_names, fill(-1e-3, 5), U))

    smeared_U, smeared_history, _ = calc_smearedU(U, network)
    @test length(smeared_U) == 4

    x = Initialize_pseudofermion_fields(U[1], "staggered")
    xi = similar(x)
    params = Dict(
        "Dirac_operator" => "staggered",
        "mass" => 0.3,
        "eps_CG" => 1e-8,
        "verbose_level" => 0,
    )
    D = Dirac_operator(U, x, params)
    action = FermiAction(D, Dict("Nf" => 4))
    gauss_sampling_in_action!(xi, smeared_U, action)
    sample_pseudofermions!(x, smeared_U, action, xi)
    @test isfinite(evaluate_FermiAction(action, smeared_U, x))

    force = calc_UdSfdU(action, smeared_U, x)
    dS_dUout = similar(smeared_U)
    for mu in 1:4
        mul!(dS_dUout[mu], smeared_U[mu]', force[mu])
    end
    zero_grad!(network)
    bare_force = back_prop(dS_dUout, network, smeared_history, U)
    @test length(bare_force) == 4
    @test all(finite_norm, bare_force)

    theta = get_parameters(network)
    state = Optimisers.setup(Optimisers.Adam(1e-4), theta)
    Optimisers.update!(state, theta, zero(theta))
    set_parameters!(network, theta)
    @test all(isfinite, theta)
end

end # module ReadmeExamples
