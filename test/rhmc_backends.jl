module RHMCBackendTests
using Test, LinearAlgebra, Random, Gaugefields, LatticeMatrices, LatticeDiracOperators, JACC
JACC.@init_backend
include("test_communicator.jl")
const snapshots = Dict{String,Vector{Float64}}()
pack(a) = vcat(vec(real.(a)), vec(imag.(a)))

struct StaggeredOperator{T}
    operator::T
end
function LinearAlgebra.mul!(y::GeneralFermion, D::StaggeredOperator, x::GeneralFermion)
    set_wing_fermion!(x)
    mul!(y, D.operator, x)
    set_wing_fermion!(y)
    return y
end
Base.adjoint(D::StaggeredOperator) = StaggeredOperator(adjoint(D.operator))

function dense_matrix(D, x)
    unit, image = similar(x), similar(x)
    dense = zeros(ComplexF64, length(x), length(x))
    for j in axes(dense, 2)
        values = zeros(ComplexF64, length(x)); values[j] = 1
        substitute_fermion!(unit, values)
        mul!(image, D, unit)
        dense[:,j] = convert_to_normalvector(image)
    end
    return dense
end

@testset "RHMC action and force across backends" begin
    dims = (2,2,2,4)
    grid = (1,1,1,ldo_test_comm_size())
    U = gauge_configuration(dims; colors=3, halo=1, start=:hot,
        seed=UInt64(0x52484d43), process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
    x = GeneralFermion(3,1,dims,grid; nw=1, phases=(1,1,1,-1), comm0=LDO_TEST_COMM)
    D = StaggeredOperator(StaggeredDiracOperator4D(getproperty.(U,:U), 0.5))
    dense = dense_matrix(D, x)
    spectrum = eigen(Hermitian(dense' * dense))
    values = ComplexF64[sin(i)+im*cos(2i) for i in 1:length(x)]
    noise, phi = similar(x), similar(x)
    substitute_fermion!(noise, values)
    direction = gauge_configuration(dims; colors=3, halo=1, start=:hot,
        seed=UInt64(0x44495252), process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
    for nf in (1,2,4,8)
        action = StaggeredFermiAction(U,x; mass=0.5, Nf=nf,
            eps_CG=1e-24, maxsteps=2000, verbose_level=0)
        if nf == 4
            normal = action.diracoperator.DdagD
            bad = GeneralFermion(3,1,dims,grid; nw=1, phases=(1,1,1,1), comm0=LDO_TEST_COMM)
            @test_throws ArgumentError solve_DinvX!(bad,normal,noise)
        end
        sample_pseudofermions!(phi,U,action,noise)
        pv = convert_to_normalvector(phi)
        expected_phi = spectrum.vectors * (spectrum.values.^(nf/16) .* (spectrum.vectors' * values))
        # Nf=4/8 use D' directly; Nf=4 retains only even sites.
        if nf in (4,8)
            expected_phi = dense' * values
            if nf == 4
                sites = CartesianIndices((3,1,dims...))
                for i in eachindex(expected_phi)
                    isodd(sum(Tuple(sites[i])[3:end])) && (expected_phi[i] = 0)
                end
            end
        end
        @test pv ≈ expected_phi rtol=2e-7 atol=2e-8
        value = evaluate_FermiAction(action,U,phi)
        power = nf in (4,8) ? 1.0 : nf/8
        expected_action = real(dot(pv, spectrum.vectors *
            (spectrum.values.^(-power) .* (spectrum.vectors' * pv))))
        @test value ≈ expected_action rtol=2e-7 atol=2e-8
        if nf != 4
            @test value ≈ real(dot(values,values)) rtol=2e-7 atol=2e-8
        end
        force = similar(U)
        calc_UdSfdU!(force,action,U,phi)
        @test all(u -> all(isfinite, Array(u.U.A)), force)
        gradient = similar(U[1])
        analytic = 0.0
        for mu in 1:4
            mul!(gradient, force[mu]', U[mu])
            analytic -= 2real(dot(gradient.U, direction[mu].U))
        end
        derivatives = Float64[]
        for epsilon in (3e-4, 1e-4)
            plus, minus = copy_configuration(U), copy_configuration(U)
            for mu in 1:4
                add_U!(plus[mu],epsilon,direction[mu])
                add_U!(minus[mu],-epsilon,direction[mu])
            end
            fd = (evaluate_FermiAction(action,plus,phi)-evaluate_FermiAction(action,minus,phi))/(2epsilon)
            push!(derivatives,fd)
            @test analytic ≈ fd rtol=2e-4 atol=2e-5
        end
        snapshots["nf$(nf)-phi"] = pack(pv)
        snapshots["nf$(nf)-action-force-derivatives"] = [value,analytic,derivatives...]
        for mu in 1:4
            snapshots["nf$(nf)-force$(mu)"] = pack(gather_and_bcast_matrix(force[mu].U))
        end
    end
end
end
