module CloverMigrationTests
using Test, LinearAlgebra, Gaugefields, LatticeDiracOperators, LatticeMatrices, JACC
JACC.@init_backend
include("test_communicator.jl")
const DO = LatticeDiracOperators.Dirac_operators
const snapshots = Dict{String,Vector{Float64}}()
pack(a) = vcat(vec(real.(a)), vec(imag.(a)))
global_values(x) = gather_and_bcast_matrix(x.f)

function set_values!(x, values)
    substitute!(x.f, LatticeMatrix(values, 4, x.f.dims;
        nw=x.f.nw, phases=x.f.phases, comm0=x.f.comm))
    set_wing_fermion!(x)
end

@testset "Clover migration and shared operator/force implementation" begin
    dims = (4, 2, 2, 4)
    grid = (1, 1, 1, ldo_test_comm_size())
    for nc in (2, 3)
        # Legacy objects are explicit host inputs, even in GPU/MPI test runs.
        old = Gaugefields.gauge_configuration(dims;
            backend=Gaugefields.LegacyBackend(), colors=nc, halo=0)
        old_x = Initialize_pseudofermion_fields(old[1], "Wilson")
        parameters = Dict{String,Any}("Dirac_operator"=>"WilsonClover", "κ"=>0.1,
            "cSW"=>1.0, "verbose_level"=>0, "eps_CG"=>1e-24)
        err = try Dirac_operator(old, old_x, parameters); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("load_configuration!", sprint(showerror, err))
        @test !haskey(parameters, "hasclover")
        compatibility = merge(parameters, Dict("Dirac_operator"=>"Wilson", "hasclover"=>true))
        @test_throws ArgumentError Dirac_operator(old, old_x, compatibility)
        @test_throws ArgumentError DO.Wilson_Dirac_operator(old, old_x, compatibility)

        U = Gaugefields.gauge_configuration(dims; colors=nc, halo=1, start=:hot,
            seed=UInt64(752), process_grid=grid, comm=LDO_TEST_COMM)
        originals = [gather_and_bcast_matrix(u.U) for u in U]
        values = reshape([sin(i)+im*cos(2i) for i in 1:nc*4prod(dims)], nc, 4, dims...)
        for bc in ((1,1,1,-1), (1,1,1,0))
            x = Initialize_pseudofermion_fields(U[1], "Wilson"; boundarycondition=bc)
            set_values!(x, values)
            @test_throws ArgumentError Dirac_operator(U, old_x, parameters)
            @test_throws ArgumentError Dirac_operator(old, x, parameters)
            @test_throws ArgumentError Dirac_operator(U, x, merge(parameters,Dict("r"=>0.5)))
            @test_throws ArgumentError Dirac_operator(U, x,
                merge(parameters,Dict("method_CG"=>"preconditiond_bicgstab")))
            outputs = Dict{Float64,Array{ComplexF64,6}}()
            for csw in (-1.0, 0.0, 1.0)
                params = merge(parameters, Dict("cSW"=>csw))
                D = Dirac_operator(U, x, params)
                @test D.U === U
                @test D.D isa WilsonDiracCloverOperator4D
                y = similar(x); z = similar(x)
                mul!(y, D, x)
                outputs[csw] = global_values(y)
                tag = string(nc, bc, csw)
                snapshots[tag*"operator"] = pack(outputs[csw])
                mul!(z, D', x)
                @test dot(x,y) ≈ dot(z,x) rtol=3e-12 atol=3e-12
                for method in ("bicg", "bicgstab")
                    solver = Dirac_operator(U, x, merge(params, Dict("method_CG"=>method)))
                    clear_fermion!(z)
                    solve_DinvX!(z, solver, x)
                    mul!(y, solver, z)
                    @test norm(global_values(y)-values)/norm(values) < 2e-11
                    snapshots[tag*method] = pack(global_values(z))
                end
                compat = DO.Wilson_Dirac_operator(U, x,
                    merge(params, Dict("hasclover"=>true)))
                alias = Dirac_operator(U, x,
                    merge(params, Dict("Dirac_operator"=>"Wilson", "hasclover"=>true)))
                for candidate in (compat, alias)
                    mul!(y, candidate, x)
                    @test global_values(y) ≈ outputs[csw] rtol=3e-12 atol=3e-12
                end
                action = FermiAction(D, Dict("Nf"=>2))
                force = [similar(u) for u in U]
                calc_UdSfdU!(force, action, U, x)
                for mu in 1:4
                    snapshots[tag*"force$mu"] = pack(gather_and_bcast_matrix(force[mu].U))
                end
                # Independent Lie-algebra variations at an interior and seam link.
                for (mu,site) in ((1,(2,1,1,2)), (4,(1,1,1,4)))
                    direction = Matrix(im*Diagonal([1.0,-1.0,zeros(nc-2)...]))
                    f = gather_and_bcast_matrix(force[mu].U)[:,:,site...]
                    expected = -2real(tr(((f-f')/2)*direction))
                    actions = Float64[]
                    for sign in (1,-1)
                        changed = copy(originals[mu])
                        changed[:,:,site...] = exp(sign*1e-5*direction)*originals[mu][:,:,site...]
                        substitute!(U[mu].U,LatticeMatrix(changed,4,grid;
                            nw=1,comm0=LDO_TEST_COMM))
                        set_wing_U!(U[mu])
                        push!(actions,evaluate_FermiAction(action,U,x))
                    end
                    @test expected ≈ (actions[1]-actions[2])/2e-5 rtol=3e-4 atol=3e-5
                    substitute!(U[mu].U,LatticeMatrix(originals[mu],4,grid;
                        nw=1,comm0=LDO_TEST_COMM))
                    set_wing_U!(U[mu])
                end
                mul!(y,D,x)
                @test global_values(y) ≈ outputs[csw] rtol=3e-12 atol=3e-12
            end
            W = Dirac_operator(U,x,merge(parameters,Dict("Dirac_operator"=>"Wilson")))
            y = similar(x); mul!(y,W,x)
            @test outputs[0.0] ≈ global_values(y) rtol=3e-12 atol=3e-12
            @test outputs[1.0]+outputs[-1.0] ≈ 2outputs[0.0] rtol=3e-12 atol=3e-12
        end
    end
end
end
