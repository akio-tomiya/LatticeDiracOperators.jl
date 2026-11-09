module WilsonCGActionTests
using Test, LinearAlgebra, Gaugefields, LatticeMatrices, LatticeDiracOperators, JACC
JACC.@init_backend
include("test_communicator.jl")
const DO=LatticeDiracOperators.Dirac_operators
const snapshots=Dict{String,Vector{Float64}}()
pack(x)=Float64.(vcat(vec(real.(x)),vec(imag.(x))))
core(x)=gather_and_bcast_matrix(x.f)
@testset "Wilson CG action and deferred-halo solves" begin
    dims=(2,2,2,4); grid=(1,1,1,ldo_test_comm_size())
    U=gauge_configuration(dims;colors=3,halo=1,start=:hot,seed=UInt64(10109),process_grid=grid,comm=LDO_TEST_COMM,verbose=0)
    for bc in ((1,1,1,-1),(1,1,1,0)), kind in ("Wilson","WilsonClover")
        x=Initialize_pseudofermion_fields(U[1],"Wilson";boundarycondition=bc)
        vals=reshape(ComplexF64[sin(i)+im*cos(2i) for i in 1:12prod(dims)],3,4,dims...)
        substitute!(x.f,LatticeMatrix(vals,4,grid;nw=1,phases=bc,comm0=LDO_TEST_COMM))
        set_wing_fermion!(x)
        p=Dict{String,Any}("Dirac_operator"=>kind,"κ"=>0.08,"cSW"=>1.0,"boundarycondition"=>collect(bc),"method_CG"=>"cg","eps_CG"=>1e-24,"MaxCGstep"=>2000,"verbose_level"=>0)
        D=Dirac_operator(U,x,p); action=FermiAction(D,Dict{String,Any}())
        flags=copy(action._temporary_fermionfields._data._flagusing)
        cgvalue=evaluate_FermiAction(action,U,x)
        @test action._temporary_fermionfields._data._flagusing==flags
        @test evaluate_FermiAction(action,U,x) ≈ cgvalue rtol=1e-13
        y=similar(x); clear_fermion!(y)
        normal=DdagD_operator(U,x,p)
        solve_DinvX!(y,normal,x)
        @test cgvalue ≈ real(dot(x,y)) rtol=1e-12
        z=similar(x);mul!(z,normal,y)
        @test norm(core(z)-vals)/norm(vals)<1e-10
        for op in (D,D')
            err=try solve_DinvX!(y,op,x);nothing catch e;e end
            @test err isa ErrorException
            @test occursin("cg is not supported",sprint(showerror,err))
        end
        @test isfinite(cgvalue)
        tag="$kind-$bc"
        snapshots["action-$tag"]=[cgvalue]
        for method in ("bicg","bicgstab")
            solver=Dirac_operator(U,x,merge(p,Dict("method_CG"=>method)))
            @test evaluate_FermiAction(FermiAction(solver,Dict{String,Any}()),U,x) ≈ cgvalue rtol=2e-11
            clear_fermion!(y)
            solve_DinvX!(y,solver,x)
            @test !LatticeMatrices.halo_is_dirty(y.f)
            before=Array(y.f.A);set_wing_fermion!(y)
            @test Array(y.f.A)==before
            mul!(z,solver,y)
            @test norm(core(z)-vals)/norm(vals)<1e-10
            snapshots["solution-$method-$tag"]=pack(core(y))
        end
        bad=Dirac_operator(U,x,merge(p,Dict("MaxCGstep"=>0)))
        failed=FermiAction(bad,Dict{String,Any}());before=copy(failed._temporary_fermionfields._data._flagusing)
        @test_throws ErrorException evaluate_FermiAction(failed,U,x)
        @test failed._temporary_fermionfields._data._flagusing==before
    end
end
end
