module WilsonMPILatticeBoundaryTests
using Test, LinearAlgebra, Gaugefields, LatticeDiracOperators, LatticeMatrices, JACC
JACC.@init_backend
include("test_communicator.jl")
const DO = LatticeDiracOperators.Dirac_operators
const snapshots = Dict{String,Vector{Float64}}()
pack(a) = vcat(vec(real.(a)), vec(imag.(a)))

# Independent cold-link stencil with explicit global boundary crossings.
function reference(source, gamma, bc; dagger=false, kappa=0.1)
    out = copy(source)
    dims = size(source)[3:6]
    for site in CartesianIndices(dims), mu in 1:4
        pos = Tuple(site)
        plus = ntuple(d -> d == mu ? mod1(pos[d]+1,dims[d]) : pos[d],4)
        minus = ntuple(d -> d == mu ? mod1(pos[d]-1,dims[d]) : pos[d],4)
        fp = pos[mu] == dims[mu] ? bc[mu] : 1
        fm = pos[mu] == 1 ? bc[mu] : 1
        g = dagger ? -gamma[:,:,mu] : gamma[:,:,mu]
        for color in 1:size(source,1)
            out[color,:,pos...] .-= kappa .* (
                fp .* ((I-g)*source[color,:,plus...]) +
                fm .* ((I+g)*source[color,:,minus...]))
        end
    end
    out
end

@testset "Wilson MPILattice boundary contract" begin
    nprocs = ldo_test_comm_size()
    grid = (1,1,1,nprocs)
    dims = (4,2,2,4)
    U = Initialize_Gaugefields(3,1,dims...; condition="cold",
        isMPILattice=true, PEs=grid, verbose_level=0)
    source_values = reshape([sin(i)+im*cos(2i) for i in 1:12prod(dims)],3,4,dims...)
    default = Initialize_pseudofermion_fields(U[1],"Wilson")
    @test default.f.phases == [1,1,1,-1]
    @test_throws AssertionError DO.Initialize_WilsonFermion(U[1];nowing=false)
    for bc in ([1,1,1,-1],[1,1,1,1],[-1,1,1,-1],[1,1,1,0],[0,1,1,0])
        x = Initialize_pseudofermion_fields(U[1],"Wilson";boundarycondition=bc)
        @test x.f.phases == bc
        @test similar(x).f.phases == bc
        substitute!(x.f, LatticeMatrix(source_values,4,grid;
            nw=1, phases=x.f.phases, comm0=x.f.comm))
        set_wing_fermion!(x)
        other_bc = copy(bc); other_bc[4] = bc[4] == 1 ? -1 : 1
        other = Initialize_pseudofermion_fields(U[1],"Wilson";boundarycondition=other_bc)
        for family in ("Wilson","WilsonClover")
            parameters = Dict{String,Any}("Dirac_operator"=>family,"κ"=>0.1,
                "cSW"=>1.17,"verbose_level"=>0,"eps_CG"=>1e-24)
            unchanged = deepcopy(parameters)
            D = Dirac_operator(U,x,parameters)
            @test D.boundarycondition == bc
            @test parameters == unchanged
            @test_throws ArgumentError Dirac_operator(U,x,merge(parameters,Dict("boundarycondition"=>other_bc)))
            @test_throws ArgumentError Dirac_operator(U,x,merge(parameters,Dict("boundarycondition"=>[1,1])))
            @test_throws ArgumentError Dirac_operator(U,x,merge(parameters,Dict("boundarycondition"=>[1,1,1,2])))
            explicit_bc = copy(bc)
            explicit = Dirac_operator(U,x,merge(parameters,Dict("boundarycondition"=>explicit_bc)))
            explicit_bc[4] = bc[4] == 1 ? -1 : 1
            @test explicit.boundarycondition == bc
            normal = DdagD_operator(U,x,parameters)
            expected = reference(source_values,D.γ,bc)
            for (label, op, oracle) in (
                ("forward",D,expected),
                ("adjoint",D',reference(source_values,D.γ,bc;dagger=true)),
                ("normal",normal,reference(expected,D.γ,bc;dagger=true)),
                ("rebuilt",D(U),expected))
                y = similar(x)
                mul!(y,op,x)
                actual = gather_and_bcast_matrix(y.f)
                @test actual ≈ oracle rtol=5e-13 atol=5e-13
                snapshots[family*string(bc)*label] = pack(actual)
                before = copy(y.f.A)
                @test_throws ArgumentError mul!(y,op,other)
                @test Array(y.f.A) == Array(before)
                @test_throws ArgumentError mul!(other,op,x)
                @test_throws ArgumentError solve_DinvX!(y,op,other)
                @test !any(D._temporary_fermi._data._flagusing)
            end
            for solver in (DO.bicg,DO.bicgstab)
                y = similar(x); clear_fermion!(y)
                solver(y,D,x;eps=1e-24,verbose=Verbose_print(0))
                solution = gather_and_bcast_matrix(y.f)
                @test norm(reference(solution,D.γ,bc)-source_values)/norm(source_values) < 1e-10
                snapshots[family*string(bc)*string(solver)] = pack(solution)
            end
            y = similar(x); clear_fermion!(y)
            solve_DinvX!(y,normal,x)
            solution = gather_and_bcast_matrix(y.f)
            @test norm(reference(reference(solution,D.γ,bc),D.γ,bc;dagger=true)-source_values)/norm(source_values)<1e-10
            @test !any(D._temporary_fermion_forCG._data._flagusing)
        end
    end
    @test snapshots["Wilson[1, 1, 1, 1]forward"] != snapshots["Wilson[1, 1, 1, -1]forward"]
end
end
