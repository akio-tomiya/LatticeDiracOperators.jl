module SSGeneralFermionTests
using Test, LinearAlgebra, Random, Gaugefields, LatticeMatrices, LatticeDiracOperators, JACC
JACC.@init_backend
include("test_communicator.jl")
include("shiftedbicg_residuals.jl")
const DO = LatticeDiracOperators.Dirac_operators
const snapshots = Dict{String,Vector{Float64}}()
pack(a) = vcat(vec(real.(a)), vec(imag.(a)))

struct StaggeredOperator{T}
    operator::T
end
function LinearAlgebra.mul!(y::GeneralFermion, D::StaggeredOperator, x::GeneralFermion)
    @assert !LatticeMatrices.halo_is_dirty(x.field)
    mul!(y, D.operator, x)
    set_wing_fermion!(y)
    return y
end
Base.adjoint(D::StaggeredOperator) = StaggeredOperator(adjoint(D.operator))

@testset "GeneralFermion global vector interoperability" begin
    dims = (2,2,2,4)
    grid = (1,1,1,ldo_test_comm_size())
    for nw in (0,1), columns in (1,2)
        x = GeneralFermion(3,columns,dims,grid; nw, phases=(1,1,1,-1), comm0=LDO_TEST_COMM)
        n = 3columns*prod(dims)
        @test length(x) == n
        values = ComplexF64[sin(i)+im*cos(2i) for i in 1:n]
        @test substitute_fermion!(x,values) === x
        @test convert_to_normalvector(x) == values
        for i in (1,n÷2,n)
            @test x[i] == values[i]
            x[i] = -2+3im
            values[i] = -2+3im
        end
        set_wing_fermion!(x)
        @test convert_to_normalvector(x) == values
        @test_throws BoundsError x[0]
        @test_throws BoundsError x[n+1]
        @test_throws BoundsError (x[n+1] = 0)
        @test_throws MethodError (x[n] = "invalid scalar")
        @test convert_to_normalvector(x) == values
        @test_throws DimensionMismatch substitute_fermion!(x,values[1:end-1])
        y = similar(x); substitute_fermion!(y,values)
        @test axpy!(2im,x,y) === y
        @test convert_to_normalvector(y) ≈ (1+2im).*values
        @test dot(x,y) ≈ dot(values,(1+2im).*values)
        snapshots["vector-nw$(nw)-columns$(columns)"] = pack(convert_to_normalvector(y))
        DO._ss_axpy!(1,x,y)
        @test LatticeMatrices.halo_is_dirty(y.field) == (nw > 0)
        @test convert_to_normalvector(y) ≈ (2+2im).*values
        DO._ss_prepare!(y)
        @test !LatticeMatrices.halo_is_dirty(y.field)
    end
end

@testset "Sakurai-Sugiura with distributed GeneralFermion" begin
    dims = (2,2,2,4)
    grid = (1,1,1,ldo_test_comm_size())
    mass = 0.3
    U = gauge_configuration(dims; colors=3, halo=1, start=:hot,
        seed=UInt64(0x53534746), process_grid=grid, comm=LDO_TEST_COMM, verbose=0)
    x = GeneralFermion(3,1,dims,grid; nw=1, phases=(1,1,1,-1), comm0=LDO_TEST_COMM)
    D = StaggeredOperator(StaggeredDiracOperator4D(getproperty.(U,:U),mass))
    n = length(x)
    dense = zeros(ComplexF64,n,n)
    unit, image = similar(x), similar(x)
    for j in 1:n
        v = zeros(ComplexF64,n); v[j] = 1
        substitute_fermion!(unit,v)
        mul!(image,D,unit)
        dense[:,j] = convert_to_normalvector(image)
    end
    values = eigvals(dense)
    rhs = ComplexF64[sin(i)+im*cos(i) for i in 1:n]
    substitute_fermion!(unit,rhs)
    shifts = [0.1+0.2im,-0.2+0.3im]
    solutions = DO.shiftedbicg_2003(shifts,D,unit;eps=1e-22,verboselevel=0)
    for (i,(shift,solution)) in enumerate(zip(shifts,solutions))
        @test !LatticeMatrices.halo_is_dirty(solution.field)
        v = convert_to_normalvector(solution)
        @test norm((dense+shift*I)*v-rhs)/norm(rhs) < 1e-10
        snapshots["shifted-solution$(i)"] = pack(v)
    end
    origin = mass + 0im
    distances = sort(abs.(values .- origin))
    k = argmax((distances[5:17] .- distances[4:16]) ./ distances[4:16]) + 3
    radius = sqrt(distances[k]*distances[k+1])
    inside = sort(filter(z -> abs(z-origin)<radius,values); by=imag)
    rng = MersenneTwister(0x53534746)
    z2!(y) = substitute_fermion!(y,ComplexF64.(rand(rng,(-1.0,1.0),length(y))))
    gaussian!(y) = substitute_fermion!(y,randn(rng,ComplexF64,length(y)))
    @test_throws ArgumentError eigensystem(D,x,radius,origin,z2!,gaussian!; residual_tolerance=0)
    @test_throws ArgumentError eigensystem(D,x,radius,origin,z2!,gaussian!; δ=0)
    found,residuals,vectors,count = eigensystem(D,x,radius,origin,z2!,gaussian!;
        L0=8,Nq=64,M=8,κ=2,MaxCGstep=3000,epsCG=1e-22,δ=1e-10,
        residual_tolerance=1e-4)
    @test count == length(inside)
    @test sort(found; by=imag) ≈ inside rtol=1e-5 atol=1e-6
    for (i,(value,residual,vector)) in enumerate(zip(found,residuals,vectors))
        v = convert_to_normalvector(vector)
        @test norm(dense*v-value*v)/norm(v) ≈ real(residual) rtol=1e-5 atol=1e-11
        @test real(residual) < 1e-4
        # Remove arbitrary eigenvector phases before comparing backends.
        v ./= norm(v)
        v .*= conj(v[argmax(abs.(v))]) / abs(v[argmax(abs.(v))])
        ordinal = findfirst(==(i), sortperm(found; by=imag))
        snapshots["eigenvector$(ordinal)"] = pack(v)
    end
    snapshots["eigenvalues"] = pack(sort(found; by=imag))
    snapshots["dense-operator"] = pack(dense)
end

struct DiagonalOperator end
Base.adjoint(D::DiagonalOperator) = D
@inline function diagonal_kernel!(i,y,x,offset,nw)
    g = i + offset
    value = g == 4 ? 1.2 : 0.1g
    @inbounds y[1,1,i+nw] = value*x[1,1,i+nw]
end
function LinearAlgebra.mul!(y::GeneralFermion, ::DiagonalOperator, x::GeneralFermion)
    f = x.field
    JACC.parallel_for(f.PN[1],diagonal_kernel!,y.field.A,f.A,f.coords[1]*f.PN[1],f.nw)
    LatticeMatrices.mark_halo_dirty!(y.field)
    set_wing_fermion!(y)
    return y
end

@testset "Sakurai-Sugiura default residual tolerance" begin
    x = GeneralFermion(1,1,(4,),(ldo_test_comm_size(),); nw=1, phases=(1,),comm0=LDO_TEST_COMM)
    rng = MersenneTwister(189)
    z2!(y) = substitute_fermion!(y,ComplexF64.(rand(rng,(-1.0,1.0),length(y))))
    gaussian!(y) = substitute_fermion!(y,randn(rng,ComplexF64,length(y)))
    found,residuals,vectors,count = eigensystem(DiagonalOperator(),x,0.5,0.,z2!,gaussian!;
        L0=4,Nq=32,M=2,epsCG=1e-24,δ=1e-10)
    @test count == 3
    @test sort(real.(found)) ≈ [0.1,0.2,0.3] atol=1e-9
    @test all(r -> real(r) < 1e-8,residuals)
    for (lambda,vector) in zip(found,vectors)
        v = convert_to_normalvector(vector)
        @test norm([0.1,0.2,0.3,1.2].*v-lambda*v)/norm(v) < 1e-8
    end
    snapshots["diagonal-eigenvalues"] = pack(sort(found;by=real))
end
end
