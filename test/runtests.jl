using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra

@testset "LatticeDiracOperators.jl" begin
    @testset "Gamma5 operations" begin
        include("gamma5.jl")
    end
    
    @testset "Staggered HMC" begin
        include("hmc.jl")
        @test true
    end

    @testset "Wilson HMC" begin
        include("wilsonhmc.jl")
        @test true
    end




    

    @testset "Basic operations" begin
        include("basic.jl")
    # Write your tests here.
        @test true
    end
end
