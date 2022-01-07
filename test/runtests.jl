using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra

@testset "LatticeDiracOperators.jl" begin
    #=
    @testset "HMC" begin
        include("hmc.jl")
        @test true
    end
    =#

    @testset "Basic operations" begin
        include("basic.jl")
    # Write your tests here.
        @test true
    end
end
