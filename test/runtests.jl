using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra

@testset "LatticeDiracOperators.jl" begin
    @testset "Basic operations" begin
        include("basic.jl")
        # Write your tests here.
        @test true
    end

    @testset "Wilson HMC" begin
        println("Wilson HMC")
        include("wilsonhmc.jl")
        @test true
    end


    @testset "Staggered HMC" begin
        println("Staggered HMC")
        include("hmc.jl")
        @test true
    end



end
