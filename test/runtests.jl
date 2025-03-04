using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra

@testset "LatticeDiracOperators.jl" begin
    @testset "Domainwall" begin
        println("Domainwall")
        include("domainwalltest.jl")
        @test true
    end


    @testset "Staggered HMC" begin
        println("Staggered HMC")
        include("hmc.jl")

        @testset "2D HMC " begin
            #println("2D HMC ")
            @testset "NC = 1" begin
                println("NC = 1")
                NC = 1
                test1_2D_NC(NC)
                @test true
            end

            @testset "NC = 2" begin
                #println("NC = 2")
                NC = 2
                test1_2D_NC(NC)
                @test true
            end
            @testset "NC = 3" begin
                #println("NC = 3")
                NC = 3
                test1_2D_NC(NC)
                @test true
            end

        end

        @testset "4D HMC " begin
            println("4D HMC ")
            test1_4D()
            @test true
        end

        @test true
    end

    @testset "Wilson HMC" begin
        println("Wilson HMC")
        include("wilsonhmc.jl")
        @test true
    end





    @testset "Basic operations" begin
        include("basic.jl")
        # Write your tests here.
        @test true
    end
end
