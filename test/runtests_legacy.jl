using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra

@testset "LatticeDiracOperators legacy long-running tests" begin
    @testset "Wilson HMC" begin
        include("wilsonhmc.jl")
    end

    @testset "Domain-wall actions" begin
        include("domainwalltest.jl")
    end

    @testset "Staggered HMC" begin
        include("hmc.jl")

        @testset "2D HMC" begin
            for colors in (1, 2, 3)
                @testset "NC = $colors" begin
                    test1_2D_NC(colors)
                    @test true
                end
            end
        end

        @testset "4D HMC" begin
            test1_4D()
            @test true
        end
    end

    @testset "Basic operations" begin
        include("basic.jl")
    end
end
