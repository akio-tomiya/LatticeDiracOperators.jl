using LatticeDiracOperators
using Gaugefields
using Test
using LinearAlgebra
import LatticeMatrices

@testset "LatticeDiracOperators.jl" begin

    include("public_api.jl")

    @testset "Solver diagnostics" begin
        include("solver_diagnostics.jl")
    end

    @testset "Wilson boundary conditions" begin
        include("wilson_boundary_conditions.jl")
    end

    include("readme_examples.jl")

    include("z4_noise.jl")
    include("pseudofermion_md_action.jl")
    include("mpi_optional.jl")

    if isdefined(LatticeMatrices, :D5DW_GeneralizedDomainwallOperator5D)
        include("latticematrices_backend.jl")
        include("wilson_clover_wrapper.jl")
        include("wilson_lm_callback_ad.jl")
        include("staggered_mpialattice.jl")
        withenv("LDO_TEST_EXPECT_NO_ENZYME" => "false") do
            include("hisq_mpialattice.jl")
        end
        include("hisq_hmc_example.jl")
        include("domainwall_mpialattice.jl")
        include("domainwall_grid_reference.jl")
    end

    @testset "Wilson HMC" begin
        println("Wilson HMC")
        include("wilsonhmc.jl")
        @test true
    end




    @testset "Domainwall" begin
        println("Domainwall")
        include("domainwalltest.jl")
        @test true
    end
    #@testset "MobiusDomainwall" begin
    #    println("MobiusDomainwall")
    #    include("hmcmobiusdomain.jl")
    #    @test true
    #end



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





    @testset "Basic operations" begin
        include("basic.jl")
        # Write your tests here.
        @test true
    end
end
