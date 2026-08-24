module ReadmeExamples

using Test

const README_MARKERS = (
    "# README_V1_WILSON",
    "# README_V1_HISQ",
    "# README_V1_DOMAIN_WALL",
)

const README_HMC_COMMON = "# README_V1_HMC_COMMON"

const README_HMC_MARKERS = (
    "# README_V1_HMC_THIN_CONVENTIONAL",
    "# README_V1_HMC_THIN_DRIVER",
    "# README_V1_HMC_STOUT_CONVENTIONAL",
    "# README_V1_HMC_STOUT_DRIVER",
)

function contains_parse_error(expr)
    expr isa Expr || return false
    expr.head in (:error, :incomplete) && return true
    return any(contains_parse_error, expr.args)
end

function readme_julia_blocks()
    readme = read(joinpath(@__DIR__, "..", "README.md"), String)
    return [
        match.captures[1]
        for match in eachmatch(r"```julia\s*\n(.*?)```"s, readme)
    ]
end

function readme_block_containing(blocks, marker)
    matches = filter(block -> occursin(marker, block), blocks)
    length(matches) == 1 || error(
        "expected one README Julia block containing $(repr(marker)), " *
        "found $(length(matches))",
    )
    return only(matches)
end

@testset "README v1 examples" begin
    blocks = readme_julia_blocks()
    @test length(blocks) ==
          length(README_MARKERS) + length(README_HMC_MARKERS) + 1

    @testset "all Julia blocks parse" begin
        for (index, block) in enumerate(blocks)
            expression = Meta.parseall(
                block; filename="README.md Julia block $index")
            @test !contains_parse_error(expression)
        end
    end

    @testset "all Julia blocks execute without Enzyme" begin
        project_directory = dirname(Base.active_project())
        for marker in README_MARKERS
            block = readme_block_containing(blocks, marker)
            checked_block = block * """

            @assert Base.get_extension(
                LatticeDiracOperators,
                :LatticeDiracOperatorsEnzymeExt,
            ) === nothing
            """
            command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) -e $(checked_block)`
            process = run(ignorestatus(command))
            @test process.exitcode == 0
        end

        common = readme_block_containing(blocks, README_HMC_COMMON)
        hmc_examples = join(
            (
                readme_block_containing(blocks, marker)
                for marker in README_HMC_MARKERS
            ),
            "\n",
        )
        checked_hmc = common * "\n" * hmc_examples * """

        @assert Base.get_extension(
            LatticeDiracOperators,
            :LatticeDiracOperatorsEnzymeExt,
        ) === nothing
        """
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(project_directory) -e $(checked_hmc)`
        process = run(ignorestatus(command))
        @test process.exitcode == 0
    end
end

end # module ReadmeExamples
