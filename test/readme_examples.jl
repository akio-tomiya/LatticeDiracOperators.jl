module ReadmeExamples

using Test

const README_MARKERS = (
    "# README_V1_WILSON",
    "# README_V1_HISQ_NO_ENZYME",
    "# README_V1_DOMAIN_WALL",
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
    @test length(blocks) == length(README_MARKERS)

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
    end
end

end # module ReadmeExamples
