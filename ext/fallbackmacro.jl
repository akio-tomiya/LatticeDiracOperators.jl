module EnzymeBFallback

import Enzyme
const ER = Enzyme.EnzymeRules

# --- helpers (runtime) ---

@inline function _fname(fconst)
    f = getproperty(fconst, :val)  # ER.Const{F} has `.val`
    return f isa Function ? string(nameof(f)) : string(f)
end

@inline _primal_type(x) = typeof(x)
@inline _primal_type(x::ER.Annotation) = typeof(getproperty(x, :val))
@inline _primal_type(x::ER.Duplicated) = typeof(getproperty(x, :val))
@inline _primal_type(x::ER.Const) = typeof(getproperty(x, :val))

function _format_argtypes(args...)
    io = IOBuffer()
    for (i, a) in enumerate(args)
        Ta = _primal_type(a)
        print(io, "  ", i, ": ", Ta)
        if a isa Union{ER.Annotation,ER.Duplicated,ER.Const}
            print(io, "    (wrapper=", typeof(a), ")")
        end
        print(io, "\n")
    end
    return String(take!(io))
end

"""
    @gen_enzyme_fallback_for_B BType N

Generate fallback Enzyme rules for `augmented_primal` and `reverse`.
If an `ER.Annotation{<:BType}` appears among the first N differentiation arguments,
and no more specific rule exists, throw an error printing function name and all argument types.
"""
macro gen_enzyme_fallback_for_B(BType, N)
    N isa Integer || error("@gen_enzyme_fallback_for_B: N must be an integer literal")
    n = Int(N)

    B = esc(BType)

    # arg1, arg2, ..., argn
    arg_syms = [Symbol("arg", i) for i in 1:n]

    defs = Expr[]

    for pos in 1:n
        # ---- augmented_primal args (Symbol or Expr)
        ap_sig_args = Any[]
        for i in 1:n
            s = arg_syms[i]
            if i == pos
                push!(ap_sig_args, :($(s)::ER.Annotation{<:$B}))
            else
                push!(ap_sig_args, s)   # <-- Symbol
            end
        end

        # ---- reverse args (Symbol or Expr)
        rv_sig_args = Any[]
        for i in 1:n
            s = arg_syms[i]
            if i == pos
                push!(rv_sig_args, :($(s)::ER.Annotation{<:$B}))
            else
                push!(rv_sig_args, s)   # <-- Symbol
            end
        end

        # tuple (arg1, arg2, ..., argn, rest...)
        all_tuple = Expr(:tuple, arg_syms..., :(rest...))

        ap = quote
            function ER.augmented_primal(cfg::ER.RevConfig,
                f::ER.Const{F},
                ::Type{RT},
                $(ap_sig_args...),
                rest...
            ) where {F,RT}
                fn = EnzymeBFallback._fname(f)
                all = $all_tuple
                msg = "No Enzyme rule implemented for function: " * fn * "\n" *
                      "B was detected (fallback matched at position $(pos)).\n" *
                      "Argument primal types:\n" *
                      EnzymeBFallback._format_argtypes(all...)
                error(msg)
            end
        end

        rv = quote
            function ER.reverse(cfg::ER.RevConfig,
                f::ER.Const{F},
                dout, tape,
                $(rv_sig_args...),
                rest...
            ) where {F}
                fn = EnzymeBFallback._fname(f)
                all = $all_tuple
                msg = "No Enzyme reverse rule implemented for function: " * fn * "\n" *
                      "B was detected (fallback matched at position $(pos)).\n" *
                      "Argument primal types:\n" *
                      EnzymeBFallback._format_argtypes(all...)
                error(msg)
            end
        end

        push!(defs, ap, rv)
    end

    return Expr(:block, defs...)
end

end # module