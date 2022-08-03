abstract type CopositiveChecker end

    include("anst_milp.jl")
    export anst_milp_CopositiveChecker, testcopositive

    include("neal.jl")
    export neal_CopositiveChecker, testcopositive

#     include("utils.jl")
#     export binary_mask, unary_mask, expand_qubo_binary, expand_qubo_unary, combine_binary, combine_unary, expand_qubo
