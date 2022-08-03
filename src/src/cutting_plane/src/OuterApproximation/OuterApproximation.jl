abstract type OuterApproximation end

# include("ellipsoid.jl")
# export ellipsoid, addcut!, center, lower_bound

# include("analytic_center.jl")
# export analytic_center, addcut!, prune!, center, lower_bound

# include("rad_analytic_center.jl")
# export rad_analytic_center, addcut!, prune!, center, lower_bound

include("binary_search.jl")
export binary_search, addcut!, prune!, center, lower_bound