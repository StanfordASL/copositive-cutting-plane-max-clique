module cutting_plane
using Dates
using Logging
using LinearAlgebra
using JuMP, Gurobi
using Printf

include("utils.jl")
export Halfspace

include("OuterApproximation/OuterApproximation.jl")

mutable struct cutting_plane_prob
    obj::AbstractVector #Objective function
    sol_type::Symbol #Solution type: feasibility / optimization form
    ub::Number #Upper bound: only relevant for sol_type == :feas
    oracle::Function #Oracle function
    oa::OuterApproximation #Outer approximation representation
    logger::AbstractLogger
    sol_status::Symbol 
    sol::AbstractVector #Optimal Solution
    rtol::Number #Optimal tolerance, only relevant for sol_type == :opt relative tol
    atol::Number # absolute tol
end

function cutting_plane_prob(obj::AbstractVector, sol_type::Symbol, ub::Number, oracle::Function, oa::OuterApproximation, logger::AbstractLogger, sol_status::Symbol, sol::AbstractVector, rtol::Number)
    return cutting_plane_prob(obj, sol_type, ub, oracle, oa, logger, sol_status, sol, rtol, 0.)
end

export cutting_plane_prob
include("solve.jl")
export solve



end