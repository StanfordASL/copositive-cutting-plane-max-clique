module COP
# using Dates
using Logging
using LinearAlgebra, SparseArrays
using JuMP, Gurobi
using Printf
using cutting_plane
using PyCall

const neal = PyNULL()

function __init__()
    copy!(neal, pyimport("neal"))
end

mutable struct COP_problem
    M::AbstractArray{Float64}
    Ai::Vector{Matrix{Float64}}
    b::Vector{Float64} 
end


export COP_problem

include("cop_checkers/copositivity_checker.jl")
include("utils.jl")
export update!, feasible_y, check_slaters, y2mat, save_sparse_matrix, init_params, oracle, multi_oracle, write_COP, read_COP
    
end
