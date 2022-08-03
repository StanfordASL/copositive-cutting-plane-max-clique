using DrWatson
@quickactivate "copositive-cutting-plane-max-clique"

using COP, cutting_plane

using ArgParse
using Graphs
using JSON
using LinearAlgebra
using Logging
using SparseArrays

function generate_cop_reduced(gx)
    gx_comp = complement(gx)
    n_vertices = nv(gx)
    e = ones(n_vertices)
    M = sparse(-e * e')
    A = adjacency_matrix(gx_comp)
    Ai = [sparse(-(A + I))]
    b = [1.]
    
    return M, Ai, b
end

function recover_graph(cop::COP_problem)
    constr_mat = cop.Ai[1]
    comp_adj = -(constr_mat + I)
    gx_comp = Graph(comp_adj)
    gx = complement(gx_comp)
    
    return gx
end

function recover_or_create(d)
    prefix = datadir("exp_raw", savename(d, "/"; ignores=["ngrid"]), "result")
    if isfile(prefix*"_M.txt") && isfile(prefix*"_constr.txt")
        println("Recovering graph")
        cop = read_COP(prefix)
        gx = recover_graph(cop)
    else
        println("Creating graph")
        gx = erdos_renyi(d["n"], d["p"]) #Generate random graph instance
        cop = COP_problem(generate_cop_reduced(gx)...) #Generate copositive program instance
        write_COP(cop, prefix) #Save cop problem data
    end
    
    return gx, cop
end

function run_cutting_plane(d, args)
    data_dir = args["data_dir"]
    result_dir = datadir(data_dir, savename(d, "/"; ignores=["ngrid"]))
    if !isdir(result_dir)
        mkpath(result_dir)
    end
    prefix = datadir(data_dir, savename(d, "/"; ignores=["ngrid"]), "result")
    
    if args["anst"]
        logfile = datadir(data_dir, savename(d, "/"; ignores=["ngrid"]), "result_cp_log_GT_anst.txt")

        io = open(logfile, "w+")
        logger = SimpleLogger(io)

        gx, cop = recover_or_create(d)

        rad = ceil(2 * sqrt(ne(gx))) #Compute upper bound on max clique
        oa = binary_search(1, rad) #Initialize outer approximation
        CC = anst_milp_CopositiveChecker(size(cop.M, 1)) #Create copositive checker instance
        cop_oracle_anst(y) = oracle(CC, cop, y, logger=logger) #Define oracle function
        cp = cutting_plane_prob(cop.b, :opt, 0., cop_oracle_anst, oa, logger, :Unsolved, cop.b, 0, 1 - 1e-4) 
        cutting_plane.solve(cp)

        flush(io)
    end
    
    if args["neal"]
        logfile = datadir(data_dir, savename(d, "/"; ignores=["ngrid"]), "result_cp_log_GT_neal.txt")
    
        io = open(logfile, "w+")
        logger = SimpleLogger(io)

        gx, cop = recover_or_create(d)

        rad = ceil(2 * sqrt(ne(gx))) #Compute upper bound on max clique
        oa = binary_search(1, rad) #Initialize outer approximation
        nreads = 250
        CC = neal_CopositiveChecker(size(cop.M, 1), nreads) #Create copositive checker instance
        cop_oracle_neal(y) = oracle(CC, cop, y, logger=logger) #Define oracle function
        cp = cutting_plane_prob(cop.b, :opt, 0., cop_oracle_neal, oa, logger, :Unsolved, cop.b, 0, 1 - 1e-4) 
        cutting_plane.solve(cp)

        with_logger(logger) do
            @info "Final outer approximation" oa.lb oa.ub
        end

        flush(io)
    end
        
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "data_dir"
            help = "a positional argument, directory to save data in"
            arg_type = String
            required = true
        "--param"
            help = "parameter file number: parameters{#}.json"
            arg_type = String
            required = true
        "--anst"
            help = "run with anstreicher milp copositive formulation"
            action = :store_true
        "--neal"
            help = "run with dwave-neal copositive checker"
            action = :store_true
    end
    return parse_args(s)
end

#List and run experiments

args = parse_commandline()
exp_dict = JSON.parsefile("../scripts/parameters/parameters"*args["param"]*".json")
for d in [dict_list(exp_dict)[1], dict_list(exp_dict)...] 
    run_cutting_plane(d, args)
end
