
function solve(cp::cutting_plane_prob)
    if cp.sol_type == :opt
        stat, x = solve_cutting_plane(cp.obj, cp.oracle, cp.oa, cp.rtol, cp.atol, cp.logger)
        cp.sol_status = Symbol(stat)
        cp.sol = x
        
    elseif cp.sol_type == :feas
        stat, x = solve_cutting_plane(cp.obj, cp.ub, cp.oracle, cp.oa, cp.logger)
        cp.sol_status = Symbol(stat)
        cp.sol = x
    else
        print("Unsupported cutting plane solution type: ", sol_type)
    end
end


function solve_cutting_plane(obj::AbstractVector, ub::Number, oracle::Function, 
        oa::OuterApproximation,logger::AbstractLogger=ConsoleLogger(stderr, Logging.LogLevel(1)))
    best_val = Inf
    best_sol = Vector{Float64}(undef, size(obj, 1))
    iter = 0
    while true
        iter += 1
        x = center(oa)
        
        ### Check if centering fails ###
        # Outer approximation abstraction should return false as center if 
        # the remainding set is effectively empty
        if x == false
            @logmsg LogLevel(1) "Problem is infeasible. Best objective found: " best_val
            return (false, best_sol)
        end
        
        result = oracle(x)
        val = obj' * x 
        
        ### Update Best Val and Best Solution ###
        result == true ? log_iteration(logger, iter, x, val, :feas) : log_iteration(logger, iter, x, val, :infeas)
        if (result == true) && (val <= best_val)
            best_val = val
            best_sol = x
            @logmsg LogLevel(1) "Objective improved. Best objective found: " best_val
        end
        
        ### Check if we are done ###
        if (result == true) & (val <= ub)
            
            return (true, x)
        end
 
        ## Update Outer Approximation. Should we add both cuts here? ###
        if !(result == true)
            if typeof(result) == Halfspace
                @logmsg LogLevel(1) "Adding infeasibility cut: " Num_cuts = 1
            elseif typeof(result) == Vector{Halfspace}
                @logmsg LogLevel(1) "Adding infeasibility cut: " Num_cuts = length(result)
            end
            addcut!(oa, result)
        end
        
        if obj' * x > ub
            @logmsg LogLevel(1) "Adding Objective"
            addcut!(oa, Halfspace(obj, ub))
        end
    end
end

### Optimization Form ### 

function solve_cutting_plane(obj::AbstractVector, oracle::Function, oa::OuterApproximation,
        rtol::Number, atol::Number, logger::AbstractLogger=ConsoleLogger(stderr, Logging.LogLevel(1)))
    best_val = upper_bound(oa, obj)
    best_sol = Vector{Float64}(undef, size(obj, 1))
    stall_counter = 0
    iter = 0
    x_prev = center(oa)
    lb_prev = lower_bound(oa, obj)
    while true
        x = center(oa)
        iter += 1
        ### Check if centering fails ###
        # Outer approximation abstraction should return false as center if 
        # the remainding set is effectively empty
        if x == false
            if best_val == Inf
                @logmsg LogLevel(1) "Problem is infeasible"
                return (false, best_sol)
            else
                @logmsg LogLevel(1) "Best objective found: " best_val
                return (true, best_sol)
            end
        end

        result = oracle(x)
        val = obj' * x
        
        ### Update Best Val and Best Solution ###
        if (result == true) && (val <= best_val)
            best_val = val
            best_sol = copy(x)
            addcut!(oa, Halfspace(obj, best_val))
            @logmsg LogLevel(1) "Best objective found: " best_val
        else
            if typeof(result)== Halfspace
                @logmsg LogLevel(1) "Adding infeasibility cut: " Num_cuts = 1
            elseif typeof(result) == Vector{Halfspace}
                @logmsg LogLevel(1) "Adding infeasibility cut: " Num_cuts = length(result)
            end
            addcut!(oa, result)
        end
        
        
        lb = lower_bound(oa, obj)
        relgap = (best_val - lb) / (1 + min(abs(best_val), abs(lb)))
        
        result == true ? log_iteration(logger, iter, x, val, :feas, relgap) : log_iteration(logger, iter, x, val, :infeas, relgap)
        absgap = best_val - lb
        LogStr = @sprintf("Abs. gap: %.3e\tRel. gap: %.3e", absgap, relgap)
        @logmsg LogLevel(1) LogStr
        if isnan(lb)
            oa.lb = lb_prev
            @logmsg LogLevel(1) "Returning due to NaN: reset LB " lb_prev
            return (false, best_sol)
        end
        if absgap < atol
            @logmsg LogLevel(1) "Absolute gap at return: " absgap
            return (true, best_sol)
        end
        
        if relgap <= rtol
            @logmsg LogLevel(1) "Relative gap at return: " relgap
            return (true, best_sol)
        end
        
        if isapprox(x, x_prev)
            stall_counter += 1
            if stall_counter >= 25
                return (false, best_sol)
            end
        else
            stall_counter = 0
        end

        x_prev = copy(x)
        lb_prev = copy(lb)
    end
end

function log_iteration(logger::AbstractLogger, iter::Int64, y::AbstractVector, obj::Float64, status::Symbol, relgap::Float64)
    curr_time = time()
    with_logger(logger) do
        @info "Cutting-plane iteration information" iter y obj status relgap curr_time
    end
end

function log_iteration(logger::AbstractLogger, iter::Int64, y::AbstractVector, obj::Float64, status::Symbol)
    curr_time = time()
    with_logger(logger) do
        @info "Cutting-plane iteration information" iter y obj status curr_time
    end
end
    

function log_iteration(filename::String, iter::Int64, y::AbstractVector, obj::Float64, status::Symbol, relgap::Float64)
    info_str = string(iter) * ", " * string(status) * ", " * string(obj) *  ", " * string(relgap) * "\n"
    vec_str = join(y, ", ")*"\n"
    
    io = open(filename, "a")
    write(io, info_str)
    write(io, vec_str)
    close(io)
end

function log_iteration(filename::String, iter::Int64, y::AbstractVector, obj::Float64, status::Symbol)
    info_str = string(iter) * ", " * string(status) * ", " * string(obj) * "\n"
    vec_str = join(y, ", ")*"\n"
    
    io = open(filename, "a")
    write(io, info_str)
    write(io, vec_str)
    close(io)
end
       