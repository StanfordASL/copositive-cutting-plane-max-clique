mutable struct anst_milp_CopositiveChecker <: CopositiveChecker
    side_dim::Int
    model::Model
    
    x::Array{VariableRef, 1}
    y::Array{VariableRef, 1}
    gamma::VariableRef
    
    A_constr::Array{ConstraintRef, 1}
    A::Matrix{Float64}
    
    
    function anst_milp_CopositiveChecker(sd::Int)
        model = direct_model(Gurobi.Optimizer())
        set_optimizer_attribute(model, "Threads", 1)
        set_silent(model)
        
        x = @variable(model, 0. <= x[1 : sd])
        y = @variable(model, y[1 : sd], Bin)
        gamma = @variable(model, gamma >= 0)
        
        @constraint(model, x .<= y)
        @constraint(model, sum(y) >= 1)
        A_constr = @constraint(model, x .+ gamma - (1 .- y) .<= 0)
        
        @objective(model, Max, gamma)
        new(sd, model, x, y, gamma, A_constr, diagm(ones(sd)))
    end
end

function testcopositive(A::AbstractMatrix, amCC::anst_milp_CopositiveChecker,
        logger::AbstractLogger=ConsoleLogger(stderr, Logging.LogLevel(1)))
    amCC.A  = A
    offdiagA = A - Diagonal(A)
    m = ones(amCC.side_dim) + sum(clamp.(offdiagA, 0., Inf), dims = 2)
    
    for i = 1 : amCC.side_dim
        set_normalized_coefficient(amCC.A_constr[i], amCC.y[i], m[i])
        set_normalized_rhs(amCC.A_constr[i], m[i])
        for j = 1 : amCC.side_dim
            set_normalized_coefficient(amCC.A_constr[i], amCC.x[j], A[i, j])
        end
    end
    
    start_time = time()
    optimize!(amCC.model)
    end_time = time()
    
    x = value.(amCC.x)
    obj = x' * amCC.A * x
    
    with_logger(logger) do
        @info "Copositivity check optimum " obj x start_time end_time
    end
    return (obj, x)
end

    
