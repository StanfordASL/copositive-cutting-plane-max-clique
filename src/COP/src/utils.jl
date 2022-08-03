function y2mat(cop::COP_problem, y::Vector{Float64})
    nconstr = size(cop.b, 1)
    return cop.M - sum(y[i] * cop.Ai[i] for i in 1:nconstr)
end

## todo Write y2mat ###
function y2mat(cop::COP_problem, y::Vector{Float64}, filename::String)
    nconstr = size(cop.b, 1)
    M = cop.M - sum(y[i] * cop.Ai[i] for i in 1:nconstr)
    save_sparse_matrix(M, filename)

    return M
end



function oracle(cc::CopositiveChecker, cop::COP_problem, y::Vector{Float64}; thresh::AbstractFloat=-1e-8, logger::AbstractLogger=ConsoleLogger(stderr, Logging.LogLevel(1)))
    val, z = testcopositive(y2mat(cop, y), cc, logger)
    if val >= thresh
        # If val ≥ 0,  M - yAi is copositive, so the oracle returns
        # true.
        return true
    elseif isapprox(z, zeros(size(z)))
        return true
    else
        # Otherwise, z ≥ 0 satisfies z'(M - yAi) z < thresh, while any copositive matrix
        # lies in the halfspace
        # {Z: z'(M - yAi) z ≥ 0} = {vec2mat(z): dot(y),  z'Ai z) ≤ y'(M )y}.
        rhs = z' * cop.M * z
        lhs = [z' * Ai * z for Ai in cop.Ai]
        
        return Halfspace(lhs, rhs)
    end
end

function oracle(cop::COP_problem, y::Vector{Float64}; thresh::AbstractFloat=0.)
        # Test if M - yAi is copositive
    cc = CopositiveChecker(size(cop.M, 1))
    val, z = testcopositive(y2mat(cop, y), cc)
    if val >= thresh
        # If val ≥ 0,  M - yAi is copositive, so the oracle returns
        # true.
        return true    
    else
        # Otherwise, z ≥ 0 satisfies z'(M - yAi) z < 0, while any copositive matrix
        # lies in the halfspace
        # {Z: z'(M - yAi) z ≥ 0} = {vec2mat(z): dot(y),  z'Ai z) ≤ y'(M )y}.
        z_outer = z * z'
        rhs = dot(z_outer, cop.M)
        lhs = [dot(z_outer, Ai) for Ai in cop.Ai]
        return Halfspace(lhs, rhs)
    end
end

function multi_oracle(cc::CopositiveChecker, cop::COP_problem, y::Vector{Float64}; thresh::AbstractFloat=0.)
    sols = testcopositive(y2mat(cop, y), cc)
    
    if length(sols) == 0
        return true
    end
    min_val = sols[1][1]
    if min_val >= thresh
        return true 
    else
        halfspaces = Vector{Halfspace}()
        for i = 1:length(sols)
            val = sols[i][1]
            if val <= thresh
                z = sols[i][2]
                rhs = z' * cop.M * z
                lhs = [z' * Ai * z for Ai in cop.Ai]
                push!(halfspaces, Halfspace(lhs, rhs))
            end
        end
        return halfspaces
    end
end

""" The file format for the constraint file will be bi[j] and then the number of non-zeros in Ai[i] and then the row, col, val entries for Ai"""
function write_COP(cop::COP_problem, file_suffix::String)
    M_file = file_suffix*"_M.txt"
    io = open(M_file, "w")
    rows, cols, vals = findnz(cop.M)
    for j in 1:length(rows)
        entry_str = string(rows[j])*", "*string(cols[j])*", "*string(vals[j])*"\n"
        write(io, entry_str)
    end
    close(io)
    
    constr_file = file_suffix*"_constr.txt"
    io = open(constr_file, "w")
    
    for i = 1:length(cop.b)
        Ai_i = cop.Ai[i]
        Ai_i = sparse(Ai_i)
        rows, cols, vals = findnz(Ai_i)
        
        bi_str = string(cop.b[i])*"\n"
        nnz_str = string(length(rows))*"\n"
        
        write(io, bi_str)
        write(io, nnz_str)
        for j in 1:length(rows)
            entry_str = string(rows[j])*", "*string(cols[j])*", "*string(vals[j])*"\n"
            write(io, entry_str)
        end
    end
    close(io)
end

function read_COP(file_suffix::String)
    M_file = file_suffix*"_M.txt"
    constr_file = file_suffix*"_constr.txt"
    
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]

    f = open(M_file, "r")
    while !eof(f)
        r, c, v = split(readline(f), ", ")
        r = parse(Int, r)
        c = parse(Int, c)
        v = parse(Float64, v)

        append!(rows, r)
        append!(cols, c)
        append!(vals, v)
    end
    close(f)

    M = sparse(rows, cols, vals)
    
    bi = Float64[]
    Ai = AbstractMatrix[]

    f = open(constr_file, "r")
    while !eof(f)
        bi_i = parse(Float64, readline(f))
        append!(bi, bi_i)

        nnz = parse(Int64, readline(f))

        rows = Int64[]
        cols = Int64[]
        vals = Float64[]

        for i = 1:nnz
            r, c, v = split(readline(f), ", ")
            r = parse(Int, r)
            c = parse(Int, c)
            v = parse(Float64, v)

            append!(rows, r)
            append!(cols, c)
            append!(vals, v)
        end
        Ai_i = sparse(rows, cols, vals)
        push!(Ai, Ai_i)
    end
    close(f)
    
    cop = COP_problem(M, Ai, bi)
    return cop
end
    
