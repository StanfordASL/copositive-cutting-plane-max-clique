mutable struct neal_CopositiveChecker <: CopositiveChecker
    side_dim::Int
    sampler::PyObject
    n_reads::Int
    A::Matrix{Float64}

    function neal_CopositiveChecker(sd::Int, n_reads::Int)
        sampler = neal.SimulatedAnnealingSampler()
        A = zeros(sd, sd)
        new(sd, sampler, n_reads, A)
    end
end

function process_neal(res::PyObject)
    min_obj, min_idx = findmin([read[2] for read in res.record])
    min_x = res.record[min_idx][1]
    return (min_obj, min_x)
end

function get_temps_qubo(Q::AbstractMatrix)
    J = (Q - Diagonal(Q)) / 4.
    h = Q * ones(size(J)[1]) / 2.
    return get_temps_ising(J, h)
end

function get_temps_ising(J::AbstractMatrix, h::AbstractVector)
    min_temp = minimum([abs.(h[h .!= 0])..., 2 .* abs.(J[J .!= 0.])...])
    max_temp = maximum(2 * sum(abs.(J), dims = 2) + abs.(h))
    
    return min_temp, max_temp
end

function testcopositive(A::AbstractMatrix, nCC::neal_CopositiveChecker,
    logger::AbstractLogger=ConsoleLogger(stderr, Logging.LogLevel(1)))
    nCC.A = A
    phot = 0.5
    pcold = 0.01

    sweeps = 100
    
    min_temp, max_temp = get_temps_qubo(A)
    min_temp /= log(1. / pcold)
    max_temp /= log(1. / phot)
    
    start_time = time()
    res = nCC.sampler.sample_qubo(A, beta_range = (1/max_temp, 1/min_temp), 
        num_reads=nCC.n_reads, num_sweeps=sweeps)
    end_time = time()
    
    obj, x = process_neal(res)
    
    with_logger(logger) do
        @info "Copositivity check optimum " obj x start_time end_time
    end
    
    return (obj, x)
end

