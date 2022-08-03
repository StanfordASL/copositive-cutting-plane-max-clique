const vol_tol = 1e-12

mutable struct ellipsoid <: OuterApproximation
    x::Vector{Float64}
    D::Matrix{Float64}
    n::Int64
#     empty::Bool=false
end

### Constructors ###
# radius + dim
function ellipsoid(r::Number, n::Int64)
    x = zeros(n)
    D = Matrix(r*I,n,n)
    
    return ellipsoid(x, D, n)
end

# linear system


### Update ###
function addcut!(E::ellipsoid, hs::Halfspace)
    a = -hs.slope
    anorm = a / sqrt(a' * E.D * a)
    Danorm = E.D * anorm / (E.n + 1)
    E.x .+= Danorm
    E.D .*= E.n^2/(E.n^2-1)
    E.D .+= - 2 * E.n ^ 2/(E.n - 1) * Danorm*Danorm'
    
        
    if !issymmetric(E.D)
        E.D = 0.5*(E.D + E.D')
    end
end

### Compute Center ###
function center(E::ellipsoid)
    if logdet(E.D) <= log(vol_tol)
        return false
    end
    return E.x
end

### Lower Bound ###
function lower_bound(E::ellipsoid, obj::AbstractVector)
    return obj' * E.x - sqrt(obj' * E.D * obj)
end

### Check emptiness ###
function is_empty(E::ellipsoid)
    error("Not implemented")
    return false
end

# """
#     ellipsoid_method(obj::AbstractVector, oracle::Function, r::Number)
# Run an ellipsoid method to minimize `dot(obj, x)` such that `norm(x) ≤ r`
# and `oracle(x) == true`. `oracle(x)` should either return `true` or a `Halfspace`.
# """
# function ellipsoid_method(obj::AbstractVector, oracle::Function, r::Number;
# opttol=1e-6, verbose=true, trackcalls=false)
#     n = length(obj)
#     if n <= 1
#         error("The number of variables $n should be 2 or more.")
#     end
#     nobj = obj / norm(obj)
#     trackcalls && (oraclecalls = 0)
#     x = zeros(n)
#     D = Matrix(r*I,n,n)
#     # The ellipsoid we consider is {y: (y-x)' D^-1 (y-x) ≤ 1}
#     while true
#         # Add the constraint y: a'y >= a'x, i.e. -a'y <= -a'x
#         if norm(x) > r
#             a = -x / norm(x)
#         else
#             result = oracle(x)
#             trackcalls && (oraclecalls += 1)
#             if result isa Halfspace # x is infeasible
#                 a = -result.slope
#             else # x is feasible
#                 # Using z = D^(-1/2)(y-x), min{obj' y: (y-x)' D^-1 (y-x) ≤ 1} =
#                 # min{obj' (x + D^(1/2) z): z'z ≤ 1}
#                 lb = obj' * x - sqrt(obj' * D * obj)
#                 relgap = (obj' * x - lb) / (1 + min(abs(obj' * x), abs(lb)))
#                 verbose && @printf("\rAbs. gap: %.3e\tRel. gap: %.3e", obj' * x - lb, relgap)
#                 if relgap <= opttol
#                     verbose && println()
#                     return trackcalls ? (x, oraclecalls) : x
#                 else
#                     a = -nobj
#                 end
#             end
#         end
#         # See Theorem 8.1 in "Introduction to Linear Optimization" by Bertsimas
#         # and Tsitsiklis.
#         try
#             anorm = a / sqrt(a' * D * a)
#             Danorm = D*anorm / (n+1)
#             x .+= Danorm
#             D .*= n^2/(n^2-1)
#             D .+= - 2 * n^2/(n-1) * Danorm*Danorm'
#             # x = x + (D*a)/(sqrt(a'*D*a) * (n+1))
#             # D = n^2/(n^2-1) * (D - 2 * D*a*a'*D/(a'*D*a * (n+1)))
#         catch e
#             println("Numerical error: cannot compute the next ellipsoid. "*
#             "The spectrum of the ellipsoid matrix follows.")
#             display(eigen(D).values)
#             rethrow(e)
#         end
#         if !issymmetric(D)
#             D = 0.5*(D + D')
#         end
#     end
# end

