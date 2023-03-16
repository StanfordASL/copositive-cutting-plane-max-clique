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


