mutable struct binary_search <: OuterApproximation
    lb::Number
    ub::Number
end

### Constructor ###
# No need since interface for binary_search is short

### Update ###
function addcut!(bs::binary_search, hs::Halfspace)
    if isapprox(hs.slope[1], 0.; atol=1e-15, rtol=0.)
        return
    end
    if hs.slope[1] >= 0.
        bs.ub = min(bs.ub, hs.constant) # slope is already normalized in the construction of hs
    else
        bs.lb = max(bs.lb, -hs.constant)
    end
end

### Compute Center ###
function center(bs::binary_search)
    return [(bs.lb + bs.ub) / 2.]
end


### Lower bound ###
function lower_bound(bs::binary_search, obj::AbstractVector)
    if obj[1] >= 0.
        return bs.lb * obj[1]
    else
        return bs.ub * obj[1]
    end
end

function upper_bound(bs::binary_search, obj::AbstractVector)
    if obj[1] >= 0.
        return bs.ub * obj[1]
    else
        return bs.lb * obj[1]
    end
end