"""
    Halfspace(slope, constant)
Define a halfspace `{x: dot(slope, x) ≤ constant}`. Note that slope is internally
normalized.
"""
struct Halfspace{Ts<:AbstractVector,Tc<:Number}
    slope::Ts
    constant::Tc
    function Halfspace(slope, constant)    
        norm_slope = slope/LinearAlgebra.norm(slope)
        norm_constant = constant/LinearAlgebra.norm(slope)
        new{typeof(norm_slope),typeof(norm_constant)}(norm_slope, norm_constant)
    end
end

