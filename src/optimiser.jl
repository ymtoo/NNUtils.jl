"""
    LARS(η = 0.01, ρ = 0.9, β = 0.0001)

[LARS](https://arxiv.org/abs/1708.03888) optimizer.

# Reference
Y. You, I. Gitman and B. Ginsburg, "Large batch training of convolutional networks", 2017.
"""
mutable struct LARS
    eta::Float64
    rho::Float64
    coef::Float64
    beta::Float64
    velocity::IdDict
end

LARS(η::T = 0.1, ρ::T = 0.9, coef::T = 0.001, beta::T = 0.0, decay::T=0.001) where {T<:AbstractFloat} = 
    Flux.Optimiser(InvDecay(decay), LARS(η, ρ, coef, beta, IdDict()))
    
function Flux.Optimise.apply!(o::LARS, x, Δ)
    η, ρ, coef, β = o.eta, o.rho, o.coef, o.beta
    v = get!(() -> zero(x), o.velocity, x)::typeof(x)
    λ = coef * norm(x) / (norm(Δ) + β * norm(x))  
    @. v = ρ * v - η * λ * (Δ  - β * x)
    @. Δ = -v
end
