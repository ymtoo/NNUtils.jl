"""
    LARS(η = 0.01, ρ = 0.9, β = 0.0001)

[LARS](https://arxiv.org/abs/1708.03888) optimizer.

# Reference
Y. You, I. Gitman and B. Ginsburg, "Large batch training of convolutional networks", 2017.
"""
mutable struct LARS{T}
    eta::T
    rho::T
    coef::T
    beta::T
    velocity::IdDict
end

LARS(η::T = 1f-1, ρ::T = 9f-1, coef::T = 1f-3, beta::T = 0f0, decay::T=1f-3) where {T<:AbstractFloat} = 
    Flux.Optimiser(InvDecay(decay), LARS(η, ρ, coef, beta, IdDict()))
    
function Flux.Optimise.apply!(o::LARS, x, Δ)
    η, ρ, coef, β = o.eta, o.rho, o.coef, o.beta
    v = get!(() -> zero(x), o.velocity, x)::typeof(x)
    λ = coef * norm(x) / (norm(Δ) + β * norm(x))  
    @. v = ρ * v - η * λ * (Δ  - β * x)
    @. Δ = -v
end
