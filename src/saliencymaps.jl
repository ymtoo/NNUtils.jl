# function jacobian(f, x)
#     y = f(x)
#     n = length(y)
#     m = length(x)
#     T = eltype(y)
#     J = Array{T,2}(undef, n, m)
#     for i in 1:n
#         J[i, :] .= gradient(x -> f(x)[i], x)[1] |> vec
#     end
#     return J
# end

function _saliencymap(δf, s)
    δf |> 
    x -> abs.(x) |>
    x -> maximum(x; dims=1) |>
    vec |>
    x -> reshape(x, s...) #|>
    #x -> mean(x; dims=3) |>
    #x -> ndims(x) == 4 ? dropdims(x; dims=3) : x
end

abstract type SaliencyMap end

struct Gradient <: SaliencyMap end

struct SmoothGradient{T<:Real} <: SaliencyMap 
    n::Int
    noiselevel::T
end

function saliencymap(::Gradient, f, x)
    #d1, d2, d3 = size(x,1), size(x,2), size(x,3)
    #f(x) = x |> reshape(x, d1 * d2 * d3, :) |> g
    δf = jacobian(f, x)[1]
    _saliencymap(δf, size(x)[1:end-1])
end

function saliencymap(sm::SmoothGradient, f, x)
    #d1, d2, d3 = size(x,1), size(x,2), size(x,3)
    #f(x) = x |> reshape(x, d1 * d2 * d3, :) |> g
    δf = jacobian(f, x)[1]
    σ = sm.noiselevel * (maximum(x) - minimum(x)) / 100
    for i ∈ 2:sm.n
        δf += jacobian(f, x .+ σ .* randn.(eltype(δf)))[1]
    end
    δf /= sm.n
    _saliencymap(δf, size(x)[1:end-1])
end