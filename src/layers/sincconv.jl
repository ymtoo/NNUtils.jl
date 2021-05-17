import Flux: expand, calc_padding, CuArray
import Functors: @functor

"""
Get time samples for the sinc function.
"""
gett(n) = iseven(n) ? (-(n÷2-1):n÷2) : -(n÷2):n÷2
function gett(n1, n2)
    n1 == 1 && (return gett(n2) |> x -> reshape(x, 1, length(x)))
    n2 == 1 && (return gett(n1) |> x -> reshape(x, length(x), 1))
    t1 = gett(n1) |> x -> reshape(x, n1, 1)
    t2 = gett(n2) |> x -> reshape(x, 1, n2)
    t1 * t2
end

"""
Convolutinal layer with parameterized sinc functions which implement band-pass filters. 

# Reference
Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
"""
struct SincConv{T,D,F,N,M}
    f1s::AbstractArray{T}
    bws::AbstractArray{T}
    fs::T
    dims::D
    σ::F
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end
function SincConv(fs::T, k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, 
    σ=identity; init=initcutofffreqs, stride=1, pad=0, dilation=1) where {T<:Real,N}
    dims = (k..., ch...)
    n = length(dims)
    f1s, bws = init(dims, fs)
    stride = expand(Val(n-2), stride)
    dilation = expand(Val(n-2), dilation)
    pad = calc_padding(Conv, pad, dims[1:n-2], dilation, stride)
    SincConv(f1s, bws, fs, dims, σ, stride, pad, dilation)
end

@functor SincConv

function sincfunctions(f1s::VT, bws::VT, dims::Tuple, fs::T=convert(T, 1)) where {VT<:AbstractArray{<:Real}, T<:Real}
    f1s = abs.(f1s)
    bws = abs.(bws)
    f2s = f1s + bws
    n1, n2 = dims[1:2]
    t = reshape(gett(n1, n2) ./ fs, dims[1:2]..., 1, 1) |> x -> f1s isa CuArray ? gpu(x) : x
    f1srep = reshape(f1s, 1, 1, dims[3:4]...)
    f2srep = reshape(f2s, 1, 1, dims[3:4]...)
    w = 2 .* f2srep .* sinc.(2 .* f2srep .* t) .- 2 .* f1srep .* sinc.(2 .* f1srep .* t)
    w ./ sum(w; dims=1)
end
function initcutofffreqs(rng::AbstractRNG, dims::Tuple, fs::T=convert(T, 1)) where {T<:Real}
    cutoff1 = 0
    cutoff2 = fs / 2
    f1s = zeros(T, dims[3:4]...)
    bws = zeros(T, dims[3:4]...)
    for i ∈ 1:dims[3]
        for j ∈ 1:dims[4]
            f1s[i,j] = rand(rng, Uniform(cutoff1, cutoff2))
            bws[i,j] = rand(rng, Uniform(f1s[i,j], cutoff2)) - f1s[i,j]
        end
    end
    f1s, bws
end
initcutofffreqs(dims::Tuple, fs::T=convert(T, 1)) where {T<:Real} = initcutofffreqs(Random.GLOBAL_RNG, dims, fs)

function (c::SincConv)(x::AbstractArray)
    weight = sincfunctions(c.f1s, c.bws, c.dims, c.fs)
    σ = c.σ
    cdims = DenseConvDims(x, weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    σ.(Flux.conv(x, weight, cdims))
end