import DSP: hamming
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
    SincConv(fs, filter, in => out, σ=identity; stride=1, pad=0, dilation=1, init=initcutofffreqs)

Convolutinal layer with parameterized sinc functions which implement band-pass filters. `fs` is the
sampling rate. For more details, you can refer to https://github.com/FluxML/Flux.jl/blob/master/src/layers/conv.jl 

# Examples
```julia-repl
julia> using Flux, NNUtils

julia> fs = 9600f0
9600.0f0

julia> model = SincConv(fs, (200, 1), 1=>8, identity)
SincConv(9600.0, (200, 1), 1=>8)

julia> params(model) |> length
2

julia> params(model)[1] |> size
(1, 8)

julia> params(model)[2] |> size
(1, 8)

julia> x = randn(Float32, 4800, 1, 1, 16)
4800×1×1×16 Array{Float32, 4}:
[:, :, 1, 1] =
 1.3832613
 0.42255098
 ⋮
 0.3134887

[:, :, 1, 2] =
 2.0712132
 0.13467419
 ⋮

 julia> model(x)
 4601×1×8×16 Array{Float32, 4}:
 [:, :, 1, 1] =
   -3498.0605
  -17983.041
   20055.25
   -1296.112
       ⋮
    1844.9044
   10672.3125
  -12256.136
 
 [:, :, 2, 1] =
    521.0923
   8794.186
   4179.6655
  -4084.1067
      ⋮
```

# Reference
Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” Arxiv
"""
struct SincConv{T,D,F,N,M}
    f1s::AbstractArray{T}
    f2s::AbstractArray{T}
    fs::T
    dims::D
    σ::F
    stride::NTuple{N,Int}
    pad::NTuple{M,Int}
    dilation::NTuple{N,Int}
end
function SincConv(fs::T, k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, 
    σ=identity; init=initmelcutofffreqs, stride=1, pad=0, dilation=1) where {T<:Real,N}
    dims = (k..., ch...)
    n = length(dims)
    f1s, f2s = init(dims, fs)
    stride = expand(Val(n-2), stride)
    dilation = expand(Val(n-2), dilation)
    pad = calc_padding(Conv, pad, dims[1:n-2], dilation, stride)
    SincConv(f1s, f2s, fs, dims, σ, stride, pad, dilation)
end

Flux.@functor SincConv (f1s, f2s)

# function sincfunctions(f1s::VT, f2s::VT, t::TT, win::WT) where {VT<:AbstractArray{<:Real},TT,WT}
#     win .* (2 .* f2s .* sinc.(2 .* f2s .* t) .- 2 .* f1s .* sinc.(2 .* f1s .* t))
# end
function sincfunctions(f1s, f2s, dims::Tuple, fs::T, window::Function=hamming) where {T<:Real}
    n1, n2, n3, n4 = dims
    t = Zygote.ignore() do
        reshape(gett(n1, n2) ./ fs, n1, n2, 1, 1) |> x -> f1s isa CuArray ? gpu(x) : x
    end
    f1s, f2s = reshape(f1s, 1, 1, n3, n4), reshape(f2s, 1, 1, n3, n4)
    win = Zygote.ignore() do 
        window((n1, n2)) |> x -> convert.(T, x) |> x -> f1s isa CuArray ? gpu(x) : x
    end
    win .* (2 .* f2s .* sinc.(2 .* f2s .* t) .- 2 .* f1s .* sinc.(2 .* f1s .* t))
    #sincfunctions(f1sabs, f2sabs, t, win)
end
sincfunctions(c::SincConv) = sincfunctions(c.f1s, c.f2s, c.dims, c.fs)

function initmelcutofffreqs(dims::Tuple, fs::T=convert(T, 1)) where {T<:Real}
    f1s = zeros(T, dims[3:4]...)
    f2s = zeros(T, dims[3:4]...)
    for i ∈ 1:dims[3]
        f1s[i,:], f2s[i,:] = melcutofffrequencies(dims[4], fs)
    end
    f1s, f2s
end

function initrandcutofffreqs(rng::AbstractRNG, dims::Tuple, fs::T=convert(T, 1)) where {T<:Real}
    cutoff1 = 0
    cutoff2 = fs / 2
    f1s = zeros(T, dims[3:4]...)
    f2s = zeros(T, dims[3:4]...)
    for i ∈ 1:dims[3]
        for j ∈ 1:dims[4]
            f1s[i,j] = rand(rng, Uniform(cutoff1, cutoff2))
            f2s[i,j] = rand(rng, Uniform(f1s[i,j], cutoff2))
        end
    end
    f1s, f2s
end
initrandcutofffreqs(dims::Tuple, fs::T=convert(T, 1)) where {T<:Real} = initrandcutofffreqs(Random.GLOBAL_RNG, dims, fs)

function (c::SincConv)(x::AbstractArray)
    weight = sincfunctions(c)
    σ = c.σ
    cdims = DenseConvDims(x, weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    σ.(Flux.conv(x, weight, cdims))
end

function Base.show(io::IO, l::SincConv)
    print(io, "SincConv(", l.fs)
    print(io, ", ", l.dims[1:2], ", ", l.dims[3], "=>", l.dims[4])
    l.σ == identity || print(io, ", ", l.σ)
    print(io, ")")
end