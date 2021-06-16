
hertz2mel(f) = 2595 * log10(1 + f/700)
mel2hertz(m) = 700 * (10 ^ (m / 2595) - 1)

"""
Return mel-scale cutoff frequencies given number of filterbanks `nfilters`.

# Reference:
https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
function melcutofffrequencies(nfilters, fs)
    lowfreqmel = 0
    highfreqmel = hertz2mel(fs / 2)
    melpoints = range(lowfreqmel, highfreqmel; length=nfilters + 2)
    hertzpoints = mel2hertz.(melpoints)
    lowcutoffs = hertzpoints[1:end-2]
    highcutoffs = hertzpoints[3:end]
    lowcutoffs, highcutoffs
end

"""
Return a Discrete Fourier Transform (DFT) matrix.

# Reference:
https://en.wikipedia.org/wiki/DFT_matrix
"""
function dftmatrix(M, N)
    ω = exp(-2π * im / N)
    W = Matrix{typeof(ω)}(undef, M, N)
    for j ∈ 0:M-1
        for k ∈ 0:N-1
            W[j+1,k+1] = ω ^ (j * k)
        end
    end
    W / √N
end
dftmatrix(N) = dftmatrix(N, N)
