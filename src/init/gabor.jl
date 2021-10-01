_hz2mel(f) = 2595 * log10(1 + f / 700)
_mel2hz(m) = 700 * (10 ^ (m / 2595) - 1) 

function _melfilter_energy(mel_filter, nfft)
    height = maximum(mel_filter)
    hz_spread = (length(findall(mel_filter .> 0)) .+ 2) * 2π / nfft
    0.5 * height * hz_spread
end

function _build_mels(nfilters, fs, minfreq, maxfreq, nfft, normalize_energy)
    melfilters = zeros(nfft ÷ 2 + 1, nfilters)
    dfreq = fs / nfft
    
    melmin = _hz2mel(minfreq)
    melmax = _hz2mel(maxfreq)
    dmelbw = (melmax - melmin) / (nfilters + 1)
    filt_edge = _mel2hz.(melmin .+ dmelbw .* range(0, nfilters+1, step=1))

    height = 1
    for filter_idx ∈ 1:nfilters
        leftfr = min(round(Int, filt_edge[filter_idx] / dfreq), nfft ÷ 2) + 1
        centerfr = min(round(Int, filt_edge[filter_idx + 1] / dfreq), nfft ÷ 2) + 1
        rightfr = min(round(Int, filt_edge[filter_idx + 2] / dfreq), nfft ÷ 2) + 1

        if centerfr != leftfr 
            leftslope = height / (centerfr - leftfr)
        else
            leftslope = 0
        end
        freq = leftfr + 1
        while freq < centerfr
            melfilters[freq,filter_idx] = (freq - leftfr) * leftslope
            freq += 1
        end
        if freq == centerfr 
            melfilters[freq,filter_idx] = height
            freq += 1
        end
        if centerfr != rightfr
            rightslope = height / (centerfr - rightfr)
        end
        while freq < rightfr
            melfilters[freq,filter_idx] = (freq - rightfr) * rightslope
            freq += 1
        end
        if normalize_energy
            energy = _melfilter_energy(melfilters[:,filter_idx], nfft)
            melfilters[:,filter_idx] ./= energy
        end
    end
    melfilters
end

function _gabor_params_from_mel(nfft, mel_filter)
    coeff = sqrt(2 * log(2)) * nfft
    mel_filter = sqrt.(mel_filter)
    center_frequency = argmax(mel_filter) - 1
    peak = mel_filter[center_frequency]
    half_magnitude = peak / 2
    spread = findall(mel_filter .> half_magnitude)
    width = max(spread[end] - spread[1], 1)
    center_frequency * 2π / nfft, coeff / (π * width)
end

function _gabor_wavelet(eta, sigma, wlen, fs)
    T = wlen * fs ÷ 1000
    gabor_function = t -> (1 / (sqrt(2π) * sigma)) * exp(im * eta * t) * exp(-(t ^ 2) / (2 * sigma ^ 2))
    [gabor_function(t) for t ∈ range(-T/2, T/2, step=1)]
end

"""
Initialize `Conv` weights based on Gabor filters.

# Reference
https://github.com/facebookresearch/tdfbanks
"""
function gaborfilters(;nfilters=40,
                       fs=9600,
                       min_freq=0,
                       max_freq=fs/2,
                       wlen=25,
                       nfft=512,
                       normalize_energy=false)
    melfilters = _build_mels(nfilters, fs, min_freq, max_freq, nfft, normalize_energy)
    gaborfilters = zeros(ComplexF64, wlen * fs ÷ 1000 + 1, 40)
    sigmas = []
    center_frequencies = []
    for (i, melfilter) ∈ enumerate(eachcol(melfilters))
        center_frequency, sigma = _gabor_params_from_mel(nfft,melfilter)
        push!(sigmas, sigma)
        push!(center_frequencies, center_frequency)
        gaborfilter = _gabor_wavelet(center_frequency, sigma, wlen, fs)
        gaborfilter .*= sqrt.(_melfilter_energy(melfilter, nfft) * 2 * sqrt(π) * sigma)
        gaborfilters[:,i] = gaborfilter
        #push!(gaborfilters, gaborfilter) 
    end
    gaborfilters, sigmas, center_frequencies
end