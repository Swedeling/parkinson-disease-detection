import numpy as np
from numpy.lib import stride_tricks
from config import SR


def stft(sig, frame_size, overlap, window=np.hanning):
    """ short time fourier transform of audio signal """
    win = window(frame_size)
    hop_size = int(frame_size - np.floor(overlap * frame_size))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frame_size / 2.0))), sig)
    # cols for windowing
    cols = np.ceil((len(samples) - frame_size) / float(hop_size)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frame_size))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frame_size),
                                      strides=(samples.strides[0] * hop_size, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


def transfrom_spectrogram_to_logscale(spec, sr=SR, factor=20.):
    """ scale frequency axis logarithmically """
    time_bins, freq_bins = np.shape(spec)

    scale = np.linspace(0, 1, freq_bins) ** factor
    scale *= (freq_bins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    logscale_spec = np.complex128(np.zeros([time_bins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            logscale_spec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            logscale_spec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    all_freqs = np.abs(np.fft.fftfreq(freq_bins * 2, 1. / sr)[:freq_bins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(all_freqs[int(scale[i]):])]
        else:
            freqs += [np.mean(all_freqs[int(scale[i]):int(scale[i + 1])])]

    return logscale_spec, freqs