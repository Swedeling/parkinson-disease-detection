import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os
from config import SR


class Spectrogram:
    def __init__(self, vowel, spectrograms_dir, audio_path, binsizes, overlaps):
        self.audio_path = audio_path
        self.target_dir = spectrograms_dir
        self.vowel = vowel

        for binsize in binsizes:
            for overlap in overlaps:
                self.generate_spectrogram(binsize, overlap)

    def generate_spectrogram(self, binsize, overlap):
        directory = os.path.join(self.target_dir, '{}_stft{}_overlap{}_npy'.format(self.vowel, binsize, overlap))
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.plot_stft(self.audio_path, directory, binsize, overlap)

    @staticmethod
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

    @staticmethod
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

    def plot_stft(self, audiopath, directory, binsize, overlap, colormap="plasma"):
        sr, signal = wav.read(audiopath)
        trimmed_signal = signal[:int(sr * 0.3)]  # trim the first 0.3 second

        spectrogram = self.stft(trimmed_signal, binsize, overlap)
        log_scale_spec, freq = self.transfrom_spectrogram_to_logscale(spectrogram, factor=1.0, sr=sr)
        log_scale_spec_db = 20. * np.log10(np.abs(log_scale_spec) / 10e-6)  # amplitude to decibel

        plt.figure(figsize=(35, 27.5))
        plt.imshow(np.transpose(log_scale_spec_db), origin="lower", aspect="auto", cmap=colormap, interpolation="none")

        file_basename = os.path.basename(audiopath)
        filename = file_basename.split('.')[0]

        full_filename_without_extension = '{}_binsize{}_overlap{}'.format(filename, binsize, overlap)

        plt.savefig(os.path.join(directory, full_filename_without_extension + '.png'))
        np.save(os.path.join(directory, full_filename_without_extension + '.npy'), np.transpose(log_scale_spec_db))

        plt.close()
