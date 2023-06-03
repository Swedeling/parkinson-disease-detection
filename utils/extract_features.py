from scipy.signal import hilbert
import numpy as np
import librosa
import librosa.display
import parselmouth
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from Signal_Analysis.features import signal as signal_analysis


def compute_hnr(signal, sr):
    return signal_analysis.get_HNR(signal, sr)


def compute_f0(signal, sr):
    return signal_analysis.get_F_0(signal, sr)


def compute_jitter(signal, sr):
    return signal_analysis.get_Jitter(signal, sr)


def compute_pulses(signal, sr):
    return signal_analysis.get_Pulses(signal, sr)


def compute_spectrogram(signal, sr):
    # signal, sr = librosa.load("audio.wav")
    spectrogram = librosa.stft(signal)
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

    # plt.figure(figsize=(12, 8))
    # librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='log')
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Spectrogram')
    # plt.show()

    return spectrogram_db


def compute_signal_energy(signal):
    return np.sum(np.square(signal))


def compute_statistical_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    return mean, variance


def compute_pitch(signal, sr):
    pitch, _ = librosa.core.piptrack(y=signal, sr=sr)
    mean_pitch = np.mean(pitch)
    std_pitch = np.std(pitch)
    return mean_pitch, std_pitch


def compute_envelopes(signal):
    # signal, sr = librosa.load('audio.wav')
    envelopes = np.abs(hilbert(signal))
    return envelopes


def time_features(signal, sr):
    samples_len = len(signal)
    duration = librosa.get_duration(S=signal, sr=sr)
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    return samples_len, duration, onset_env

