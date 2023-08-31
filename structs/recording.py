from utils.audio_spectrogram import *
from utils.preprocessing import silence_removing
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import entropy
import cv2
from dataclasses import dataclass
import librosa
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.preprocessing import MinMaxScaler
import wave
import math
from PIL import Image

COLOR_PALETTES = [ 'gray', 'jet'] # 'bone', 'cool', 'copper', 'hot', 'hsv', 'pink'
SIZE = 227

@dataclass
class Recording:
    dir_path: str
    vowel: str
    filename: str
    classname: int
    settings: list

    def __post_init__(self):
        self.recording_path = os.path.join(self.dir_path, "recordings", self.vowel, self.filename)
        self.spectrogram_dir = os.path.join(self.dir_path, "spectrograms", self.vowel)
        self.melspectrogram_dir = os.path.join(self.dir_path, "melspectrograms", self.vowel)
        scaler = MinMaxScaler(feature_range=(0, 1))

        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=SR)
        except:
            self.audio = []

        self.trimmed_audio = self._trim_recording()
        self.normalized_signal = scaler.fit_transform(self.trimmed_audio.reshape(-1, 1)).flatten()
        self.normalized_signal = self.trimmed_audio
        self.length = len(self.normalized_signal)
        if self.length < self.sr * 0.4:
            print(self.length)

        self.spectrograms = {}

        for setting in self.settings:
            setting_params = setting.split("_")
            spectrogram_type = setting_params[0]

            binsize, overlap = int(setting_params[1]), float(setting_params[2])
            filename = self.filename.split(".")[0] + "_" + setting

            if spectrogram_type == "melspectrogram":
                melspectrogram_dir = os.path.join(self.melspectrogram_dir, setting)
                if not os.path.exists(melspectrogram_dir):
                    os.makedirs(melspectrogram_dir)
                spectrograms = self._get_melspectrogram(binsize, overlap, setting, filename)
                self.spectrograms[setting] = spectrograms

            elif spectrogram_type == "spectrogram":
                spectrogram_dir = os.path.join(self.spectrogram_dir, setting)
                if not os.path.exists(spectrogram_dir):
                    os.makedirs(spectrogram_dir)
                spectrograms = self._get_spectrogram(binsize, overlap, setting, filename)
                self.spectrograms[setting] = spectrograms

    def get_features(self):
        y = self.trimmed_audio
        sr = self.sr
        # Obliczenie cechy jitter (RR-interval variability)
        rr_intervals = np.diff(librosa.times_like(y))
        jitter = np.mean(np.abs(np.diff(rr_intervals)))

        # Obliczenie cechy shimmer (amplitude variation between consecutive periods)
        peaks, _ = find_peaks(y, height=0)
        shimmer = np.mean(np.abs(np.diff(peaks))) / len(peaks)

        # # Obliczenie cechy HNR (Harmonic-to-Noise Ratio)
        # harmonics, percussive = librosa.effects.hpss(y)
        # hnr = np.mean(librosa.feature.spectral_centroid() / librosa.feature.spectral_centroid(percussive))

        # Obliczenie cechy f0 (fundamental frequency)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Obliczenie cech MFCC (Mel-Frequency Cepstral Coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # Obliczenie cechy NHR (Noise-to-Harmonic Ratio)
        nhr = np.mean(librosa.effects.harmonic(y) / librosa.effects.percussive(y))

        # W tej wersji brakuje cech RPDE i DFA, ponieważ nie ma dostępnej biblioteki

        # Obliczenie cechy PPE (Pitch Period Entropy)
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr / np.max(autocorr)  # Normalizacja
        ppe = entropy(autocorr)

        return [jitter, shimmer, f0, voiced_flag, voiced_probs, mfcc, nhr, ppe]

    def _trim_recording(self):
        trimmed_recording_dir = os.path.join(self.dir_path, "trimmed_recordings", self.vowel)

        if not os.path.exists(trimmed_recording_dir):
            os.makedirs(trimmed_recording_dir)

        trimmed_recording_path = os.path.join(trimmed_recording_dir, self.filename)

        if os.path.exists(trimmed_recording_path):
            trimmed_recording, sr = librosa.load(trimmed_recording_path, sr=SR)
        else:
            trimmed_recording = silence_removing(self.audio, self.filename)
            if trimmed_recording:
                audio_array = np.array(trimmed_recording)
                output_file = wave.open(trimmed_recording_path, 'w')
                output_file.setparams((1, 2, SR, len(audio_array), 'NONE', 'not compressed'))
                for sample in audio_array:
                    sample = int(sample * 32767)
                    output_file.writeframes(sample.to_bytes(2, byteorder='little', signed=True))
                output_file.close()
            else:
                shutil.copy2(self.recording_path, trimmed_recording_path)
                trimmed_recording = self.audio

        trimmed_recording = np.array(trimmed_recording)
        trimmed_recording = np.trim_zeros(trimmed_recording)
        return np.trim_zeros(trimmed_recording)

    def _get_spectrogram(self, binsize, overlap, settings_dir, filename):
        spectrogram_path = os.path.join(self.spectrogram_dir, settings_dir, filename)
        spectrograms = []
        for palette in COLOR_PALETTES:
            if not os.path.exists(spectrogram_path + f"{palette}.png"):
                self._generate_spectrogram(binsize, overlap, settings_dir, filename, palette)

            image_array = cv2.imread(spectrogram_path + f"{palette}.png")
            resized_image = cv2.resize(image_array, (SIZE, SIZE))
            spectrograms.append(resized_image)
        return spectrograms

    def _generate_spectrogram(self, binsize, overlap, settings_dir, filename, palette, colormap="gnuplot2"):
        trimmed_signal = self.normalized_signal[:int(self.sr * 0.75)]  # trim the first 0.4 second

        D = librosa.amplitude_to_db(librosa.stft(trimmed_signal), ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(D, sr=self.sr, x_axis='off', y_axis='off', cmap=palette)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.spectrogram_dir, settings_dir, filename + '{}.png'.format(palette)), pad_inches=0, transparent=True)
        plt.close()

    def _get_melspectrogram(self, binsize, overlap, settings_dir, filename):
        melspectrogram_path = os.path.join(self.melspectrogram_dir, settings_dir, filename)
        if not os.path.exists(melspectrogram_path + ".png"):
            self._generate_melspectrogram(binsize, overlap, settings_dir, filename)

        image_array = cv2.imread(melspectrogram_path + ".png")
        resized_image = cv2.resize(image_array, (SIZE, SIZE))
        return resized_image


    def _generate_melspectrogram(self, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        y = self.normalized_signal[:int(self.sr * 0.5)]
        hop_length = int(binsize * (1 - overlap))
        n_fft = 2048  # Rozmiar okna FFT
        hop_length = 512  # Krok analizy
        n_mels = 128  # Liczba pasm Mel

        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=self.sr, x_axis='off', y_axis='off', cmap=colormap)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.png'), pad_inches=0,
                    transparent=True)
        plt.close()


        # plt.colorbar(format='%+2.0f dB')
        # melspec = librosa.feature.melspectrogram(y=trimmed_signal, sr=self.sr, n_fft=binsize, n_mels=128,fmax=44100,
        #                                          hop_length=int(binsize * (1 - overlap)))
        # melspec_db = librosa.power_to_db(melspec, ref=np.max)
        #
        # fig, ax = plt.subplots(sharex=True, figsize=(20, 15))
        # img = librosa.display.specshow(melspec_db, sr=self.sr, hop_length=int(binsize * (1 - overlap)), x_axis='time',
        #                                y_axis='mel', cmap=colormap)
        #
        # duration = len(trimmed_signal) / self.sr
        # ax.set_xlim([0, duration])
        #
        # fig.colorbar(img)
        # ax.set(title='Mel spectrogram')
        # ax.label_outer()
        # ax.set_ylim([0, self.sr / 2])  # Domyślna wartość dla osi y mel-spectrogramu
        #
        # plt.savefig(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.png'))
        # np.save(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.npy'), S_DB)
        #
        # plt.close()
        # mel_spectrogram = librosa.feature.melspectrogram(y=self.normalized_signal, sr=self.sr)
        #
        # # Przekształcenie do skali decybelowej
        # mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        #
        # # Tworzenie obrazu z mel-spectrogramem
        # plt.figure(figsize=(10, 6))
        # librosa.display.specshow(mel_spectrogram_db, sr=self.sr, x_axis='off', y_axis='off')
        # plt.axis('off')
        # plt.tight_layout()
        #
        # plt.savefig(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.png'), pad_inches=0, transparent=True)
        # plt.close()
