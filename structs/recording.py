from config import AUGMENTATION, SIGNAL_DURATION
from utils.signal_processing import *

import cv2
from dataclasses import dataclass
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from scipy.signal import find_peaks
import shutil
import wave


@dataclass
class Recording:
    dir_path: str
    vowel: str
    filename: str
    classname: int
    settings: list
    dataset: str
    language: str

    def __post_init__(self):
        self.recording_path = os.path.join(self.dir_path, "recordings", self.vowel, self.dataset, self.filename)
        self.spectrogram_dir = os.path.join(self.dir_path, "spectrograms", self.vowel)
        self.mel_spectrogram_dir = os.path.join(self.dir_path, "mel-spectrograms", self.vowel)
        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=SR, duration=SIGNAL_DURATION)
        except:
            self.audio = []

        self.trimmed_audio = self._trim_recording()
        self.length = len(self.trimmed_audio)
        self.spectrograms = {}

        for setting in self.settings:
            self.spectrograms[setting] = []
            setting_params = setting.split("_")
            spectrogram_type = setting_params[0]

            filename = self.filename.split(".")[0] + "_" + setting

            if spectrogram_type == "mel-spectrogram":
                n_mels = int(setting_params[1])
                binsize = int(setting_params[2])
                overlap = int(setting_params[3])
                mel_spectrogram_dir = os.path.join(self.mel_spectrogram_dir, setting)
                if not os.path.exists(mel_spectrogram_dir):
                    os.makedirs(mel_spectrogram_dir)
                spectrograms = self._get_mel_spectrogram(n_mels, binsize, overlap, setting, filename)
                self.spectrograms[setting] += spectrograms

            elif spectrogram_type == "spectrogram":
                binsize = int(setting_params[1])
                overlap = int(setting_params[2])
                spectrogram_dir = os.path.join(self.spectrogram_dir, setting)
                if not os.path.exists(spectrogram_dir):
                    os.makedirs(spectrogram_dir)
                spectrograms = self._get_spectrogram(binsize, overlap, setting, filename)
                self.spectrograms[setting] += spectrograms

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

    @staticmethod
    def load_spectrogram(path):
        spectrogram = np.load(path, allow_pickle=True)
        index_of_min = spectrogram.shape.index(min(spectrogram.shape))
        size = max(spectrogram.shape)
        spectrogram = np.repeat(spectrogram, size / spectrogram.shape[index_of_min], axis=index_of_min)
        spectrogram = cv2.resize(spectrogram, (size, size), interpolation=cv2.INTER_AREA)

        return spectrogram

    def _get_spectrogram(self, binsize, overlap, settings_dir, filename):
        spectrogram_path = os.path.join(self.spectrogram_dir, settings_dir, filename)
        spectrograms = []
        for augmentation in AUGMENTATION:
            if not os.path.exists(spectrogram_path + f"_{augmentation}.npy"):
                self._generate_spectrogram(augmentation, binsize, overlap, settings_dir, filename)

            if os.path.exists(spectrogram_path + f"_{augmentation}.npy"):
                spectrogram = self.load_spectrogram(spectrogram_path + f"_{augmentation}.npy")
                spectrograms.append(spectrogram)
        return spectrograms

    def _generate_spectrogram(self, augmentation, binsize, overlap, settings_dir, filename):
        signal = apply_audio_augmentation(self.audio, augmentation)
        if signal is not None:
            d = np.abs(librosa.stft(signal, n_fft=binsize, hop_length=overlap))
            d_db = librosa.amplitude_to_db(d, ref=np.max)

            spectrogram_path = os.path.join(self.spectrogram_dir, settings_dir, filename)
            self.save_spectrogram(d_db, spectrogram_path, augmentation)

    def _get_mel_spectrogram(self, n_mels, binsize, overlap, settings_dir, filename):
        mel_spectrogram_path = os.path.join(self.mel_spectrogram_dir, settings_dir, filename)
        mel_spectrograms = []

        for augmentation in AUGMENTATION:
            if not os.path.exists(mel_spectrogram_path + f"_{augmentation}.npy"):
                self._generate_mel_spectrogram(augmentation, n_mels, binsize, overlap, settings_dir, filename)

            if os.path.exists(mel_spectrogram_path + f"_{augmentation}.npy"):
                mel_spectrogram = self.load_spectrogram(mel_spectrogram_path + f"_{augmentation}.npy")
                mel_spectrograms.append(mel_spectrogram)

        return mel_spectrograms

    def _generate_mel_spectrogram(self, augmentation, n_mels, binsize, overlap, settings_dir, filename):
        signal = apply_audio_augmentation(self.audio, augmentation)

        if signal is not None:
            s = librosa.feature.melspectrogram(y=signal, sr=self.sr, n_mels=n_mels, fmax=self.sr,  n_fft=binsize,
                                               hop_length=overlap, window=scipy.signal.windows.hann)
            s_db = librosa.power_to_db(s, ref=np.max)
            mel_spectrogram_path = os.path.join(self.mel_spectrogram_dir, settings_dir, filename)
            self.save_spectrogram(s_db, mel_spectrogram_path, augmentation)

    def save_spectrogram(self, s_db, spectrogram_path, augmentation):
        fig, ax = plt.subplots(figsize=(20, 15))
        img = librosa.display.specshow(s_db, x_axis='time', y_axis='mel', fmax=self.sr)
        fig.colorbar(img)
        ax.set(title='Mel spectrogram')
        ax.label_outer()
        ax.set_ylim([0, 22000])

        spectrogram_png = spectrogram_path + f'_{augmentation}.png'
        spectrogram_npy = spectrogram_path + f'_{augmentation}.npy'
        plt.savefig(spectrogram_png, pad_inches=0, transparent=True)
        np.save(spectrogram_npy, s_db)
        print('file was saved: ', spectrogram_png)
        plt.close()
