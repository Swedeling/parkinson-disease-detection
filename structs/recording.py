from utils.audio_spectrogram import *
from utils.preprocessing import silence_removing
import numpy as np
from scipy.signal import find_peaks, butter, lfilter
import cv2
from dataclasses import dataclass
import librosa
import matplotlib.pyplot as plt
import os
import shutil
import wave
import random

AUGUMENTATION = ["filtered", "pitch", "slow", "speed"] #,  "pitch", "filtered", "rolled", "slow", "speed"
SIZE = 224

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
        self.recording_path = os.path.join(self.dir_path, "trimmed_recordings", self.vowel, self.filename)
        self.spectrogram_dir = os.path.join(self.dir_path, "spectrograms", self.vowel)
        self.melspectrogram_dir = os.path.join(self.dir_path, "melspectrograms", self.vowel)
        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=44100, duration=0.5)
        except:
            self.audio = []

        self.trimmed_audio = self.audio
        self.length = len(self.trimmed_audio)
        self.spectrograms = {}

        for setting in self.settings:
            self.spectrograms[setting] = []
            setting_params = setting.split("_")
            spectrogram_type = setting_params[0]

            binsize, overlap = int(setting_params[1]), float(setting_params[2])
            filename = self.filename.split(".")[0] + "_" + setting

            if spectrogram_type == "melspectrogram":
                melspectrogram_dir = os.path.join(self.melspectrogram_dir, setting)
                if not os.path.exists(melspectrogram_dir):
                    os.makedirs(melspectrogram_dir)
                spectrograms = self._get_melspectrogram(binsize, overlap, setting, filename)
                self.spectrograms[setting] += spectrograms

            elif spectrogram_type == "spectrogram":
                spectrogram_dir = os.path.join(self.spectrogram_dir, setting)
                if not os.path.exists(spectrogram_dir):
                    os.makedirs(spectrogram_dir)
                spectrograms = self._get_spectrogram(binsize, overlap, setting, filename)
                self.spectrograms[setting] = spectrograms

    def random_roll(self, signal):
        shift_amount = np.random.randint(0, len(signal)) if len(signal) > 0 else 0
        rolled_signal = np.roll(signal, shift_amount)
        return rolled_signal

    def pitch_change(self, y):
        time_stretch_factor = random.uniform(3, 5) * random.choice([-1, 1])
        stretched_audio = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=time_stretch_factor)
        resampled_audio = librosa.resample(stretched_audio, orig_sr=self.sr, target_sr=self.sr)
        return resampled_audio

    def speed_up(self, y):
        speed_factor = random.uniform(1.2, 2.5)
        y_fast = librosa.resample(y, orig_sr=self.sr, target_sr=self.sr)
        return y_fast

    def slow_down(self, y):
        speed_factor = random.uniform(0.2, 0.8)
        y_slow = librosa.resample(y, orig_sr=self.sr, target_sr=self.sr)
        return y_slow

    def color_noise(self, y):
        power_law_exponent = random.uniform(4, 6)
        noise = np.random.normal(0, 0.5, len(y))
        power_law_noise = noise ** power_law_exponent
        power_law_noise[np.isnan(power_law_noise)] = 0
        noisy_audio = y + power_law_noise
        return noisy_audio

    def bandpass_filter(self, signal):
        low_freq = 500  # Hz
        high_freq = 1500  # Hz
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        b, a = butter_bandpass(low_freq, high_freq, self.sr)
        y = lfilter(b, a, signal)
        return y

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
        for augmentation in AUGUMENTATION:
            if not os.path.exists(spectrogram_path + f"_{augmentation}.png"):
                self._generate_spectrogram(augmentation, binsize, overlap, settings_dir, filename,)

            image_array = cv2.imread(spectrogram_path + f"_{augmentation}.png")
            resized_image = cv2.resize(image_array, (SIZE, SIZE))
            spectrograms.append(resized_image)
        return spectrograms

    def _generate_spectrogram(self, augmentation, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        aug = augmentation.split("_")[0]
        if aug == "clear":
            y = self.trimmed_audio
        if aug == "rolled":
            y = self.trimmed_audio
            y = self.random_roll(y)
        if aug == "filtered":
            y = self.trimmed_audio
            y = self.bandpass_filter(y)
        if aug == "aug":
            y = self.trimmed_audio  # [int(self.sr * 0.1):int(self.sr * 0.6)]
            y = self.random_roll(y)
            y = self.bandpass_filter(y)
        D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(D, sr=self.sr, x_axis='off', y_axis='off')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.spectrogram_dir, settings_dir, filename + f'_{augmentation}.png'), pad_inches=0, transparent=True)
        plt.close()

    def _get_melspectrogram(self, binsize, overlap, settings_dir, filename):
        melspectrogram_path = os.path.join(self.melspectrogram_dir, settings_dir, filename)
        images = []
        for augmentation in AUGUMENTATION:
            if not os.path.exists(melspectrogram_path + f"_{augmentation}.npy"):
                self._generate_melspectrogram(augmentation, binsize, overlap, settings_dir, filename)

            melspectrogram = np.load(melspectrogram_path + f"_{augmentation}.npy", allow_pickle=True)
            melspectrogram = np.repeat(melspectrogram, 128 // melspectrogram.shape[1], axis=1)

            melspectrogram = cv2.resize(melspectrogram, (128, 128), interpolation=cv2.INTER_AREA)

            images.append(melspectrogram)
        return images

    def _generate_melspectrogram(self, augmentation, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        aug = augmentation.split("_")[0]
        y = self.audio
        if aug == "clear":
            pass
        if aug == "filtered":
            pass
        if aug == "pitch":
            y = self.pitch_change(y)
        if aug == "speed":
            y = self.speed_up(y)
        if aug == "slow":
            y = self.slow_down(y)
        if aug == "noise":
            y = self.color_noise(y)

        y = y[int(self.sr*0.41):]

        if aug == "rolled":
            y = self.random_roll(y)

        y = self.bandpass_filter(y)

        sr = self.sr
        librosa.feature.mfcc(y=y, sr=sr)

        librosa.feature.mfcc(y=y, sr=sr, hop_length=binsize, htk=True)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=overlap, fmax=44100)

        librosa.feature.mfcc(S=librosa.power_to_db(S))

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

        melspectrogram_path = os.path.join(self.melspectrogram_dir, settings_dir, filename)

        filename_without_extensionx = melspectrogram_path

        filename_without_extension = filename_without_extensionx + f'_{augmentation}.png'

        filename_without_extension_npy = filename_without_extensionx + f'_{augmentation}.npy'

        fig, ax = plt.subplots(sharex=True, figsize=(20, 15))

        img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', fmax=44100)

        fig.colorbar(img)

        ax.set(title='Mel spectrogram')

        ax.label_outer()

        ax.set_ylim([0, 22000])

        plt.savefig(filename_without_extension,  pad_inches=0, transparent=True)
        np.save(filename_without_extension_npy, librosa.power_to_db(S, ref=np.max))

        print('file was saved: ', filename_without_extension)

        plt.close()

