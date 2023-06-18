import cv2
from dataclasses import dataclass
import os
import shutil
import wave

from utils.audio_spectrogram import *
from utils.preprocessing import silence_removing
from utils.extract_features import *


@dataclass
class Recording:
    dir_path: str
    vowel: str
    dataset: str
    filename: str
    classname: int
    settings: list

    def __post_init__(self):
        self.recording_path = os.path.join(self.dir_path, "recordings", self.vowel, self.dataset, self.filename)
        self.spectrogram_dir = os.path.join(self.dir_path, "spectrograms", self.vowel)
        self.melspectrogram_dir = os.path.join(self.dir_path, "melspectrograms", self.vowel)

        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=SR)
            self.sound = parselmouth.Sound(self.recording_path)
        except:
            self.audio = []
            self.sound = []

        self.length = len(self.audio)
        self.trimmed_audio = self._trim_recording()

        melspectrogram_dir = os.path.join(self.melspectrogram_dir)
        if not os.path.exists(melspectrogram_dir):
            os.makedirs(melspectrogram_dir)

        self.spectrograms = {}

        for setting in self.settings:
            setting_params = setting.split("_")
            spectrogram_type = setting_params[0]

            if spectrogram_type == "melspectrogram":
                self.spectrograms[setting] = self._get_melspectrogram(
                    '{}_melspectrogram'.format(self.filename.split(".")[0]))

            elif spectrogram_type == "spectrogram" and len(setting_params) > 1:
                binsize, overlap = int(setting_params[1]), float(setting_params[2])

                filename = self.filename.split(".")[0] + "_" + setting
                spectrogram_dir = os.path.join(self.spectrogram_dir, setting)
                if not os.path.exists(spectrogram_dir):
                    os.makedirs(spectrogram_dir)

                self.spectrograms[setting] = self._get_spectrogram(binsize, overlap, setting, filename)

    def _trim_recording(self):
        trimmed_recording_dir = os.path.join(self.dir_path, "trimmed_recordings", self.vowel)

        if not os.path.exists(trimmed_recording_dir):
            os.makedirs(trimmed_recording_dir)

        trimmed_recording_path = os.path.join(trimmed_recording_dir, self.filename)

        if os.path.exists(trimmed_recording_path):
            trimmed_recording, sr = librosa.load(self.recording_path, sr=SR)
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

    def _get_spectrogram(self, binsize, overlap, dir_settings, filename):
        spectrogram_path = os.path.join(self.spectrogram_dir, dir_settings, filename)

        if not os.path.exists(spectrogram_path + '.npy'):
            self._generate_spectrogram(binsize, overlap, dir_settings, filename)

        spectrogram = np.load(spectrogram_path + '.npy')
        return cv2.resize(spectrogram, (227, 227))

    def _generate_spectrogram(self, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        trimmed_signal = self.trimmed_audio[:int(self.sr * 0.3)]  # trim the first 0.3 second

        spectrogram = stft(trimmed_signal, binsize, overlap)

        log_scale_spec, freq = transfrom_spectrogram_to_logscale(spectrogram, factor=1.0, sr=self.sr)
        log_scale_spec_db = 20. * np.log10(np.abs(log_scale_spec) / 10e-6)  # amplitude to decibel

        plt.figure(figsize=(35, 27.5))
        plt.imshow(np.transpose(log_scale_spec_db), origin="lower", aspect="auto", cmap=colormap, interpolation="none")

        plt.savefig(os.path.join(self.spectrogram_dir, settings_dir, filename + '.png'))
        np.save(os.path.join(self.spectrogram_dir, settings_dir, filename + '.npy'), np.transpose(log_scale_spec_db))
        plt.close()

    def _get_melspectrogram(self, filename):
        melspectrogram_path = os.path.join(self.melspectrogram_dir, filename)

        if not os.path.exists(melspectrogram_path + '.npy'):
            self._generate_melspectrogram(filename)

        spectrogram = np.load(melspectrogram_path + '.npy')
        return cv2.resize(spectrogram, (227, 227))

    def _generate_melspectrogram(self, filename):
        trimmed_signal = self.trimmed_audio[:int(self.sr * 0.09)]  # trim the first 0.09 second
        librosa.feature.mfcc(y=trimmed_signal, sr=self.sr)
        librosa.feature.mfcc(y=trimmed_signal, sr=self.sr, hop_length=512, htk=True)

        s = librosa.feature.melspectrogram(y=trimmed_signal, sr=self.sr, n_mels=128, fmax=44100)

        librosa.feature.mfcc(S=librosa.power_to_db(s))
        mfccs = librosa.feature.mfcc(y=trimmed_signal, sr=self.sr, n_mfcc=12)

        fig, ax = plt.subplots(sharex=True, figsize=(20, 15))
        img = librosa.display.specshow(librosa.power_to_db(s, ref=np.max), x_axis='time', y_axis='mel', fmax=44100)

        fig.colorbar(img)
        ax.set(title='Mel spectrogram')
        ax.label_outer()
        ax.set_ylim([0, 22000])

        plt.savefig(os.path.join(self.melspectrogram_dir, filename + '.png'))
        np.save(os.path.join(self.melspectrogram_dir, filename + '.npy'), mfccs)

        plt.close()
