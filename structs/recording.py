import cv2
from dataclasses import dataclass
import os
import shutil
import wave

from utils.audio_spectrogram import *
from utils.preprocessing import silence_removing
from utils.extract_features import *
from sklearn.preprocessing import MinMaxScaler

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
        scaler = MinMaxScaler(feature_range=(0, 1))

        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=SR)
            self.sound = parselmouth.Sound(self.recording_path)
        except:
            self.audio = []
            self.sound = []

        self.length = len(self.audio)
        self.trimmed_audio = self._trim_recording()
        self.normalized_signal = scaler.fit_transform(self.trimmed_audio.reshape(-1, 1)).flatten()

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
                self.spectrograms[setting] = self._get_melspectrogram(binsize, overlap, setting, filename)
            elif spectrogram_type == "spectrogram":
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

    def _get_spectrogram(self, binsize, overlap, settings_dir, filename):
        spectrogram_path = os.path.join(self.spectrogram_dir, settings_dir, filename)

        if not os.path.exists(spectrogram_path + '.npy'):
            self._generate_spectrogram(binsize, overlap, settings_dir, filename)

        spectrogram = np.load(spectrogram_path + '.npy')
        return cv2.resize(spectrogram, (227, 227))

    def _generate_spectrogram(self, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        trimmed_signal = self.normalized_signal[:int(self.sr * 0.3)]  # trim the first 0.3 second

        spectrogram = stft(trimmed_signal, binsize, overlap)

        log_scale_spec, freq = transfrom_spectrogram_to_logscale(spectrogram, factor=1.0, sr=self.sr)
        log_scale_spec_db = 20. * np.log10(np.abs(log_scale_spec) / 10e-6)  # amplitude to decibel

        plt.figure(figsize=(35, 27.5))
        plt.imshow(np.transpose(log_scale_spec_db), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
        plt.colorbar()

        plt.savefig(os.path.join(self.spectrogram_dir, settings_dir, filename + '.png'))
        np.save(os.path.join(self.spectrogram_dir, settings_dir, filename + '.npy'), np.transpose(log_scale_spec_db))
        plt.close()

    def _get_melspectrogram(self, binsize, overlap, settings_dir, filename):
        melspectrogram_path = os.path.join(self.melspectrogram_dir, settings_dir, filename)

        if not os.path.exists(melspectrogram_path + '.npy'):
            self._generate_melspectrogram(binsize, overlap, settings_dir, filename)

        spectrogram = np.load(melspectrogram_path + '.npy')
        return cv2.resize(spectrogram, (227, 227))

    def _generate_melspectrogram(self, binsize, overlap, settings_dir, filename, colormap="gnuplot2"):
        y = self.normalized_signal[:int(self.sr * 0.3)]
        hop_length = int(binsize * (1 - overlap))
        n_mels = 30

        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_fft=binsize, hop_length=hop_length, n_mels=n_mels)
        S_DB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=self.sr, hop_length=hop_length, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
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
        plt.savefig(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.png'))
        np.save(os.path.join(self.melspectrogram_dir, settings_dir, filename + '.npy'), S_DB)

        plt.close()
