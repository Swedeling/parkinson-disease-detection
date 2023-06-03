from dataclasses import dataclass
import os
import wave
from config import SR, COMPUTE_ADDITIONAL_FEATURES
from utils.audio_spectrogram import Spectrogram
from utils.preprocessing import silence_removing
from utils.extract_features import *

BINSIZE = [512]  # TODO Obsługa większej liczby możliwości
OVERLAP = [0.1]


@dataclass
class Recording:
    dir_path: str
    vowel: str
    filename: str
    classname: int

    def __post_init__(self):
        self.recording_path = os.path.join(self.dir_path, "recordings", self.vowel, self.filename)

        try:
            self.audio, self.sr = librosa.load(self.recording_path, sr=SR)
            self.sound = parselmouth.Sound(self.recording_path)
        except:
            self.audio = []
            self.sound = []

        self.length = len(self.audio)
        self.trimmed_audio = self._trim_recording()
        self.spectrogram = self._generate_spectrogram()
        # self.features = self.extract_features()

        if COMPUTE_ADDITIONAL_FEATURES:
            self.mfcc_features = self._compute_mfcc_features()
            self.hnr = compute_hnr(self.audio, self.sr)
            self.f0 = compute_f0(self.audio, self.sr)
            jitter = compute_jitter(self.audio, self.sr)
            self.features = {"mfcc": self.mfcc_features, "hnr": self.hnr, "f0": self.f0,
                             "jitter_local": jitter['local'], "jitter_local_absolute": jitter['local, absolute'],
                             "jitter_rap": jitter['rap'], "jitter_ppq5": jitter['ppq5'], "jitter_ddp": jitter['ddp']}

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
                output_file.setparams((1, 2, 44100, len(audio_array), 'NONE', 'not compressed'))
                for sample in audio_array:
                    sample = int(sample * 32767)
                    output_file.writeframes(sample.to_bytes(2, byteorder='little', signed=True))
                output_file.close()

        return trimmed_recording

    def _generate_spectrogram(self):
        spectrogram_path = os.path.join(self.dir_path, "spectrograms", self.vowel,
                                        '{}_stft{}_overlap{}_npy'.format(self.vowel, BINSIZE[0], OVERLAP[0]),
                                        '{}_binsize{}_overlap{}.npy'.format(self.filename.split(".")[0], BINSIZE[0],
                                                                            OVERLAP[0]))

        if not os.path.exists(spectrogram_path):
            spectrograms_dir = os.path.join(self.dir_path, "spectrograms", self.vowel)
            audio_path = os.path.join(self.dir_path, "trimmed_recordings", self.vowel, self.filename)
            spectrogram = Spectrogram(self.vowel, spectrograms_dir, audio_path, BINSIZE, OVERLAP)

        img = np.load(spectrogram_path)
        top = bottom = 0
        left = 98 # 123
        right = 99 # 124

        img_with_border = np.pad(img, ((top, bottom), (left, right)), mode='constant')

        return img_with_border

    def _compute_mfcc_features(self):
        mfcc_features = librosa.feature.mfcc(y=self.audio, sr=self.sr)
        flattened_list = [np.mean(sublist) for sublist in mfcc_features]
        normalized_features = (flattened_list - np.mean(flattened_list)) / np.std(flattened_list)
        return normalized_features

    def _compute_chromagram(self):
        return librosa.feature.chroma_stft(y=self.audio, sr=self.sr)

    def extract_features(self):
        features = []
        # f1, f2, f3 = compute_formants(self.recording_path)
        # signal_energy = compute_signal_energy(self.audio)
        # mean, variance = compute_statistical_features(self.audio)
        # mean_pitch, std_pitch = compute_pitch(self.audio, self.sr)
        # pulses = compute_pulses(self.audio, self.sr)
        # print(pulses)
        # spectrogram_db = compute_spectrogram(self.audio, self.sr)
        # envelopes = compute_envelopes(self.audio)
        return features
