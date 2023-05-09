from dataclasses import dataclass
from utils.preprocessing import silence_removing
import librosa
import os
from config import SR
import wave
import numpy as np
import shutil
from utils.audio_spectrogram import Spectrogram

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
        except:
            self.audio = []

        self.length = len(self.audio)
        self.trimmed_audio = self._trim_recording()
        # self.spectrogram = self._generate_spectrogram()

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

        top = bottom = 10
        left = 123
        right = 124

        img_with_border = np.pad(img, ((top, bottom), (left, right)), mode='constant')

        return img_with_border
