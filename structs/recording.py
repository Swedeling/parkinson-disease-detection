from dataclasses import dataclass
from utils.preprocessing import silence_removing
import librosa
import os
from config import SR


@dataclass
class Recording:
    dir_path: str
    filename: str
    classname: int

    def __post_init__(self):
        self.filepath = os.path.join(self.dir_path, self.filename)

        try:
            self.audio, self.sr = librosa.load(self.filepath, sr=SR)
        except:
            self.audio = []

        self.length = len(self.audio)
        self.trimmed_recording = silence_removing(self.audio, self.filename)
