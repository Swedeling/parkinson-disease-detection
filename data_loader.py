import os
import librosa
import pandas as pd
from pydub import AudioSegment

DATA_ROOT_DIR = "data"
SR = 44100
PD_RECORDINGS_DIR = os.path.join(DATA_ROOT_DIR, "AGH-dataset", "Polish_audio_03_10_2022", "Polish_raw")
HS_RECORDINGS_DIR = os.path.join(DATA_ROOT_DIR, "polish_audio_sustained_HS")


class DataLoader:
    def __init__(self):
        self.a_pd_off_recordings, self.a_pd_off_labels = self.load_recordings("PD", "a")
        self.e_pd_off_recordings, self.e_pd_off_labels = self.load_recordings("PD", "e")
        self.i_pd_off_recordings, self.i_pd_off_labels = self.load_recordings("PD", "i")
        self.o_pd_off_recordings, self.o_pd_off_labels = self.load_recordings("PD", "o")
        self.u_pd_off_recordings, self.u_pd_off_labels = self.load_recordings("PD", "u")

        self.a_hs_recordings, self.a_hs_labels = self.load_recordings("HS", "a")
        self.e_hs_recordings, self.e_hs_labels = self.load_recordings("HS", "e")
        self.i_hs_recordings, self.i_hs_labels = self.load_recordings("HS", "i")
        self.o_hs_recordings, self.o_hs_labels = self.load_recordings("HS", "o")
        self.u_hs_recordings, self.u_hs_labels = self.load_recordings("HS", "u")

        self.metadata = self.load_metadata()

    @staticmethod
    def load_recordings(mode, vowel):
        data, labels = [], []

        if mode == "PD":
            vowel_dir_path = os.path.join(PD_RECORDINGS_DIR, "off", vowel)
        elif mode == "HS":
            vowel_dir_path = os.path.join(HS_RECORDINGS_DIR, vowel)
        else:
            vowel_dir_path = []

        for recording_name in os.listdir(vowel_dir_path):
            if recording_name.endswith("m4a"):
                convert_file_extension_into_wav(vowel_dir_path, recording_name)

            try:
                signal, sr = librosa.load(os.path.join(vowel_dir_path, recording_name.split('.')[0] + '.wav'), sr=SR)
                data.append(signal)
                labels.append(recording_name)

            except:
                print("Problem with loading file: {}".format(recording_name))

        return data, labels

    @staticmethod
    def load_metadata():
        return pd.read_excel(os.path.join("data/AGH-dataset", "sustained_polish/UPDRS_description.xlsx"), nrows=28)


def convert_file_extension_into_wav(dir_path, filename, overwrite=False):
    formats_to_convert = ['.m4a']

    if filename.endswith(tuple(formats_to_convert)):
        (path, file_extension) = os.path.splitext(filename)
        file_extension = file_extension.replace('.', '')
        wav_filename = filename.replace(file_extension, 'wav')
        wav_file_path = os.path.join(dir_path, wav_filename)

        if not os.path.exists(wav_file_path) or overwrite is True:
            try:
                track = AudioSegment.from_file(os.path.join(dir_path, filename), format=file_extension)
                print('CONVERTING: ' + str(wav_file_path))
                file_handle = track.export(wav_file_path, format='wav')
            except:
                print("Problem with converting file: {}".format(filename))
