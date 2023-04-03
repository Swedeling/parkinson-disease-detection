import os
from structs.recording import Recording
import pandas as pd
from config import HS_RECORDINGS_DIR, PD_RECORDINGS_DIR, MODE


class DataLoader:
    def __init__(self):
        self.a_pd_off_recordings = self.load_recordings("PD", "a")
        # self.a_pd_off_recordings = self.load_recordings("PD", "a")
        # self.e_pd_off_recordings = self.load_recordings("PD", "e")
        # self.i_pd_off_recordings = self.load_recordings("PD", "i")
        # self.o_pd_off_recordings = self.load_recordings("PD", "o")
        # self.u_pd_off_recordings = self.load_recordings("PD", "u")
        # self.a_hs_recordings = self.load_recordings("HS", "a")
        # self.e_hs_recordings = self.load_recordings("HS", "e")
        # self.i_hs_recordings = self.load_recordings("HS", "i")
        # self.o_hs_recordings = self.load_recordings("HS", "o")
        # self.u_hs_recordings = self.load_recordings("HS", "u")

        self.metadata = self.load_metadata(MODE)

    @staticmethod
    def load_recordings(mode, vowel):
        data = []

        if mode == "PD":
            vowel_dir_path = os.path.join(PD_RECORDINGS_DIR, vowel)
            classname = 1
        elif mode == "HS":
            vowel_dir_path = os.path.join(HS_RECORDINGS_DIR, vowel)
            classname = 0
        else:
            vowel_dir_path, classname = "", None

        for recording_name in os.listdir(vowel_dir_path):
            data.append(Recording(vowel_dir_path, str(recording_name), classname))
        return data

    @staticmethod
    def load_metadata(mode):
        df = pd.read_excel("data/database_summary.xlsx")
        if mode == "italian":
            return df[df['language'] == 'italian']
        if mode == "polish":
            return df[df['language'] == 'polish']
        if mode == "all":
            return df
        return None
