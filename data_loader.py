import os
from structs.recording import Recording
import pandas as pd
from config import AVAILABLE_LANGUAGES, RECORDINGS_DIR

LANGUAGE = "all"


class DataLoader:
    def __init__(self):
        self.language = LANGUAGE

        if self.language == "all":
            languages_to_load = ["italian", "polish"]
        elif self.language not in AVAILABLE_LANGUAGES:
            print("Language not available. I am using default language --> polish")
            languages_to_load = ["polish"]
        else:
            languages_to_load = [self.language]

        for language in languages_to_load:
            if language == "polish":
                self.a_pd_pol_recordings = self.load_recordings("PD", language, "a")
                self.a_hs_pol_recordings = self.load_recordings("HS", language, "a")

            if language == "italian":
                self.a_pd_itl_recordings = self.load_recordings("PD", language, "a")
                self.a_hs_itl_recordings = self.load_recordings("HS", language, "a")

        self.metadata = self.load_metadata(LANGUAGE)

    @staticmethod
    def load_recordings(label, language, vowel):
        data = []
        dir_path = os.path.join(RECORDINGS_DIR, language, "{}_{}".format(label, language))
        if label == "PD":
            classname = 1
        elif label == "HS":
            classname = 0
        else:
            dir_path, classname = "", None

        vowel_dir_path = os.path.join(dir_path, "recordings", vowel)

        for recording_name in os.listdir(vowel_dir_path):
            data.append(Recording(dir_path, vowel, str(recording_name), classname))
        return data

    @staticmethod
    def load_metadata(language):
        df = pd.read_excel("data/database_summary.xlsx")
        if language == "italian":
            return df[df['language'] == 'italian']
        if language == "polish":
            return df[df['language'] == 'polish']
        if language == "all":
            return df
        return None
