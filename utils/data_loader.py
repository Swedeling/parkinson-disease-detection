import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *
from structs.recording import Recording


class DataLoader:
    def __init__(self):
        self.settings = get_settings()
        self.languages_to_load = get_languages_to_load()

        # self.metadata = self.load_metadata(LANGUAGE)

    def load_recordings(self, label, vowel, mode):
        data = []
        for language in self.languages_to_load:
            dir_path = os.path.join(RECORDINGS_DIR, language, "{}_{}".format(label, language))

            if label == "PD":
                classname = 1
            elif label == "HC":
                classname = 0
            else:
                dir_path, classname = "", None
            vowel_dir_path = os.path.join(dir_path, "recordings", vowel, mode)

            ids = []
            limit = 40
            num = 0
            for recording_name in os.listdir(vowel_dir_path):
                # if "_".join(recording_name.split("_")[0:2]) not in ids:
                print(recording_name)
                recording = Recording(dir_path, vowel, str(recording_name), classname, self.settings, mode, language)
                data.append(recording)
                    # ids.append("_".join(recording_name.split("_")[0:2]))
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


def divide_data_into_subsets(class0_data, class1_data):
    class0_labels = len(class0_data) * [0]
    class1_labels = len(class1_data) * [1]

    data = class0_data + class1_data
    labels = class0_labels + class1_labels

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
    train_data = (np.array(x_train), np.array(y_train))

    if USE_VALIDATION_DATASET:
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.33, random_state=42)
        return train_data, (np.array(x_test), np.array(y_test)), (np.array(x_val), np.array(y_val))
    else:
        return train_data, (np.array(x_test), np.array(y_test))



