from config import *
from structs.recording import Recording
from .utils import get_settings, mix_lists
from sklearn.model_selection import StratifiedKFold

import numpy as np


class DataLoader:
    def __init__(self):
        self.settings = get_settings(N_MELS, BINSIZE, OVERLAP, SPECTROGRAMS, MELSPECTROGRAMS)
        self.languages_to_load = get_languages_to_load(LANGUAGES, AVAILABLE_LANGUAGES)

    def prepare_datasets(self, vowel, mode):
        hc_data = self.load_recordings(HC, vowel, mode)
        pd_data = self.load_recordings(PD, vowel, mode)
        x_data = pd_data + hc_data
        y_data = [CLASSES[HC]] * len(hc_data) + [CLASSES[PD]] * len(pd_data)
        x_data, y_data = mix_lists(x_data, y_data)
        return np.array(x_data), np.array(y_data)

    def load_recordings(self, label, vowel, mode):
        data = []
        classname = CLASSES.get(label, None)

        if classname is None:
            return data

        for language in self.languages_to_load:
            dir_path = os.path.join(RECORDINGS_DIR, language, "{}_{}".format(label, language))
            vowel_dir_path = os.path.join(dir_path, "recordings", vowel, mode)

            for recording_name in os.listdir(vowel_dir_path):
                recording = Recording(dir_path, vowel, recording_name, classname, self.settings, mode, language)
                data.append(recording)

        return data

    @staticmethod
    def prepare_data_for_cross_validation(data):
        speakers_ids = []
        speakers_classes = []

        for recording in data:
            if "_".join(recording.filename.split("_")[0:2]) not in speakers_ids:
                speakers_ids.append("_".join(recording.filename.split("_")[0:2]))
                speakers_classes.append(recording.classname)

        speakers_ids = np.array(speakers_ids)
        speakers_classes = np.array(speakers_classes)

        kf = StratifiedKFold(n_splits=CROSS_VALIDATION_SPLIT, shuffle=True)
        cv_x_train, cv_x_val = [], []
        cv_y_train, cv_y_val = [], []

        for train_indexes, val_indexes in kf.split(speakers_ids, speakers_classes):
            train_data, val_data = [], []
            train_labels, val_labels = [], []

            x_train = speakers_ids[train_indexes]
            x_val = speakers_ids[val_indexes]

            for recording in data:
                if "_".join(recording.filename.split("_")[0:2]) in x_train:
                    train_data.append(recording)
                    train_labels.append(recording.classname)

                if "_".join(recording.filename.split("_")[0:2]) in x_val:
                    val_data.append(recording)
                    val_labels.append(recording.classname)

            cv_x_train.append(train_data)
            cv_x_val.append(val_data)
            cv_y_train.append(train_labels)
            cv_y_val.append(val_labels)

        cv_train = cv_x_train, cv_y_train
        cv_val = cv_x_val, cv_y_val
        return cv_train, cv_val

    @staticmethod
    def get_cross_validation_data_with_settings(data, setting):
        cv_spectrograms = []
        cv_labels = []
        for x, y in zip(data[0], data[1]):
            spectrograms = [recording.spectrograms[setting] for recording in x]
            spectrograms_list_flattened = np.array([item for sublist in spectrograms for item in sublist])
            labels = np.array([item for item in y for _ in range(len(AUGMENTATION))])

            cv_spectrograms.append(spectrograms_list_flattened)
            cv_labels.append(labels)

        return cv_spectrograms, cv_labels

    @staticmethod
    def get_test_data_with_settings(data, setting):
        x_test = [recording.spectrograms[setting] for recording in data[0]]
        x_test = [sublist[0] for sublist in x_test]     # no augmentation in test dataset - only filtered signals
        x_test = np.array(x_test)
        y_test = np.array(data[1])
        return x_test, y_test
