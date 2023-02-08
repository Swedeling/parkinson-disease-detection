import os
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_ROOT_DIR = "data"


class MetadataLoader:
    def __init__(self):
        self.metadata = pd.read_excel(os.path.join(DATA_ROOT_DIR, "sustained_polish/UPDRS_description.xlsx"), nrows=28).astype(int)

        self.features_names = ['UPDRS OFF', 'UPDRS 30', 'UPDRS 60', 'UPDRS 120', 'UPDRS 180']
        self.class_names = ['very_mild', 'mild', 'moderate', 'intermediate', 'advanced', 'very_advanced']

        self.labels = self._set_labels()
        self.features = self._prepare_data()

        self.x_train, self.x_test, self.y_train, self.y_test = self._split_data()

    def _set_labels(self):
        disease_len = self.metadata["disease length [years]"]
        return list(map(self._categorize_disease_len, disease_len))

    @staticmethod
    def _categorize_disease_len(disease_len):
        if 0 < disease_len < 5:
            return 0
        if 5 <= disease_len < 10:
            return 1
        if 10 <= disease_len < 15:
            return 2
        if 15 <= disease_len < 20:
            return 3
        if 20 <= disease_len < 25:
            return 4
        if 25 <= disease_len:
            return 5

    def _prepare_data(self):
        return self.metadata[self.features_names]

    def _split_data(self):
        return train_test_split(self.features, self.labels, test_size=0.33, random_state=42)
