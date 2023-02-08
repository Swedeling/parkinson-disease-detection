import os
import pandas as pd

DATA_ROOT_DIR = "data"


class DataLoader:
    def __init__(self):
        self.metadata = pd.read_excel(os.path.join(DATA_ROOT_DIR, "sustained_polish/UPDRS_description.xlsx"), nrows=28)

        self.recording_no, self.disease_length, self.age = None, None, None
        self.updrs_off, self.updrs_30, self.updrs_60, self.updrs_120, self.updrs_180 = None, None, None, None, None

        self.prepare_labels()

    def prepare_labels(self):
        self.recording_no = self.metadata["recording no"]
        self.age = self.metadata["age"]
        self.updrs_off = self.metadata["UPDRS OFF"]
        self.updrs_30 = self.metadata["UPDRS 30"]
        self.updrs_60 = self.metadata["UPDRS 60"]
        self.updrs_120 = self.metadata[" UPDRS 120"]
        self.updrs_180 = self.metadata[" UPDRS 180"]
        self.disease_length = self.metadata["disease length [years]"]

    def load_data(self):
        pass

