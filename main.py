from classifiers.AlexNet import AlexNet
from config import RUN_DATASET_ANALYSIS
from data_loader import DataLoader

from data_stats.db_stats import run_dataset_analysis
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


# GPUOptions.allow_growth = True
# print(device_lib.list_local_devices())
device_name = 'CPU:0'


if __name__ == "__main__":
    with tf.device(device_name):

        if RUN_DATASET_ANALYSIS:
            print("[INFO] Run dataset analysis...")
            run_dataset_analysis()

        print("[INFO] Loading data...")
        data = DataLoader()
        print(len(data.a_hs_pol_recordings))
        print(len(data.a_pd_pol_recordings))
        print(len(data.a_hs_itl_recordings))
        print(len(data.a_pd_itl_recordings))

        a_pd_spectrograms = [recording.spectrogram for recording in data.a_pd_pol_recordings]
        a_hs_spectrograms = [recording.spectrogram for recording in data.a_hs_pol_recordings]

        pd_train, pd_test = train_test_split(a_pd_spectrograms, test_size=0.3, random_state=42)
        pd_test, pd_val = train_test_split(pd_test, test_size=0.33, random_state=42)

        hs_train, hs_test = train_test_split(a_hs_spectrograms, test_size=0.3, random_state=42)
        hs_test, hs_val = train_test_split(hs_test, test_size=0.33, random_state=42)

        X_train = pd_train + hs_train
        X_test = pd_test + hs_test
        X_val = pd_val + hs_val

        y_train = [1] * len(pd_train) + [0] * len(hs_train)
        y_test = [1] * len(pd_test) + [0] * len(hs_test)
        y_val = [1] * len(pd_val) + [0] * len(hs_val)

        zipped = list(zip(X_train, y_train))
        random.shuffle(zipped)
        X_train, y_train = zip(*zipped)

        zipped = list(zip(X_test, y_test))
        random.shuffle(zipped)
        X_test, y_test = zip(*zipped)

        zipped = list(zip(X_val, y_val))
        random.shuffle(zipped)
        X_val, y_val = zip(*zipped)

        train_data = (np.array(X_train), np.array(y_train))
        test_data = (np.array(X_test), np.array(y_test))
        val_data = (np.array(X_val), np.array(y_val))

        alex_net = AlexNet(train_data, test_data, val_data)
        alex_net.run_classifier()
#
#     # save_recording_len_to_df(data_loader.a_pd_off_recordings)
#     # divide_records("data/source/path)
#     # rename_polish_recordings_by_info_file("data/path)
