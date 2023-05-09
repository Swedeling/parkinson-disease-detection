from tensorboard.compat.proto.config_pb2 import GPUOptions
import os
from classifiers.AlexNet import AlexNet
from config import RUN_DATASET_ANALYSIS, SR
import tensorflow as tf
from data_loader import DataLoader
from data_stats.db_stats import run_dataset_analysis
from data_stats.db_stats import save_recording_len_to_df

from utils.divide_records import divide_records
from utils.rename_recordings import rename_polish_recordings_by_info_file, encode_italian_recordings
from utils.audio_spectrogram import Spectrogram
import cv2
import librosa
import random
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import numpy as np

# GPUOptions.allow_growth = True
#
# print(device_lib.list_local_devices())
# tf.device('CPU:0')
device_name = 'GPU:0'
shape = (int(1500), 1500)


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

        itl_pd_train, itl_pd_test = train_test_split(data.a_pd_itl_recordings, test_size=0.2, random_state=42)
        itl_hs_train, itl_hs_test = train_test_split(data.a_hs_itl_recordings, test_size=0.2, random_state=42)
        pol_pd_train, pol_pd_test = train_test_split(data.a_pd_pol_recordings, test_size=0.2, random_state=42)
        pol_hs_train, pol_hs_test = train_test_split(data.a_hs_pol_recordings, test_size=0.2, random_state=42)

        print(len(itl_pd_train))
        print(len(itl_hs_train))
        print(len(pol_pd_train))
        print(len(pol_hs_train))


        # a_pd_spectrograms = [recording.spectrogram for recording in data.a_pd_off_recordings]
        # a_hs_spectrograms = [recording.spectrogram for recording in data.a_hs_recordings]
        #
        # pd_train, pd_test = train_test_split(a_pd_spectrograms, test_size=0.2, random_state=42)
        # hs_train, hs_test = train_test_split(a_hs_spectrograms, test_size=0.2, random_state=42)
        #
        # X_train = pd_train + hs_train
        # X_test = pd_test + hs_test
        #
        # y_train = [1] * len(pd_train) + [0] * len(hs_train)
        # y_test = [1] * len(pd_test) + [0] * len(hs_test)
        #
        # zipped = list(zip(X_train, y_train))
        # random.shuffle(zipped)
        # X_train, y_train = zip(*zipped)
        #
        # zipped = list(zip(X_test, y_test))
        # random.shuffle(zipped)
        # X_test, y_test = zip(*zipped)
        #
        # print(len(X_train), len(X_test))

#         train_data = (np.array(X_train), np.array(y_train))
#         test_data = (np.array(X_test), np.array(y_test))
#
#         print(X_train[0].shape)
#
#         print(len(train_data))
#         alex_net = AlexNet(train_data, test_data)
#         alex_net.run_classifier()
# #
#     # save_recording_len_to_df(data_loader.a_pd_off_recordings)
#     #
#     # divide_records("data/source/path)
#     # rename_polish_recordings_by_info_file("data/path)
