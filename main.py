from classifiers.AlexNet import AlexNet
from classifiers.GNet import GNet
from classifiers.InceptionV3 import InceptionNet
from classifiers.ResNet50 import ResNet
from classifiers.VGGNet import VGGNet
from classifiers.LeNet5 import LeNet5
import random
from config import *
from utils.data_loader import DataLoader
from data_stats.db_stats import run_dataset_analysis
from utils.utils import set_cpu, set_gpu
from utils.results_analysis import summarize_all_results, prepare_plots

import numpy as np
import os
import pandas as pd

if DEVICE == "GPU":
    set_gpu()
else:
    set_cpu()


def initialize_classifiers(train, test, setting, settings_dir, val):
    classifiers = {}
    for cls in CLASSIFIERS_TO_TEST:
        if cls == "AlexNet":
            classifiers[cls] = AlexNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
        elif cls == "GNet":
            classifiers[cls] = GNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
        elif cls == "InceptionV3":
            classifiers[cls] = InceptionNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
        elif cls == "LeNet-5":
            classifiers[cls] = LeNet5(train, test, settings=setting, results_dir=settings_dir, val_data=val)
        elif cls == "ResNet50":
            classifiers[cls] = ResNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
        elif cls == "VGGNet":
            classifiers[cls] = VGGNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
    return classifiers


if __name__ == "__main__":

    if RUN_DATASET_ANALYSIS:
        print("[INFO] Run dataset analysis...")
        run_dataset_analysis(["polish"])

    spectrograms_results = {"settings": ["model_name", "accuracy", "precision", "recall", "f1-score"]}

    print("[INFO] Loading data...")
    data = DataLoader()

    for vowel in VOWELS_TO_LOAD:
        print("[INFO] Vowel: ", vowel)
        vowel_dir = os.path.join(RESULTS_DIR, vowel)
        if not os.path.exists(vowel_dir):
            os.mkdir(vowel_dir)

        hs_test_data_mrps = data.load_recordings("HS", vowel, "test")
        hs_test_data_orps = data.load_recordings("HS", vowel, "test - orps")
        pd_test_data = data.load_recordings("PD", vowel, "test")

        hs_train_data = data.load_recordings("HS", vowel, "train")
        pd_train_data = data.load_recordings("PD", vowel, "train")

        hs_val_data = data.load_recordings("HS", vowel, "val")
        pd_val_data = data.load_recordings("PD", vowel, "val")

        print("Number of HS recordings [ORPS]: ", len(hs_test_data_orps), len(hs_train_data), len(hs_val_data))
        print("Number of HS recordings [MRPS]: ", len(hs_test_data_mrps), len(hs_train_data), len(hs_val_data))
        print("Number of PD recordings: ", len(pd_test_data + pd_train_data + pd_val_data))

        test_data_orps = hs_test_data_orps + pd_test_data
        test_data_mrps = hs_test_data_mrps + pd_test_data
        train_data = hs_train_data + pd_train_data
        val_data = hs_val_data + pd_val_data

        y_test_orps = [0] * len(hs_test_data_orps) + [1] * len(pd_test_data)
        y_test_mrps = [0] * len(hs_test_data_mrps) + [1] * len(pd_test_data)

        y_train = [0] * len(hs_train_data) + [1] * len(pd_train_data)
        y_val = [0] * len(hs_val_data) + [1] * len(pd_val_data)

        print(y_test_mrps)
        print(y_train)

        temp = list(zip(test_data_orps, y_test_orps))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        test_data_orps, y_test_orps = list(res1), list(res2)

        temp = list(zip(test_data_mrps, y_test_mrps))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        test_data_mrps, y_test_mrps = list(res1), list(res2)


        temp = list(zip(train_data, y_train))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        train_data, y_train = list(res1), list(res2)


        temp = list(zip(val_data, y_val))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        val_data, y_val = list(res1), list(res2)

        print(y_test_mrps)
        print(y_train)


        for setting in data.settings:
            results = {}
            print("[INFO] Settings: ", setting)
            settings_dir = os.path.join(vowel_dir, setting)
            if not os.path.exists(settings_dir):
                os.mkdir(settings_dir)

            x_train = [recording.spectrograms[setting] for recording in train_data]
            x_test_orps = [recording.spectrograms[setting] for recording in test_data_orps]
            x_test_mrps = [recording.spectrograms[setting] for recording in test_data_mrps]
            x_val = [recording.spectrograms[setting] for recording in val_data]

            train = (np.array(x_train), np.array(y_train))

            if USE_VALIDATION_DATASET:
                test_orps = (np.array(x_test_orps), np.array(y_test_orps))
                test_mrps = (np.array(x_test_mrps), np.array(y_test_mrps))
                val = (np.array(x_val), np.array(y_val))
            else:
                test_orps = (np.array(x_test_orps + x_val), np.array(y_test_orps + y_val))
                test_mrps = (np.array(x_test_mrps + x_val), np.array(y_test_mrps + y_val))
                val = False

            test = {"orps": test_orps, "mrps": test_mrps}

            classifiers = initialize_classifiers(train, test, setting, settings_dir, val)
            orps_results = {}
            mrps_results = {}


            for loss_fcn in LOSS_FUNCTION:
                for optimizer in OPTIMIZER:
                    for batch_size in BATCH_SIZE:
                        for cls_name, cls in classifiers.items():
                            print(cls_name)
                            cls.run_classifier(loss_fcn, optimizer, batch_size)
                            model_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size)
                            cls.model.save(os.path.join(MODELS_DIR, vowel, setting, cls_name, model_params))
                            orps_results.update(cls.orps_results)
                            mrps_results.update(cls.mrps_results)

            summary_orps_path = os.path.join(settings_dir, "summary_orps.xlsx")
            if os.path.exists(summary_orps_path):
                results_df_orps = pd.read_excel(summary_orps_path)
                for model_name, values in orps_results.items():
                    if model_name not in results_df_orps.columns:
                        results_df_orps[model_name] = values
            else:
                results_df_orps = pd.DataFrame(orps_results)
            results_df_orps.to_excel(summary_orps_path)

            summary_mrps_path = os.path.join(settings_dir, "summary_mrps.xlsx")
            if os.path.exists(summary_mrps_path):
                results_df_mrps = pd.read_excel(summary_mrps_path)
                for model_name, values in mrps_results.items():
                    if model_name not in results_df_mrps.columns:
                        results_df_mrps[model_name] = values
            else:
                results_df_mrps = pd.DataFrame(mrps_results)
            results_df_mrps.to_excel(summary_mrps_path)

    summarize_all_results("mrps")
    summarize_all_results("orps")

    # prepare_plots()
