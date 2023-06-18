from classifiers.AlexNet import AlexNet
from classifiers.GNet import GNet
from config import *
from data_loader import DataLoader
from data_stats.db_stats import run_dataset_analysis
from utils.utils import set_cpu, set_gpu

import numpy as np
import os
import pandas as pd

if DEVICE == "GPU":
    set_gpu()
else:
    set_cpu()

if __name__ == "__main__":

    if RUN_DATASET_ANALYSIS:
        print("[INFO] Run dataset analysis...")
        run_dataset_analysis()

    spectrograms_results = {"settings": ["model_name", "accuracy", "precision", "recall", "f1-score"]}

    print("[INFO] Loading data...")
    data = DataLoader()

    for vowel in VOWELS_TO_LOAD:
        print("[INFO] Vowel: ", vowel)
        vowel_dir = os.path.join(RESULTS_DIR, vowel)
        if not os.path.exists(vowel_dir):
            os.mkdir(vowel_dir)

        hs_test_data = data.load_recordings("HS", vowel, "test")
        pd_test_data = data.load_recordings("PD", vowel, "test")

        hs_train_data = data.load_recordings("HS", vowel, "train")
        pd_train_data = data.load_recordings("PD", vowel, "train")

        hs_val_data = data.load_recordings("HS", vowel, "val")
        pd_val_data = data.load_recordings("PD", vowel, "val")

        print("Number of HS recordings: ", len(hs_test_data + hs_train_data + hs_val_data))
        print("Number of PD recordings: ", len(pd_test_data + pd_train_data + pd_val_data))

        test_data = hs_test_data + pd_test_data
        train_data = hs_train_data + pd_train_data
        val_data = hs_val_data + pd_val_data

        y_test = [0] * len(hs_test_data) + [1] * len(pd_test_data)
        y_train = [0] * len(hs_train_data) + [1] * len(pd_train_data)
        y_val = [0] * len(hs_val_data) + [1] * len(pd_val_data)

        for setting in data.settings:
            results = {}
            print("[INFO] Settings: ", setting)
            settings_dir = os.path.join(vowel_dir, setting)
            if not os.path.exists(settings_dir):
                os.mkdir(settings_dir)

            x_train = [recording.spectrograms[setting] for recording in train_data]
            x_test = [recording.spectrograms[setting] for recording in test_data]
            x_val = [recording.spectrograms[setting] for recording in val_data]

            train = (np.array(x_train), np.array(y_train))

            if USE_VALIDATION_DATASET:
                test = (np.array(x_test), np.array(y_test))
                val = (np.array(x_test), np.array(y_test))
            else:
                test = (np.array(x_test + x_val), np.array(y_test + y_val))
                val = False

            alex_net = AlexNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
            g_net = GNet(train, test, settings=setting, results_dir=settings_dir, val_data=val)
            for loss_fcn in LOSS_FUNCTION:
                for optimizer in OPTIMIZER:
                    for batch_size in BATCH_SIZE:
                        for epochs_number in EPOCHS_NUMBER:
                            alex_net_model = alex_net.run_classifier(loss_fcn, optimizer, batch_size, epochs_number)
                            g_net_model = g_net.run_classifier(loss_fcn, optimizer, batch_size, epochs_number)
                            model_params = "{}_{}_{}_{}".format(loss_fcn, optimizer, batch_size, epochs_number)
                            alex_net_model.save(os.path.join(MODELS_DIR, vowel, setting, "AlexNet", model_params))

            results = alex_net.results.copy()
            results.update(g_net.results)

            results_df = pd.DataFrame(results)

            acc_row = results_df.drop('Model', axis=1).loc[5].astype(float)
            max_acc_value = acc_row.max()
            model_name = acc_row.idxmax()

            model_settings = '_'.join(model_name.rsplit("_", maxsplit=5)[1:])
            precision, recall, f1 = results_df[model_name].loc[6], \
                                    results_df[model_name].loc[7], \
                                    results_df[model_name].loc[8]

            spectrograms_results[setting] = [model_settings, max_acc_value, precision, recall, f1]
            results_df.to_excel(os.path.join(settings_dir, "summary.xlsx"))

        spectrograms_results_df = pd.DataFrame(spectrograms_results)
        spectrograms_results_df.to_excel(os.path.join(RESULTS_DIR, vowel, "spectrograms_summary.xlsx"))
