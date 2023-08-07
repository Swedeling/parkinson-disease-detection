import numpy as np
import os
import pandas as pd
import random

from classifiers.utils import initialize_classifiers
from config import *
from data_stats.db_stats import run_dataset_analysis
from utils.data_loader import DataLoader
from utils.utils import set_cpu, set_gpu, mix_lists, prepare_datasets
from utils.results_analysis import summarize_all_results, prepare_plots


if DEVICE == "GPU":
    set_gpu()
else:
    set_cpu()


if __name__ == "__main__":
    if RUN_DATASET_ANALYSIS:
        print("[INFO] Run dataset analysis...")
        run_dataset_analysis([LANGUAGE_TO_LOAD])

    spectrograms_results = {
        "settings": ["model_name", "accuracy", "precision", "recall", "f1-score"]
    }

    print("[INFO] Loading data...")
    data = DataLoader()

    for vowel in VOWELS_TO_LOAD:
        print("[INFO] Vowel: ", vowel)
        vowel_dir = os.path.join(RESULTS_DIR, vowel)
        if not os.path.exists(vowel_dir):
            os.mkdir(vowel_dir)

        train_data, y_train = prepare_datasets(data, vowel, "train")
        test_data, y_test = prepare_datasets(data, vowel, "test")
        val_data, y_val = prepare_datasets(data, vowel, "val")
        
        print("Train dataset size: ", len(y_train))
        print("Test dataset size: ", len(y_test))
        print("Val dataset size: ", len(y_val))

        for setting in data.settings:
            results = {}
            print("[INFO] Settings: ", setting)
            settings_dir = os.path.join(vowel_dir, setting)
            print(settings_dir)
            if not os.path.exists(settings_dir):
                os.mkdir(settings_dir)
                            
            x_train = [recording.spectrograms[setting] for recording in train_data]
            x_test = [recording.spectrograms[setting] for recording in test_data]
            x_val = [recording.spectrograms[setting] for recording in val_data]

            train = (np.array(x_train), np.array(y_train))
            test = (np.array(x_test), np.array(y_test))
            val = (np.array(x_val), np.array(y_val))

            classifiers = initialize_classifiers(train, test, setting, settings_dir, val)

            for loss_fcn in LOSS_FUNCTIONS:
                for optimizer in OPTIMIZERS:
                    for batch_size in BATCH_SIZES:
                        for cls_name, cls in classifiers.items():
                            print(cls_name)
                            cls.run_classifier(loss_fcn, optimizer, batch_size)
                            model_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size)
                            # cls.model.save(os.path.join(MODELS_DIR, vowel, setting, cls_name, model_params))
                            results.update(cls.results)

            summary_path = os.path.join(settings_dir, "summary.xlsx")
            if os.path.exists(summary_path):
                results_df = pd.read_excel(summary_path)
                for model_name, values in results.items():
                    if model_name not in results_df.columns:
                        results_df[model_name] = values
            else:
                results_df = pd.DataFrame(results)
            results_df.to_excel(summary_path)


        summarize_all_results()

    # # prepare_plots()
