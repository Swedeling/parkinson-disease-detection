import numpy as np
import pandas as pd
from classifiers.utils import initialize_classifiers
from config import *
from data_stats.db_stats import run_dataset_analysis
from utils.data_loader import DataLoader
from utils.utils import set_cpu, set_gpu, mix_lists, prepare_datasets
from utils.results_analysis import summarize_all_results, prepare_plots
from sklearn.model_selection import StratifiedKFold
from structs.recording import AUGUMENTATION

if DEVICE == "GPU":
    set_gpu()
else:
    set_cpu()


if __name__ == "__main__":
    if RUN_DATASET_ANALYSIS:
        print("[INFO] Run dataset analysis...")
        run_dataset_analysis([LANGUAGE_TO_LOAD])

    spectrograms_results = {"settings": ["model_name", "accuracy", "precision", "recall", "f1-score"]}

    print("[INFO] Loading data...")
    data = DataLoader()

    for vowel in VOWELS_TO_LOAD:
        print("[INFO] Vowel: ", vowel)
        vowel_dir = os.path.join(RESULTS_DIR, vowel)
        if not os.path.exists(vowel_dir):
            os.mkdir(vowel_dir)

        x_train_data, y_train_data = prepare_datasets(data, vowel, "train")
        x_test, y_test = prepare_datasets(data, vowel, "test")

        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
        x_test, y_test = np.array(x_test), np.array(y_test)

        print(x_test[0].filename, x_test[0].classname, y_test[0])

        speakers_ids = []
        speakers_classes = []

        for recording in x_train_data:
            if "_".join(recording.filename.split("_")[0:2]) not in speakers_ids:
                speakers_ids.append("_".join(recording.filename.split("_")[0:2]))
                speakers_classes.append(recording.classname)

        speakers_ids = np.array(speakers_ids)
        speakers_classes = np.array(speakers_classes)

        kf = StratifiedKFold(n_splits=10, shuffle=True)
        cv_train_data, cv_val_data = [], []
        cv_train_labels, cv_val_labels = [], []

        for train_idx, val_idx in kf.split(speakers_ids, speakers_classes):
            train_data, val_data = [], []
            train_labels, val_labels = [], []
            X_train, X_val = speakers_ids[train_idx], speakers_ids[val_idx]

            for recording in x_train_data:
                if "_".join(recording.filename.split("_")[0:2]) in X_train:
                    train_data.append(recording)
                    train_labels.append(recording.classname)

                if "_".join(recording.filename.split("_")[0:2]) in X_val:
                    val_data.append(recording)
                    val_labels.append(recording.classname)

            cv_train_data.append(train_data)
            cv_val_data.append(val_data)
            cv_train_labels.append(train_labels)
            cv_val_labels.append(val_labels)

        for setting in data.settings:
            results = {}
            print("[INFO] Settings: ", setting)
            settings_dir = os.path.join(vowel_dir, setting)
            print(settings_dir)
            if not os.path.exists(settings_dir):
                os.mkdir(settings_dir)

            cv_spectrograms_train = []
            cv_labels_train = []
            for x_train, y_train in zip(cv_train_data, cv_train_labels):
                x_train_spectrograms = [recording.spectrograms[setting] for recording in x_train]
                flattened_list = [item for sublist in x_train_spectrograms for item in sublist]
                x_train_spectrograms = np.array([spectrogram for spectrogram in flattened_list])
                y_train = np.array([item for item in y_train for _ in range(len(AUGUMENTATION))])
                cv_labels_train.append(y_train)
                cv_spectrograms_train.append(x_train_spectrograms)

            cv_spectrograms_val = []
            cv_labels_val = []
            for x_val, y_val in zip(cv_val_data, cv_val_labels):
                x_val_spectrograms = [recording.spectrograms[setting] for recording in x_val]
                flattened_list = [item for sublist in x_val_spectrograms for item in sublist]
                x_val_spectrograms = np.array([spectrogram for spectrogram in flattened_list])  #.astype('float32') / 255
                y_val = np.array([item for item in y_val for _ in range(len(AUGUMENTATION))])
                cv_labels_val.append(y_val)
                cv_spectrograms_val.append(x_val_spectrograms)

            x_test = [recording.spectrograms[setting] for recording in x_test]
            x_test = [sublist[0] for sublist in x_test]

            x_test = np.array([spectrogram for spectrogram in x_test])
            y_test = np.array(y_test)

            print("Train dataset size: ", len(cv_spectrograms_train[0]), len(cv_labels_train[0]))
            print("Test dataset size: ", len(x_test), len(y_test))
            print("Val dataset size: ", len(cv_spectrograms_val[0]), len(cv_labels_val[0]))

            train = (np.array(cv_spectrograms_train), np.array(cv_labels_train))
            test = (x_test, y_test)
            val = (cv_spectrograms_val, cv_labels_val)

            classifiers = initialize_classifiers(train, test, setting, settings_dir, val)

            batch_size = BATCH_SIZES[0]
            optimizer = OPTIMIZERS[0]

            for cls_name, cls in classifiers.items():
                print(cls_name)
                cls.run_classifier(LOSS_FUNCTION, optimizer, batch_size)
                model_params = "{}_{}_{}".format(LOSS_FUNCTION, optimizer, batch_size)
                cls.model.save(os.path.join(MODELS_DIR, vowel, setting, cls_name, model_params))
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


        # summarize_all_results()
    #
    # # prepare_plots()
