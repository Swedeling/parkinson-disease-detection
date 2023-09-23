import pandas as pd

from classifiers.classifiers_init import initialize_classifiers
from config import *
from utils.data_loader import DataLoader
from utils.utils import set_cpu, set_gpu
from utils.results_saver import summarize_all_results

if DEVICE == "GPU":
    set_gpu()
else:
    set_cpu()


if __name__ == "__main__":

    print("[INFO] Loading data...")
    data_loader = DataLoader()

    for vowel in VOWELS_TO_ANALYZE:
        print("[INFO] Vowel: ", vowel)
        results_vowel_dir = os.path.join(RESULTS_DIR, vowel)
        if not os.path.exists(results_vowel_dir):
            os.mkdir(results_vowel_dir)

        # Load data
        x_train, y_train = data_loader.prepare_datasets(vowel, "train")
        cv_train, cv_val = data_loader.prepare_data_for_cross_validation(x_train)
        test = data_loader.prepare_datasets(vowel, "test")

        for setting in data_loader.settings:
            results = {}
            print("[INFO] Settings: ", setting)
            results_settings_dir = os.path.join(results_vowel_dir, setting)
            if not os.path.exists(results_settings_dir):
                os.mkdir(results_settings_dir)

            # Get cross validation data with specific settings
            train = data_loader.get_cross_validation_data_with_settings(cv_train, setting)
            val = data_loader.get_cross_validation_data_with_settings(cv_val, setting)
            test = data_loader.get_test_data_with_settings(test, setting)

            # Initialize classifiers
            classifiers = initialize_classifiers(train, test, val, setting, results_settings_dir)

            # Run experiments
            for cls_name, cls in classifiers.items():
                print("[INFO] Classifier: ", cls_name)
                cls.run_classifier()
                results.update(cls.results)

            # Saving results
            print("[INFO] Saving results...")
            summary_path = os.path.join(results_settings_dir, "summary.xlsx")
            if os.path.exists(summary_path):
                results_df = pd.read_excel(summary_path)
                for model_name, values in results.items():
                    if model_name not in results_df.columns:
                        results_df[model_name] = values
            else:
                results_df = pd.DataFrame(results)
            results_df.to_excel(summary_path)
            print("---------------------------------------")

        print("[INFO] Summarize results...")
        summarize_all_results()
