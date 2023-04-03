from data_loader import DataLoader
from data_stats.db_stats import run_dataset_analysis
from data_stats.db_stats import save_recording_len_to_df
from config import RUN_DATASET_ANALYSIS


if __name__ == "__main__":
    if RUN_DATASET_ANALYSIS:
        print("[INFO] Run dataset analysis...")
        run_dataset_analysis()

    print("[INFO] Loading data...")
    data_loader = DataLoader()
    save_recording_len_to_df(data_loader.a_pd_off_recordings)
