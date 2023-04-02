from data_loader import DataLoader
from data_stats.data_stats import run_dataset_analysis
from utils.rename_recordings import rename_polish_recordings_by_info_file
from utils.audio_player import plot_voice_recording
from utils.divide_records import divide_records

if __name__ == "__main__":

    # print("[INFO] Loading data...")
    # data_loader = DataLoader()
    print("[INFO] Run dataset analysis...")
    run_dataset_analysis()


