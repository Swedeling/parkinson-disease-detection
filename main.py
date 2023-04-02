from data_loader import DataLoader
from data_stats.data_stats import run_dataset_analysis
from utils.rename_records import rename_italian_recordings_by_info_file
from utils.audio_player import plot_voice_recording
from utils.divide_records import divide_records

if __name__ == "__main__":

    print("[INFO] Loading data...")
    # data_loader = DataLoader()
    # run_dataset_analysis()

    rename_italian_recordings_by_info_file("data/Italian Parkinson's Voice and speech/TEST")



