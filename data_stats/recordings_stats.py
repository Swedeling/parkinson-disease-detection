from data_stats.db_stats import _get_data_info


def get_recording_len(recording):
    return len(recording)


def save_recording_len_to_df(data_path):
    df = _get_data_info(data_path, "polish")
    print(df)


def compute_recording_stats(database):
    for recording, filename in database:
        print(len(recording))
