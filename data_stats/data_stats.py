import os
import pandas as pd

DATA_PATH = "data/polish_audio_sustained_HS"


def run_dataset_analysis():
    _get_data_info()


def _get_data_info():
    dirs_list = os.listdir(DATA_PATH)
    db_info = {"filename": [], "initials": [], "id": [], "sex": [], "age": [], "vowel": [], "rec_num": []}
    names = ["AO", "TO", "AD", "BS", "AA", "AB", "BA", "BB", "BC", "KR", "LC"]

    for dirname in dirs_list:
        if dirname in ["a", "e", "i", "o", "u"]:
            files_list = os.listdir(os.path.join(DATA_PATH, dirname))
            for file in files_list:
                (filename, file_extension) = os.path.splitext(file)
                if file_extension != ".wav":
                    os.remove(os.path.join(DATA_PATH, dirname, filename + file_extension))

                info = filename.split("_")

                if len(info) == 5:
                    db_info["filename"].append(filename)
                    db_info["initials"].append(info[0])
                    db_info["id"].append(names.index(info[0]) + 1)
                    db_info["sex"].append(info[1])
                    db_info["age"].append(info[2])
                    db_info["vowel"].append(info[3])
                    db_info["rec_num"].append(info[4])
                else:
                    print("Incompatible filename: ", file)

    df = pd.DataFrame(db_info)
    print(df)

    df.to_csv('data/database_summary.csv', index=False)
