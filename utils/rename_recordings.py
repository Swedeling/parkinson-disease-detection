import json
import os
import shutil


def rename_polish_recordings_by_info_file(data_dir):
    data_dir = os.path.abspath(data_dir)
    encoding_file = os.path.join(data_dir, "encoding_file.json")
    if not os.path.isfile(encoding_file):
        print("Encoding file not found.")
        return

    with open(encoding_file, 'r') as file:
        recordings_info = json.load(file)

    for dirpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename.endswith(".wav") and not filename.startswith("p_"):
                new_filename = filename
                for key, value in recordings_info.items():
                    if key in filename:
                        new_filename = filename.replace(key, str(value))
                old_path = os.path.join(dirpath, filename)
                new_path = os.path.join(dirpath, "p_" + new_filename)
                os.rename(old_path, new_path)


def load_encoding_file(data_dir):
    encoding_file = os.path.join(data_dir, "encoding_file.json")
    if not os.path.isfile(encoding_file):
        print("Encoding file not found.")
        return

    with open(encoding_file, 'r') as file:
        recordings_info = json.load(file)

    return recordings_info


def rename_italian_recordings_by_info_file(data_dir):
    recordings_info = load_encoding_file(data_dir)

    for info in recordings_info:
        if len(info) != 4:
            print("Invalid format in encoding file.")

        if os.path.exists(os.path.join(data_dir, info[0])):
            rename_italian_recordings(data_dir, info[0], info[1], info[2], info[3])


def rename_italian_recordings(data_dir, dirname, age, id, sex):
    data_dir = os.path.abspath(data_dir)
    dirname_path = os.path.join(data_dir, dirname)

    if not os.path.isdir(data_dir):
        print("Data directory does not exist.")
        return

    if not os.path.isdir(dirname_path):
        print("Directory {} does not exist.".format(dirname))
        return

    try:
        age = int(age)
    except ValueError:
        print("Age must be an integer.")
        return

    if sex not in ["M", "K"]:
        print("Sex must be 'M' or 'F'.")
        return

    for file in os.listdir(dirname_path):
        if str(file).startswith('i_'):
            continue

        if file[1] == "A":
            vowel = "a"
        elif file[1] == "E":
            vowel = "e"
        elif file[1] == "I":
            vowel = "i"
        elif file[1] == "O":
            vowel = "o"
        elif file[1] == "U":
            vowel = "u"
        else:
            print("Invalid filename: ", file)
            continue

        if file[2] == "1":
            recording_num = "001"
        elif file[2] == "2":
            recording_num = "002"
        else:
            print("Invalid filename: ", file)
            continue

        new_filename = "i_{}_{}_{}_{}_{}.wav".format(id, sex, age, vowel, recording_num)
        vowel_dir = os.path.join(data_dir, vowel)

        if not os.path.isdir(vowel_dir):
            os.mkdir(vowel_dir)

        if not os.path.exists(os.path.join(vowel_dir, new_filename)):
            os.rename(os.path.join(dirname_path, file), os.path.join(vowel_dir, new_filename))
        else:
            print("File already exists: ", new_filename)

    shutil.rmtree(dirname_path)


def assign_files_to_folders(data_dir):
    if not os.path.isdir(data_dir):
        print("Data directory does not exist.")
        return

    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            try:
                vowel = file.split("_")[4]
                if vowel in ["a", "e", "i", "o", "u"]:
                    src_file = os.path.join(data_dir, file)
                    vowel_dir = os.path.join(data_dir, vowel)
                    if not os.path.isdir(vowel_dir):
                        os.mkdir(vowel_dir)
                    shutil.move(src_file, vowel_dir)
                else:
                    log_file = os.path.join(data_dir, "rejected_files.log")
                    with open(log_file, "a") as f:
                        f.write(f"Incompatible filename: {file}\n")
            except Exception as e:
                log_file = os.path.join(data_dir, "rejected_files.log")
                with open(log_file, "a") as f:
                    f.write(f"Incompatible filename: {file}. {e}\n")
