import os
from pydub import AudioSegment
import random
import tensorflow as tf


def mix_lists(list1, list2):
    zipped = list(zip(list1, list2))
    random.shuffle(zipped)
    list1, list2 = zip(*zipped)
    return list1, list2


def convert_file_extension_into_wav(dir_path, filename, overwrite=False):
    formats_to_convert = ['.m4a']

    if filename.endswith(tuple(formats_to_convert)):
        (path, file_extension) = os.path.splitext(filename)
        file_extension = file_extension.replace('.', '')
        wav_filename = filename.replace(file_extension, 'wav')
        wav_file_path = os.path.join(dir_path, wav_filename)

        if not os.path.exists(wav_file_path):
            try:
                track = AudioSegment.from_file(os.path.join(dir_path, filename), format=file_extension)
                print('CONVERTING: ' + str(wav_file_path))
                file_handle = track.export(wav_file_path, format='wav')

                if overwrite:
                    os.remove(os.path.join(dir_path, filename))

            except:
                print("Problem with converting file: {}".format(filename))


def set_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUS: ", gpus)
    if gpus:
        try:
            tf.config.set_visible_devices(gpus, 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def get_languages_to_load(languages, available_languages):
    languages_to_load = []
    for language in languages:
        if language in available_languages:
            languages_to_load.append(language)
    return languages_to_load


def get_settings(n_mels, binsizes, overlaps, analyze_spectrograms, analyze_melspectrograms):
    settings = []
    for binsize in binsizes:
        for overlap in overlaps:
            if analyze_spectrograms:
                settings.append("spectrogram_{}_{}".format(binsize, overlap))
            if analyze_melspectrograms:
                settings.append("mel-spectrogram_{}_{}_{}".format(n_mels, binsize, overlap))

    return settings


def get_optimizer(optimizer_name, available_optimizers, learning_rate):
    if optimizer_name not in available_optimizers:
        print(f"Warning: The specified optimizer ({optimizer_name}) is not supported. "
              f"Please choose from: {available_optimizers}. Using the default optimizer (sgd).")
        optimizer_name = "sgd"

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        print("Using the default optimizer (sgd).")
        optimizer_name = "sgd"
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    return optimizer_name, optimizer
