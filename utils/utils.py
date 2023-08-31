import os
import random
import tensorflow as tf


def mix_list_and_df(df, label):
    df["class"] = label
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    labels = list(df_shuffled["class"])
    df_shuffled = df_shuffled.drop('class', axis=1)

    return df_shuffled, labels


def mix_lists(list1, list2):
    zipped = list(zip(list1, list2))
    random.shuffle(zipped)
    list1, list2 = zip(*zipped)
    return list(list1), list(list2)


def prepare_datasets(data, vowel):
    hc_data = data.load_recordings("HC", vowel)
    pd_data = data.load_recordings("PD", vowel)
    x_data = pd_data + hc_data
    y_data = [1] * len(pd_data) + [0] * len(hc_data)
    x_data, y_data = mix_lists(x_data, y_data)
    return x_data, y_data


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
