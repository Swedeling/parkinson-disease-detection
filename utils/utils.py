import random


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