import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import SUMMARY_PATH, PRINT_DB_INFO, SAVE_PLOTS, GENDER_ENCODING


def run_dataset_analysis(variants=("polish+italian", "polish", "italian")):
    for variant in variants:
        print("RUN ANALYSIS FOR {} VARIANT".format(variant.upper()))
        df = _get_data_info(SUMMARY_PATH, variant)
        print(df)
        if PRINT_DB_INFO:
            print_db_info(df)
        _plot_graphs(df, variant)
        _describe_dataset(df, variant)
        print("=================================================")


def _get_data_info(data_path, variant):
    df = pd.read_excel(data_path)
    df = df.drop('filename', axis=1)
    df = df.drop(["a", "e", "i", "o", "u"], axis=1)
    if variant == "polish":
        df = df.loc[df['language'] == 'polish']
    if variant == "italian":
        df = df.loc[df['language'] == 'italian']
    return df


def print_db_info(df):
    print("Database info: ")
    print(df.info(), "\n")
    print("Summary of missing data: ")
    print(df.isna().sum(), "\n")
    print(df.describe(), "\n")


def _plot_graphs(df, variant):
    print("Number of rows removed due to missing values: {}\n".format(len(df) - len(df.dropna())))
    df.dropna(inplace=True)
    df['class'] = df['label'].replace({1: 'PD', 0: 'HS'})
    df['gender'] = df['gender'].replace({"K": "Female", "M": "Male"})

    plt.style.use('seaborn-whitegrid')
    colors = sns.color_palette('flare')

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # fig.suptitle("Summary of dataset", fontsize=16)

    # Gender distribution
    axs[0, 0].bar(df['gender'].value_counts().index, df['gender'].value_counts().values,
                  color=[colors[0], colors[1]], edgecolor='black', linewidth=1.2)
    axs[0, 0].set_xlabel('Gender')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_title("Gender distribution")

    # Class distribution
    axs[1, 0].bar(df['class'].value_counts().index, df['class'].value_counts().values,
                  color=[colors[2], colors[3]], edgecolor='black', linewidth=1.2)
    axs[1, 0].set_xlabel('Class')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title("Class distribution")

    # Class and gender distribution
    counts = df.groupby(['class', 'gender']).size().unstack()
    counts.plot(kind='bar', stacked=False, ax=axs[0, 1], color=[colors[0], colors[1]], edgecolor='black', linewidth=1.2)
    for p in axs[0, 1].patches:
        axs[0, 1].annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                           (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                           textcoords='offset points')

    axs[0, 1].set_xlabel('Class')
    axs[0, 1].set_ylabel('Count')
    axs[0, 1].set_title("Class and age distribution")

    # Age and class distribution
    bins = np.arange(50, 100, 10)
    age_labels = ['50-59', '60-69', '70-79', '80-89', '90+']
    df['category'] = np.digitize(df.age, bins, right=True)
    counts = df.groupby(['category', 'class']).age.count().unstack()
    counts.plot(kind='bar', stacked=False, ax=axs[1, 1], color=[colors[2], colors[3]], edgecolor='black', linewidth=1.2)
    for p in axs[1, 1].patches:
        axs[1, 1].annotate(np.round(p.get_height(), decimals=0).astype(np.int64),
                           (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(2, 10),
                           textcoords='offset points')

    axs[1, 1].set_xlabel('Age group')
    axs[1, 1].set_ylabel('Count')
    axs[1, 1].set_title("Class and age distribution")
    axs[1, 1].set_xticklabels(age_labels)

    fig.tight_layout()

    if SAVE_PLOTS:
        plt.savefig("data_stats/plots/" + "{}_dataset_summary.png".format(variant), dpi=300)

    plt.close()


def _describe_dataset(df, variant):
    with open('data_stats/descriptions/{}_dataset_description.txt'.format(variant), 'w') as f:
        f.write("SUMMARY FOR {} DATASET\n\n".format(variant.upper()))

        f.write("NUMBER OF RECORDINGS: {} \n\n".format(len(df)))

        f.write("CLASSES: \n")
        f.write("PD: " + str(df['label'].value_counts()[1]) + "\n")
        f.write("HS: " + str(df['label'].value_counts()[0]) + "\n\n")

        f.write("GENDER: \n")
        f.write("M: " + str(df['gender'].value_counts()[1]) + "\n")
        f.write("K: " + str(df['gender'].value_counts()[0]) + "\n\n")

        f.write("AGE: \n")
        f.write("Mean: " + str(df['age'].mean()) + "\n")
        f.write("STD: " + str(df['age'].std()) + "\n\n")


def save_recording_len_to_df(data):
    paths, rec_lengths, rec_trimmed_lengths = [], [], []
    for recording in data:
        paths.append(recording.dir_path + recording.filename)
        rec_lengths.append(recording.length)
        rec_trimmed_lengths.append(len(recording.trimmed_recording))

        df = pd.DataFrame({'path': paths, 'length': rec_lengths, 'trimmed length': rec_trimmed_lengths})
        df.to_excel('data/recordings_summary.xlsx', index=False)
