import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = "data/database_summary.xlsx"
PRINT_DB_INFO = False
SAVE_PLOTS = True

CLASS_ENCODING = {"PD": 1, "HS": 0}
GENDER_ENCODING = {'K': 0, 'M': 1}


def run_dataset_analysis():
    variants = ["polish+italian", "polish", "italian"]
    for variant in variants:
        print("RUN ANALYSIS FOR {} VARIANT".format(variant.upper()))
        df = _get_data_info(DATA_PATH, variant)
        if PRINT_DB_INFO:
            print_db_info(df)
        _plot_graphs(df, variant)
        _describe_dataset(df, variant)
        print("=================================================")


def _get_data_info(data_path, variant):
    df = pd.read_excel(data_path)
    df = df.drop('filename', axis=1)
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

    plt.style.use('seaborn-whitegrid')
    colors = sns.color_palette('flare')

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Summary of {} dataset".format(variant), fontsize=16)

    axs[0, 0].bar(df['gender'].value_counts().index, df['gender'].value_counts().values,
                  color=colors, edgecolor='black', linewidth=1.2)
    axs[0, 0].set_xlabel('Gender')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_title("Gender distribution")

    df['gender'] = df['gender'].replace(GENDER_ENCODING)
    sns.heatmap(df.corr(numeric_only=True), annot=True, ax=axs[0, 1], cmap='coolwarm')
    axs[0, 1].set_title("Correlation between variables")

    df['age'].plot.hist(bins=10, ax=axs[1, 0], color=colors[1], edgecolor='black', linewidth=1.2)
    axs[1, 0].set_xlabel('Age')
    axs[1, 0].set_ylabel('Count')
    axs[1, 0].set_title("Age distribution")

    df['language'].value_counts().plot.pie(ax=axs[1, 1], colors=colors,
                                           wedgeprops={'linewidth': 1.2, 'edgecolor': 'black'})
    axs[1, 1].set_title("Languages")
    axs[1, 1].legend(loc='upper right')

    fig.tight_layout()

    if SAVE_PLOTS:
        plt.savefig("data_stats/plots/" + "{}_dataset_summary.png".format(variant), dpi=300)

    plt.close()


def _describe_dataset(df, variant):
    with open('data_stats/descriptions/{}_dataset_description.txt'.format(variant), 'w') as f:
        f.write("SUMMARY FOR {} DATASET\n\n".format(variant.upper()))

        f.write("NUMBER OF RECORDINGS: {} \n\n".format(len(df)))
        f.write("GENDER: \n")
        f.write("M: " + str(df['gender'].value_counts()[1]) + "\n")
        f.write("K: " + str(df['gender'].value_counts()[0]) + "\n\n")

        f.write("AGE: \n")
        f.write("Mean: " + str(df['age'].mean()) + "\n")
        f.write("STD: " + str(df['age'].std()) + "\n\n")
