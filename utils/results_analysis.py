import matplotlib.pyplot as plt
import os
import numpy as np

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR, USE_VALIDATION_DATASET

COLORS = sns.color_palette('flare')


def plot_accuracy_and_loss(history, settings_dir, settings):
    if not os.path.exists(settings_dir):
        os.mkdir(settings_dir)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    accuracy_axis = axes[0]
    loss_axis = axes[1]

    accuracy_axis.plot(history.history['accuracy'], color=COLORS[0])
    loss_axis.plot(history.history['loss'], color=COLORS[0])

    if USE_VALIDATION_DATASET:
        accuracy_axis.plot(history.history['val_accuracy'], color=COLORS[2])
        loss_axis.plot(history.history['val_loss'], color=COLORS[2])
        accuracy_axis.legend(['Train', 'Val'], loc='upper left')
        loss_axis.legend(['Train', 'Val'], loc='upper left')
    else:
        accuracy_axis.legend(['Train'], loc='upper left')
        loss_axis.legend(['Train'], loc='upper left')

    accuracy_axis.set_title("Dokładność modelu")
    accuracy_axis.set_ylabel('dokładność')
    accuracy_axis.set_xlabel('epoka')

    loss_axis.set_title('Strata modelu')
    loss_axis.set_ylabel('strata')
    loss_axis.set_xlabel('epoka')

    fig.subplots_adjust(hspace=0.4)

    plt.savefig(os.path.join(settings_dir, f'{settings}_accuracy_and_loss.png'))
    plt.close()


def plot_confusion_matrix(labels, predictions, settings_dir, settings):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=COLORS)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=[0.5, 1.5], labels=['HC', 'PD'])
    plt.yticks(ticks=[0.5, 1.5], labels=['HC', 'PD'])
    # plt.title('Confusion Matrix')
    plt.savefig(os.path.join(settings_dir, f'{settings}_cm.png'))
    plt.close()


def summarize_all_results(mode):
    spectrograms_results = {}
    for vowel in os.listdir(RESULTS_DIR):
        print(vowel)
        for settings in os.listdir((os.path.join(RESULTS_DIR, vowel))):
            if settings not in ["spectrograms_summary_orps.xlsx", "spectrograms_summary_mrps.xlsx", "melspectrograms_summary_orps.xlsx", "melspectrograms_summary_mrps.xlsx"]:
                print(settings)
                df = pd.read_excel(os.path.join(RESULTS_DIR, vowel, settings, "summary_{}.xlsx".format(mode)))
                columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
                df = df.drop(columns=columns_to_drop)
                df.to_excel(os.path.join(RESULTS_DIR, vowel, settings, "summary_{}.xlsx".format(mode)))

                f1_row = df.drop('Model', axis=1).loc[7].astype(float)
                max_f1_value = f1_row.max()
                model_name = f1_row.idxmax()

                model_settings = '_'.join(model_name.rsplit("_", maxsplit=5)[1:])
                accuracy, precision, recall = df[model_name].loc[4], \
                                        df[model_name].loc[5], \
                                        df[model_name].loc[6]

                spectrograms_results[settings] = [model_settings, accuracy, precision, recall, max_f1_value]
        spectrograms_results_df = pd.DataFrame(spectrograms_results)
        spectrograms_results_df.to_excel(os.path.join(RESULTS_DIR, vowel, "spectrograms_summary_{}.xlsx".format(mode)))


def prepare_plots():
    settings = ["1024_0.1", "1024_0.25", "1024_0.5", "512_0.1", "512_0.25", "512_0.5"]
    spectrograms_one_rec_acc = {"settings": ["1024_0.1", "1024_0.25", "1024_0.5", "512_0.1", "512_0.25", "512_0.5"]}
    spectrograms_many_rec_acc = {"settings": ["1024_0.1", "1024_0.25", "1024_0.5", "512_0.1", "512_0.25", "512_0.5"]}
    melspectrograms_one_rec_acc = {"settings": ["1024_0.1", "1024_0.25", "1024_0.5", "512_0.1", "512_0.25", "512_0.5"]}
    melspectrograms_many_rec_acc = {"settings": ["1024_0.1", "1024_0.25", "1024_0.5", "512_0.1", "512_0.25", "512_0.5"]}

    for vowel in ["a", "e", "i", "o", "u"]:
        df = pd.read_excel("RESULTS.xlsx", sheet_name=vowel)
        split_on_spectrograms = df.index[df["SPEKTROGRAMY"] == "MELSPEKTROGRAMY"][0]

        spectrograms = df.iloc[:split_on_spectrograms]  # Pierwsza część (od początku do wiersza row_index)
        melspectrograms = df.iloc[split_on_spectrograms:]  # Druga część (od wiersza row_index do końca)
        melspectrograms = melspectrograms.reset_index(drop=True)
        melspectrograms = melspectrograms.drop(0)

        split_on_spec = spectrograms.index[spectrograms["SPEKTROGRAMY"] == "Wiele nagrań od jednego mówcy"][0]
        split_on_melspec = melspectrograms.index[melspectrograms["SPEKTROGRAMY"] == "Wiele nagrań od jednego mówcy"][0]

        spectrograms_one_rec = spectrograms.iloc[:split_on_spec]
        spectrograms_one_rec = spectrograms_one_rec.drop(2)
        first_two_rows = spectrograms_one_rec.iloc[:2]

        headers = [f"{first_two_rows.iloc[0, i]}_{first_two_rows.iloc[1, i]}" for i in range(len(first_two_rows.columns))]
        headers = [header.replace('.0_', '_') for header in headers]
        spectrograms_one_rec.columns = [headers]
        spectrograms_one_rec = spectrograms_one_rec.drop(0)
        spectrograms_one_rec = spectrograms_one_rec.drop(1)
        spectrograms_one_rec = spectrograms_one_rec.reset_index(drop=True)

        spectrograms_many_rec = spectrograms.iloc[split_on_spec:]
        spectrograms_many_rec.columns = [headers]
        spectrograms_many_rec = spectrograms_many_rec.reset_index(drop=True)
        spectrograms_many_rec = spectrograms_many_rec.drop(0)
        spectrograms_many_rec = spectrograms_many_rec.dropna()
        spectrograms_many_rec = spectrograms_many_rec.reset_index(drop=True)


        melspectrograms_one_rec = melspectrograms.iloc[:split_on_melspec]
        melspectrograms_one_rec = melspectrograms_one_rec.drop(2)
        melspectrograms_one_rec.columns = [headers]
        melspectrograms_one_rec = melspectrograms_one_rec.reset_index(drop=True)
        melspectrograms_one_rec = melspectrograms_one_rec.drop(0)
        melspectrograms_one_rec = melspectrograms_one_rec.drop(1)
        melspectrograms_one_rec = melspectrograms_one_rec.dropna()
        melspectrograms_one_rec = melspectrograms_one_rec.reset_index(drop=True)

        melspectrograms_many_rec = melspectrograms.iloc[split_on_melspec:]
        melspectrograms_many_rec.columns = [headers]
        melspectrograms_many_rec = melspectrograms_many_rec.reset_index(drop=True)
        melspectrograms_many_rec = melspectrograms_many_rec.dropna()

        spectrograms_one_rec_acc[vowel] = spectrograms_one_rec.iloc[0].values[1:]
        spectrograms_many_rec_acc[vowel] = spectrograms_many_rec.iloc[0].values[1:]
        melspectrograms_one_rec_acc[vowel] = melspectrograms_one_rec.iloc[0].values[1:]
        melspectrograms_one_rec_acc[vowel] = melspectrograms_one_rec.iloc[0].values[1:]
        melspectrograms_many_rec_acc[vowel] = melspectrograms_many_rec.iloc[0].values[1:]


    spectrograms_one_rec_acc = pd.DataFrame(spectrograms_one_rec_acc)
    spectrograms_many_rec_acc = pd.DataFrame(spectrograms_many_rec_acc)
    melspectrograms_one_rec_acc = pd.DataFrame(melspectrograms_one_rec_acc)
    melspectrograms_many_rec_acc = pd.DataFrame(melspectrograms_many_rec_acc)

    print(spectrograms_one_rec_acc)
    print(spectrograms_many_rec_acc)
    print(melspectrograms_one_rec_acc)
    print(melspectrograms_many_rec_acc)

    colors = sns.color_palette('flare')

    title = 'Spektrogramy (jedno nagranie dla jednego mówcy)'
    plt.figure(figsize=(10, 12))
    ax = spectrograms_one_rec_acc.plot(kind='bar', color=colors, edgecolor='black')
    ax.set_xticklabels(settings)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), ncol=1)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('Rows', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(title, fontsize=12)
    xlabels = [label.get_text().replace("_", '\n') for label in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels, rotation=0)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

    plt.savefig('plots/{}.png'.format(title))
    plt.show()


    # plot settings vs accuracy for each vowel

    vowel = "e"
    mode = "many"

    if mode == "many":
        settings = spectrograms_many_rec_acc["settings"]
        vowel_spectrograms_acc = spectrograms_many_rec_acc[vowel]
        vowel_melspectrograms_acc = melspectrograms_many_rec_acc[vowel]
    else:
        settings = spectrograms_one_rec_acc["settings"]
        vowel_spectrograms_acc = spectrograms_one_rec_acc[vowel]
        vowel_melspectrograms_acc = melspectrograms_one_rec_acc[vowel]

    index = np.arange(len(settings))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    vowel_spectrograms = ax.bar(index, vowel_spectrograms_acc, width, edgecolor='black', label='Spektrogramy', color=colors[1])
    vowel_melspectrograms = ax.bar(index + width, vowel_melspectrograms_acc, width, edgecolor='black', label='Melspektrogramy', color=colors[2])
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(0.85, 1), ncol=1)
    plt.style.use('seaborn-whitegrid')
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax.set_ylim(0, 1)
    ax.set_xticks(index + width / 2)
    ax.set_xticklabels(settings)
    ax.set_xlabel('Ustawienia (binsize_overlap)')
    ax.set_ylabel('Dokładność')

    # Wyświetlanie wykresu
    plt.show()


