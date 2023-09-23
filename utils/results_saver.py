import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR

COLORS = sns.color_palette('flare')


def plot_accuracy_and_loss(cv_history, history_dir, settings):
    for idx, history in enumerate(cv_history):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        accuracy_axis, loss_axis = axes[0], axes[1]

        accuracy_axis.plot(history['accuracy'], color=COLORS[0])
        loss_axis.plot(history['loss'], color=COLORS[0])

        accuracy_axis.plot(history['val_accuracy'], color=COLORS[2])
        loss_axis.plot(history['val_loss'], color=COLORS[2])
        accuracy_axis.legend(['Train', 'Val'], loc='upper left')
        loss_axis.legend(['Train', 'Val'], loc='upper left')

        accuracy_axis.set_title("Dokładność modelu")
        accuracy_axis.set_ylabel('Dokładność')
        accuracy_axis.set_xlabel('Epoka')

        loss_axis.set_title('Strata modelu')
        loss_axis.set_ylabel('Strata')
        loss_axis.set_xlabel('Epoka')

        accuracy_axis.set_ylim(bottom=0)
        loss_axis.set_ylim(bottom=0)

        fig.subplots_adjust(hspace=0.4)

        plt.savefig(os.path.join(history_dir, f'{settings}_accuracy_and_loss_cv{idx+1}.png'))
        plt.close()


def plot_confusion_matrix(y_true, predictions, results_dir, settings):
    cm_dir = os.path.join(results_dir, "cm")
    if not os.path.exists(cm_dir):
        os.mkdir(cm_dir)

    for idx, y_pred in enumerate(predictions):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap=COLORS)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(ticks=[0.5, 1.5], labels=['HC', 'PD'])
        plt.yticks(ticks=[0.5, 1.5], labels=['HC', 'PD'])
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(cm_dir, f'{settings}_cm_cv{idx+1}.png'))
        plt.close()


def summarize_all_results():
    spectrograms_results = {}
    for vowel in os.listdir(RESULTS_DIR):
        for settings in os.listdir((os.path.join(RESULTS_DIR, vowel))):
            if settings not in ["spectrograms_summary.xlsx", "melspectrograms_summary.xlsx"]:
                df = pd.read_excel(os.path.join(RESULTS_DIR, vowel, settings, "summary.xlsx"))
                columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
                df = df.drop(columns=columns_to_drop)
                df.to_excel(os.path.join(RESULTS_DIR, vowel, settings, "summary.xlsx"))

                f1_row = df.drop('Model', axis=1).loc[5].astype(float)
                max_f1_value = f1_row.max()
                model_name = f1_row.idxmax()

                model_settings = '_'.join(model_name.rsplit("_", maxsplit=5)[1:])
                accuracy, precision, recall = df[model_name].loc[2],  df[model_name].loc[3], df[model_name].loc[4]

                spectrograms_results[settings] = [model_settings, accuracy, precision, recall, max_f1_value]
        spectrograms_results_df = pd.DataFrame(spectrograms_results)
        spectrograms_results_df.to_excel(os.path.join(RESULTS_DIR, vowel, "spectrograms_summary.xlsx"))
