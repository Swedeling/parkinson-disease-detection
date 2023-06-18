import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_accuracy_and_loss(history, settings_dir, settings, val_data=False):
    if not os.path.exists(settings_dir):
        os.mkdir(settings_dir)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    accuracy_axis = axes[0]
    loss_axis = axes[1]

    accuracy_axis.plot(history.history['accuracy'])
    loss_axis.plot(history.history['loss'])

    if val_data:
        accuracy_axis.plot(history.history['val_accuracy'])
        loss_axis.plot(history.history['val_loss'])
        accuracy_axis.legend(['Train', 'Validation'], loc='upper left')
        loss_axis.legend(['Train', 'Validation'], loc='upper left')
    else:
        accuracy_axis.legend(['Train'], loc='upper left')
        loss_axis.legend(['Train'], loc='upper left')

    accuracy_axis.set_title('model accuracy')
    accuracy_axis.set_ylabel('accuracy')
    accuracy_axis.set_xlabel('epoch')

    loss_axis.set_title('model loss')
    loss_axis.set_ylabel('loss')
    loss_axis.set_xlabel('epoch')

    fig.subplots_adjust(hspace=0.4)

    plt.savefig(os.path.join(settings_dir, f'{settings}_accuracy_and_loss.png'))
    plt.close()


def plot_confusion_matrix(labels, predictions, settings_dir, settings):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(settings_dir, f'{settings}_cm.png'))
    plt.close()
