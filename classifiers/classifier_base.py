import keras.applications
from keras.models import Sequential
from config import *
from utils.results_analysis import plot_accuracy_and_loss, plot_confusion_matrix
from sklearn.metrics import classification_report
from abc import ABC, abstractmethod
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense,Flatten
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from keras.optimizers import Adadelta, RMSprop,SGD,Adam

class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, settings, results_dir, val_data):
        self.name = self._name()
        self.x_train, self.y_train = train_data[0], train_data[1]
        self.x_test, self.y_test = self._prepare_data(test_data)
        self.x_val, self.y_val = val_data[0], val_data[1]
        
        self.model = self._create_model()
        self.results = {'Model': ["CNN", "loss_function", "optimizer", "batch_size", 'accuracy', 'Precision',
                                       'Recall', 'F1-score']}

        self.settings = settings
        self.settings_dir = os.path.join(results_dir, self.name)
        if not os.path.exists(self.settings_dir):
            os.mkdir(self.settings_dir)

    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    def _prepare_data(self, data):
        x_data = data[0].astype('float32') / np.min(data[0])
        x_data = tf.expand_dims(x_data, axis=-1)
        x_data = np.repeat(x_data, 3, axis=3)
        y_data = data[1]
        # y_data = tf.keras.utils.to_categorical(data[1], NUM_CLASSES)
        return x_data, y_data

    def run_classifier(self, loss_fcn, optimizer, batch_size):
        plt.imshow(self.x_train[0][0])
        plt.axis('off')
        plt.show()

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        losses = []
        vowel = "a"

        for idx, (x_train, x_val, y_train, y_val) in enumerate(zip(self.x_train, self.x_val, self.y_train, self.y_val)):
            x_train, y_train = self._prepare_data((x_train, y_train))
            x_val, y_val = self._prepare_data((x_val, y_val))
            model = self._create_model()
            model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=['accuracy'])
            history = model.fit(x_train, y_train, epochs=EPOCHS_NUMBER, shuffle=True, batch_size=batch_size,
                           validation_data=(x_val, y_val), callbacks=[callback])

            print("CROSSVAL: ", idx)
            plt.plot(history.history['accuracy'], label='Dokładność treningowa')
            plt.plot(history.history['val_accuracy'], label='Dokładność walidacyjna')
            plt.title('Dokładność')
            plt.xlabel('Epoka')
            plt.ylabel('Dokładność')
            plt.legend()
            plt.savefig(f'accuracy_{self.name}_cv{idx}_{vowel}.png')  # Zapisz wykres jako obrazek
            plt.close()

            # Tworzenie wykresu funkcji straty treningowej i walidacyjnej
            plt.plot(history.history['loss'], label='Strata treningowa')
            plt.plot(history.history['val_loss'], label='Strata walidacyjna')
            plt.title('Funkcja Straty')
            plt.xlabel('Epoka')
            plt.ylabel('Strata')
            plt.legend()
            plt.savefig(f'loss_{self.name}_cv{idx}_{vowel}.png')  # Zapisz wykres jako obrazek
            model.save(f'model_{self.name}_cv{idx}_{vowel}.h5')
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(f'history_{self.name}_cv{idx}_{vowel}.csv', index=False)

            y_pred = model.predict(self.x_test)
            y_pred = (y_pred > 0.5).astype(int)
            y_true = self.y_test

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            print("RESULTS")
            print('Test accuracy:', accuracy)
            report = classification_report(y_true, y_pred)
            print(report)

            # Zachowanie wyników metryk
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        print("=====")
        optimizer = "sgd"
        model_name = "{}_{}_{}_{}_{}".format(self.settings, self.name, loss_fcn, optimizer, batch_size)
        learning_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size)

        results_dir = os.path.join(self.settings_dir, learning_params)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        accuracy = np.mean(accuracy_scores)
        precision = np.mean(precision_scores)
        recall = np.mean(recall_scores)
        f1 = np.mean(f1_scores)
        self.results[model_name] = [self.name, loss_fcn, optimizer, batch_size, accuracy, precision, recall, f1]
        y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        plot_accuracy_and_loss(history, results_dir, self.settings)
        plot_confusion_matrix(y_true, y_pred, results_dir, self.settings)

        return model
