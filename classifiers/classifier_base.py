import keras.applications
from keras.models import Sequential
from config import *
from utils.results_analysis import plot_accuracy_and_loss, plot_confusion_matrix

from abc import ABC, abstractmethod
import numpy as np
import os
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

class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, settings, results_dir, val_data):
        self.name = self._name()
        self.x_train, self.y_train = train_data[0], train_data[1]
        self.x_test, self.y_test = test_data[0], test_data[1]
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
        x_data = data[0].astype('float32') / 255
        # x_data = tf.expand_dims(x_data, axis=3)
        # if self.name in ["ResNet50"]:
        #     x_data = np.repeat(x_data, 3, axis=-1)

        y_data = tf.keras.utils.to_categorical(data[1], NUM_CLASSES)
        return x_data, y_data

    def run_classifier(self, loss_fcn, optimizer, batch_size):
        plt.imshow(self.x_train[0][0])
        plt.axis('off')  # Wyłączenie osi
        plt.show()

        learning_rate = 0.0001
        optimizer = Adam(learning_rate=learning_rate)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5)

        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        losses = []


        for x_train, x_val, y_train, y_val in zip(self.x_train, self.x_val, self.y_train, self.y_val):
            model = self._create_model()
            model.compile(loss=loss_fcn, optimizer=optimizer, metrics=['accuracy'])
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS_NUMBER, verbose=1, validation_data=(x_val, y_val), callbacks=[callback])

            print("RESULTS")
            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

            # Ocena modelu na zbiorze testowym
            y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
            y_true = np.argmax(self.y_test, axis=1)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            # Zachowanie wyników metryk
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        print("=====")
        optimizer = "adam"
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
