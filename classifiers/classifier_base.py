from config import *
from results_analysis import plot_accuracy_and_loss, plot_confusion_matrix

from abc import ABC, abstractmethod
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        self.name = self._name()
        self.x_train, self.y_train = self._prepare_data(train_data)
        self.x_test, self.y_test = self._prepare_data(test_data)
        self.val_data = self._prepare_data(val_data) if val_data else None
        self.model = self._create_model()
        self.results = {'Model': ["CNN", "loss_function", "optimizer", "batch_size", 'Epochs', 'Accuracy',
                                  'Precision', 'Recall', 'F1-score']}
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

    @staticmethod
    def _prepare_data(data):
        x_data = data[0].astype('float32') / 255
        x_data = tf.expand_dims(x_data, axis=3)
        y_data = tf.keras.utils.to_categorical(data[1], NUM_CLASSES)
        return x_data, y_data

    def run_classifier(self, loss_fcn, optimizer, batch_size, epochs):
        model = self.model
        model.compile(loss=loss_fcn, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, verbose=0,
                            validation_data=self.val_data)

        y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        model_name = "{}_{}_{}_{}_{}".format(self.settings, self.name, loss_fcn, optimizer, batch_size, epochs)
        learning_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size, epochs)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        self.results[model_name] = [self.name, loss_fcn, optimizer, batch_size, epochs, accuracy, precision, recall, f1]

        results_dir = os.path.join(self.settings_dir, learning_params)

        plot_accuracy_and_loss(history, results_dir, self.settings, val_data=False)
        plot_confusion_matrix(y_true, y_pred, results_dir, self.settings)

        return model
