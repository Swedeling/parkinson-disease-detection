from config import *
from utils.results_analysis import plot_accuracy_and_loss, plot_confusion_matrix

from abc import ABC, abstractmethod
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        self.name = self._name()
        self.x_train, self.y_train = self._prepare_data(train_data)
        self.test_orps = test_data["orps"]
        self.test_mrps = test_data["mrps"]

        self.x_test_orps, self.y_test_orps = self._prepare_data(self.test_orps)
        self.x_test_mrps, self.y_test_mrps = self._prepare_data(self.test_mrps)

        self.val_data = self._prepare_data(val_data) if val_data else None
        self.model = self._create_model()
        self.orps_results = {'Model': ["CNN", "loss_function", "optimizer", "batch_size", 'accuracy', 'Precision',
                                       'Recall', 'F1-score']}

        self.mrps_results = {'Model': ["CNN", "loss_function", "optimizer", "batch_size", 'accuracy', 'Precision',
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
        x_data = tf.expand_dims(x_data, axis=3)
        if self.name in ["ResNet50", "InceptionV3"]:
            x_data = np.repeat(x_data, 3, axis=-1)
        y_data = tf.keras.utils.to_categorical(data[1], NUM_CLASSES)
        return x_data, y_data

    def run_classifier(self, loss_fcn, optimizer, batch_size):
        model = tf.keras.models.clone_model(self.model)
        model.build((None, 2))
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        model.compile(loss=loss_fcn, optimizer=optimizer, metrics=['accuracy'])
        history = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=EPOCHS_NUMBER, verbose=0,
                            validation_data=self.val_data, callbacks=[callback])

        model_name = "{}_{}_{}_{}_{}".format(self.settings, self.name, loss_fcn, optimizer, batch_size)
        learning_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size)

        results_dir = os.path.join(self.settings_dir, learning_params)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        orps_results_dir = os.path.join(results_dir, "orps")
        if not os.path.exists(orps_results_dir):
            os.mkdir(orps_results_dir)

        mrps_results_dir = os.path.join(results_dir, "mrps")
        if not os.path.exists(mrps_results_dir):
            os.mkdir(mrps_results_dir)

        # ORPS
        y_pred = np.argmax(model.predict(self.x_test_orps, verbose=0), axis=1)
        y_true = np.argmax(self.y_test_orps, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.orps_results[model_name] = [self.name, loss_fcn, optimizer, batch_size, accuracy, precision, recall, f1]

        plot_accuracy_and_loss(history, orps_results_dir, self.settings)
        plot_confusion_matrix(y_true, y_pred, orps_results_dir, self.settings)

        # MRPS
        y_pred = np.argmax(model.predict(self.x_test_mrps, verbose=0), axis=1)
        y_true = np.argmax(self.y_test_mrps, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.mrps_results[model_name] = [self.name, loss_fcn, optimizer, batch_size, accuracy, precision, recall, f1]

        plot_accuracy_and_loss(history, mrps_results_dir, self.settings)
        plot_confusion_matrix(y_true, y_pred, mrps_results_dir, self.settings)

        return model
