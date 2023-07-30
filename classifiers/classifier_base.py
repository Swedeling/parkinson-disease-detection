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


class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        self.name = self._name()
        self.x_train, self.y_train = self._prepare_data(train_data)
        self.x_test, self.y_test = self._prepare_data(test_data)
        self.x_val, self.y_val = self._prepare_data(val_data) if val_data else None
        
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
        x_data = tf.expand_dims(x_data, axis=3)
        if self.name in ["ResNet50", "InceptionV3"]:
            x_data = np.repeat(x_data, 3, axis=-1)
        y_data = tf.keras.utils.to_categorical(data[1], NUM_CLASSES)
        return x_data, y_data

    def run_classifier(self, loss_fcn, optimizer, batch_size):
        model = self._create_model()
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        model.compile(loss=loss_fcn, optimizer=optimizer, metrics=['accuracy'])
        
        # train_val = np.concatenate((self.x_train, self.x_val))
        # train_val_labels = np.concatenate((self.y_train, self.y_val))
        # kf = KFold(n_splits=5, shuffle=True)
        # Listy do przechowywania wyników z każdej iteracji cross-walidacji
        # train_loss_per_fold = []
        # train_accuracy_per_fold = []
        # val_loss_per_fold = []
        # val_accuracy_per_fold = []
                
        # for train_index, val_index in kf.split(train_val):
        #     # Podział danych na foldy treningowe i walidacyjne
        #     x_train, x_val = train_val[train_index], train_val[val_index]
        #     y_train, y_val = train_val_labels[train_index], train_val_labels[val_index]



            # Konwersja etykiet na one-hot encoding
            # y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
            # y_val = keras.utils.to_categorical(y_val, NUM_CLASSES)

            # Trening modelu na foldzie treningowym
            # history = model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS_NUMBER, verbose=1, validation_data=(x_val, y_val), callbacks=[callback])
        history = model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=EPOCHS_NUMBER, verbose=1, validation_data=(self.x_val, self.y_val), callbacks=[callback])
            
        #         # Zapisanie wyników z każdej epoki do odpowiednich list
        #     train_loss_per_fold.append(history.history['loss'])
        #     train_accuracy_per_fold.append(history.history['accuracy'])
        #     val_loss_per_fold.append(history.history['val_loss'])
        #     val_accuracy_per_fold.append(history.history['val_accuracy'])


        # # print(train_loss_per_fold)
        # # print(train_accuracy_per_fold)
        # # print(val_loss_per_fold)
        # # print(val_accuracy_per_fold)
        
        print("RESULTS")
        score = model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])


        model_name = "{}_{}_{}_{}_{}".format(self.settings, self.name, loss_fcn, optimizer, batch_size)
        learning_params = "{}_{}_{}".format(loss_fcn, optimizer, batch_size)

        results_dir = os.path.join(self.settings_dir, learning_params)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
     
        y_pred = np.argmax(model.predict(self.x_test, verbose=0), axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.results[model_name] = [self.name, loss_fcn, optimizer, batch_size, accuracy, precision, recall, f1]

        plot_accuracy_and_loss(history, results_dir, self.settings)
        plot_confusion_matrix(y_true, y_pred, results_dir, self.settings)

        return model
