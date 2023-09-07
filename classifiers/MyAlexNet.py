from config import *
from classifiers.classifier_base import ClassifierBase
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class MyAlexNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "AlexNet"

    def _create_model(self):
        model = Sequential([
            Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(227,227,3)),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(384, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(NUM_CLASSES, activation='sigmoid')
        ])
        return model
