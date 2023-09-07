from config import *
from classifiers.classifier_base import ClassifierBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import regularizers

class LeNet5(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "LeNet-5"

    def _create_model(self):
        l2_strength=0.01
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(227, 227, 3), kernel_regularizer=regularizers.l2(l2_strength)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', kernel_regularizer=regularizers.l2(l2_strength)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(120, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)))
        model.add(Dense(84, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)))
        model.add(Dense(NUM_CLASSES, activation='sigmoid'))
        return model
