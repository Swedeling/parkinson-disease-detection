from config import *
from classifiers.classifier_base import ClassifierBase
import tensorflow as tf


class VGGNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "VGGNet"

    def _create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(227, 227, 3)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')])
        return model
