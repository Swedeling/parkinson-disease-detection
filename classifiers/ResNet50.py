from config import *
from classifiers.classifier_base import ClassifierBase

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50


class ResNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "ResNet50"

    def _create_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(227, 227, 3))
        model = Sequential()
        model.add(base_model)
        model.add(Dense(NUM_CLASSES, activation='softmax'))
        return model
