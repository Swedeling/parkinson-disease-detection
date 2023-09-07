from config import *
from classifiers.classifier_base import ClassifierBase

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model


class ResNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "ResNet50"


    def _create_model(self):
        input_shape = (224, 224, 3)

        # Utworzenie wejściowego tensora
        input_tensor = Input(shape=input_shape)

        # Utworzenie modelu ResNet-50 bez ostatniej warstwy klasyfikacyjnej
        base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        # Zamrożenie wag warstw bazowego modelu
        for layer in base_model.layers:
            layer.trainable = False

        for layer in base_model.layers[-10:]:
            layer.trainable = False

        # Dodanie nowej warstwy klasyfikacyjnej
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        # Utworzenie nowego modelu
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
