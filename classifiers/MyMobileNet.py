from config import *
from classifiers.classifier_base import ClassifierBase

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


class MyMobileNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "MobileNet"

    def _create_model(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers[:-1]:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # Warstwa dropout do uniknięcia overfittingu
        predictions = Dense(1, activation='sigmoid')(x)  # Warstwa wyjściowa do klasyfikacji binarnej

        # Utworzenie modelu
        model = Model(inputs=base_model.input, outputs=predictions)

        return model
