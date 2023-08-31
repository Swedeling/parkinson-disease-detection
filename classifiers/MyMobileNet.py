from config import *
from classifiers.classifier_base import ClassifierBase

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class MyMobileNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "ResNet50"

    def _create_model(self):
        base_model = MobileNet(weights='imagenet', include_top=False)

        # Zamrożenie warstw konwolucyjnych, aby nie były trenowane
        for layer in base_model.layers:
            layer.trainable = True

        # Dodanie warstwy Global Average Pooling i warstw Fully Connected
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # num_classes to liczba klas

        # Tworzenie nowego modelu na podstawie bazowego modelu i dodanych warstw
        model = Model(inputs=base_model.input, outputs=predictions)

        return model
