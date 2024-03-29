from classifiers.ClassifierBase import ClassifierBase
from keras.applications import VGG16
from keras.layers import Dropout, Dense, Flatten
from keras.models import Model


class MyVGG16(ClassifierBase):
    def __init__(self, train_data, test_data, val_data, settings, results_dir):
        super().__init__(train_data, test_data, val_data, settings, results_dir)

    def _name(self):
        return "VGG16"

    def _create_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
