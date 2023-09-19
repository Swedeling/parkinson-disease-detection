from classifiers.classifier_base import ClassifierBase

from keras.applications import InceptionV3
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model


class MyInceptionV3(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "InceptionV3"

    def _create_model(self):
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(227, 227, 3))

        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
