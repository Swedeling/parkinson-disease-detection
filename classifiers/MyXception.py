from classifiers.ClassifierBase import ClassifierBase
from keras.applications import Xception
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model


class MyXception(ClassifierBase):
    def __init__(self, train_data, test_data, val_data, settings, results_dir):
        super().__init__(train_data, test_data, val_data, settings, results_dir)

    def _name(self):
        return "Xception"

    def _create_model(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
