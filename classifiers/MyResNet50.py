from classifiers.ClassifierBase import ClassifierBase
from keras.applications import ResNet50
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model


class MyResNet50(ClassifierBase):
    def __init__(self, train_data, test_data, val_data, settings, results_dir):
        super().__init__(train_data, test_data, val_data, settings, results_dir)

    def _name(self):
        return "ResNet50"

    def _create_model(self):
        input_shape = (224, 224, 3)

        input_tensor = Input(shape=input_shape)

        base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
        for layer in base_model.layers:
            layer.trainable = False

        for layer in base_model.layers:
            if "BatchNormalization" in layer.__class__.__name__:
                layer.trainable = True

        x = Flatten()(base_model.output)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model
