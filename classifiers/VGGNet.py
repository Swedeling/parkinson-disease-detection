from keras.layers import Dropout

from config import *
from classifiers.classifier_base import ClassifierBase
import tensorflow as tf
from keras.applications import VGG16
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Flatten
from tensorflow.keras.models import Model


class VGGNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "VGGNet"

    def _create_model(self):
        # model = VGG16(weights='imagenet', include_top=False)
        # # Dodaj nową warstwę klasyfikacyjną
        # x = model.output
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = Dense(2, activation='softmax')(x)  # lub activation='sigmoid' dla klasyfikacji binarnej
        # model = Model(inputs=model.input, outputs=x)

        vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        for layer in vgg16.layers:
            layer.trainable = False
        # for layer in vgg16.layers[-4:]:
        #     layer.trainable = True
        x = Flatten()(vgg16.output)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.25)(x)

        # x = BatchNormalization()(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=vgg16.input, outputs=predictions)
        return model
