from config import *
from classifiers.classifier_base import ClassifierBase
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
import keras
from keras import layers
import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical

class AE(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "AE"

    def _create_model(self):
        def encoder(input_img):
            # encoder
            # input = 28 x 28 x 1 (wide and thin)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # 28 x 28 x 32
            conv1 = BatchNormalization()(conv1)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
            conv1 = BatchNormalization()(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)  # 14 x 14 x 64
            conv2 = BatchNormalization()(conv2)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
            conv2 = BatchNormalization()(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)  # 7 x 7 x 128 (small and thick)
            conv3 = BatchNormalization()(conv3)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
            conv3 = BatchNormalization()(conv3)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)  # 7 x 7 x 256 (small and thick)
            conv4 = BatchNormalization()(conv4)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
            conv4 = BatchNormalization()(conv4)
            return conv4

        def decoder(conv4):
            # decoder
            conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)  # 7 x 7 x 128
            conv5 = BatchNormalization()(conv5)
            conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
            conv5 = BatchNormalization()(conv5)
            conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)  # 7 x 7 x 64
            conv6 = BatchNormalization()(conv6)
            conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
            conv6 = BatchNormalization()(conv6)
            up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
            conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)  # 14 x 14 x 32
            conv7 = BatchNormalization()(conv7)
            conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
            conv7 = BatchNormalization()(conv7)
            up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)  # 28 x 28 x 1
            return decoded

        input_img = Input(shape=(224, 224, 1))
        autoencoder = Model(input_img, decoder(encoder(input_img)))

        return autoencoder
