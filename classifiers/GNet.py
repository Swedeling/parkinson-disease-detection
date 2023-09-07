from config import *
from classifiers.classifier_base import ClassifierBase
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from keras.applications import VGG16
import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from tensorflow.keras.models import Model


import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNet(ClassifierBase):
    def __init__(self, train_data, test_data, settings, results_dir, val_data=None):
        super().__init__(train_data, test_data, settings, results_dir, val_data)

    def _name(self):
        return "GNet"

    def _create_model(self):
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Zbudowanie własnej głowy klasyfikacyjnej
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # Warstwa dropout do uniknięcia overfittingu
        predictions = Dense(1, activation='sigmoid')(x)  # Warstwa wyjściowa do klasyfikacji binarnej

        # Utworzenie modelu
        model = Model(inputs=base_model.input, outputs=predictions)

        return model



# Define a custom autoencoder model based on the described architecture
class ModifiedResNetAutoencoder(nn.Module):
    def __init__(self):
        super(ModifiedResNetAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),  # First Conv Layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Max Pooling Layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # Second Conv Layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # Max Pooling Layer
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),  # Third Conv Layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Max Pooling Layer
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.Sigmoid()  # Use sigmoid activation for the final layer for pixel values between 0 and 1
        )

        # Define a three-layer dense network with PReLU and dropout
        self.dense_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.PReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        # Encoding
        x = self.encoder(x)

        # Flatten the output for the dense network
        x = x.view(x.size(0), -1)

        # Apply the dense network
        x = self.dense_network(x)

        # Decoding
        x = x.view(x.size(0), 256, 1, 1)  # Reshape for the decoder
        x = self.decoder(x)
        return x
