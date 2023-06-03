from config import *
import tensorflow as tf


class AlexNet:
    def __init__(self, train_data, test_data, val_data):
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data
        self.x_val, self.y_val = val_data
        self.model = self._create_model()

    def _name(self):
        self.name = "AlexNet"

    @staticmethod
    def _create_model():
        model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu',
                                       input_shape=(257, 227, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
            ])
        return model

    def run_classifier(self):
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        self.x_train = tf.expand_dims(self.x_train, axis=3)
        self.x_test = tf.expand_dims(self.x_test, axis=3)
        self.x_val = tf.expand_dims(self.x_val, axis=3)

        self.y_train = tf.keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, NUM_CLASSES)
        self.y_val = tf.keras.utils.to_categorical(self.y_val, NUM_CLASSES)

        print("Learning...")

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train,
                       batch_size=5,
                       epochs=10,
                       validation_data=(self.x_val, self.y_val),
                       shuffle=True)

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
