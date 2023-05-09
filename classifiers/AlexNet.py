from classifiers.classifier_base import ClassifierBase
import tensorflow as tf


class AlexNet:
    def __init__(self, train_data, test_data):
        self.x_train, self.y_train = train_data
        self.x_test, self.y_test = test_data

        print(self.y_train)

    def _name(self):
        self.name = "AlexNet"

    def run_classifier(self):
        self.x_train = self.x_train.astype('float32') / 255
        self.x_test = self.x_test.astype('float32') / 255

        print("S ", len(self.x_train))

        # self.x_train = tf.expand_dims(self.x_train, axis=0)
        # self.x_test = tf.expand_dims(self.x_test, axis=0)

        print("Z ", len(self.x_train), len(self.x_test))

        num_classes = 2

        self.y_train = tf.keras.utils.to_categorical(self.y_train, num_classes)
        self.y_test = tf.keras.utils.to_categorical(self.y_test, num_classes)

        # Definiowanie modelu AlexNet
        print("Learning...")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(4096, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1000, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        model.fit(self.x_train, self.y_train,
                  batch_size=2,
                  epochs=10,
                  validation_data=(self.x_test, self.y_test),
                  shuffle=True)

        # score = model.evaluate(x_test, y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
