from abc import ABC, abstractmethod
from config import *
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, log_loss
import tensorflow as tf
from utils.results_saver import plot_accuracy_and_loss, plot_confusion_matrix
from utils.utils import get_optimizer


OPTIMIZER_NAME, OPTIMIZER = get_optimizer(OPTIMIZER, AVAILABLE_OPTIMIZERS, LEARNING_RATE)


class ClassifierBase(ABC):
    def __init__(self, train_data, test_data, val_data, settings, results_dir):
        self.name = self._name()
        self.x_train, self.y_train = train_data[0], train_data[1]
        self.x_val, self.y_val = val_data[0], val_data[1]
        self.x_test, self.y_test = self._prepare_data(test_data)

        self.model = self._create_model()
        self.results = {'Model': ["CNN", 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Loss']}

        self.settings = settings
        self.results_dir = os.path.join(results_dir, self.name)
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)
        
    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def _create_model(self):
        pass

    @staticmethod
    def resize_data(data, size):
        resized_data = []
        for img in data:
            resized_img = cv2.resize(img, (size, size))
            resized_data.append(resized_img)

        return np.array(resized_data)

    def _prepare_data(self, data):
        min_value = np.min(data[0])
        max_value = np.max(data[0])

        x_data = (data[0].astype('float32') - min_value) / (max_value - min_value)

        x_data = tf.expand_dims(x_data, axis=-1)
        x_data = np.repeat(x_data, 3, axis=3)

        if self.name in ["VGG16", "MobileNetV2", "ResNet50"]:
            x_data = self.resize_data(x_data, 224)
        if self.name in ["InceptionV3", "Xception"]:
            x_data = self.resize_data(x_data, 299)

        y_data = data[1]
        return x_data, y_data

    def run_classifier(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        learning_params = "{}_{}_{}".format(LOSS_FUNCTION, OPTIMIZER_NAME, BATCH_SIZE)
        results_dir = os.path.join(self.results_dir, learning_params)
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        accuracy_scores, precision_scores, recall_scores, f1_scores, losses = [], [], [], [], []
        history_per_iteration, predictions_per_iteration, reports_per_iteration, iteration_names = [], [], [], []

        for idx, (x_train, x_val, y_train, y_val) in enumerate(zip(self.x_train, self.x_val, self.y_train, self.y_val)):
            print(f"{idx+1}/{CROSS_VALIDATION_SPLIT} cross-validation iteration:")
            print(f"    Train-test-val: {len(x_train)}-{len(self.x_test)}-{len(x_val)}")

            x_train, y_train = self._prepare_data((x_train, y_train))
            x_val, y_val = self._prepare_data((x_val, y_val))

            model = self._create_model()

            model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=['accuracy'])
            history = model.fit(x_train, y_train, epochs=EPOCHS, shuffle=True, batch_size=BATCH_SIZE,
                                validation_data=(x_val, y_val), callbacks=[callback], verbose=SHOW_LEARNING_PROGRESS)

            y_pred = model.predict(self.x_test, verbose=False)
            y_pred = (y_pred > 0.5).astype(int)
            predictions_per_iteration.append(y_pred)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=1)
            f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=1)
            loss = log_loss(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, zero_division=1)

            print('    Test accuracy:', accuracy)
            print('    Test loss:', loss)
            print(report)

            iteration_names.append(f'Iteracja_{idx + 1}')
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            losses.append(loss)
            reports_per_iteration.append(report)

            history_df = pd.DataFrame(history.history)
            history_df['Iteration'] = idx + 1
            history_per_iteration.append(history_df)

            model.save(os.path.join(results_dir, "models", f"model_cv_{idx+1}.h5"))

        accuracy = np.mean(accuracy_scores)
        precision = np.mean(precision_scores)
        recall = np.mean(recall_scores)
        f1 = np.mean(f1_scores)
        loss = np.mean(losses)

        model_name = "{}_{}_{}_{}_{}".format(self.settings, self.name,  LOSS_FUNCTION, OPTIMIZER_NAME, BATCH_SIZE)
        self.results[model_name] = [self.name, accuracy, precision, recall, f1, loss]

        # Save results
        history_dir = os.path.join(results_dir, "history")
        if not os.path.exists(history_dir):
            os.mkdir(history_dir)

        plot_accuracy_and_loss(history_per_iteration, history_dir, self.settings)
        plot_confusion_matrix(self.y_test, predictions_per_iteration, results_dir, self.settings)

        with pd.ExcelWriter(os.path.join(history_dir, "cross_validation_history.xlsx")) as writer:
            for i, history_df in enumerate(history_per_iteration):
                history_df.to_excel(writer, sheet_name=f'Iteration_{i+1}', index=False)

        with pd.ExcelWriter(os.path.join(results_dir, 'cross_validation_results.xlsx')) as writer:
            results_df = pd.DataFrame({
                'Accuracy': accuracy_scores,
                'Precision': precision_scores,
                'Recall': recall_scores,
                'F1 Score': f1_scores,
                'Loss': losses}, index=[iteration_names])

            summary_row = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Loss': loss}
            summary_row_idx = 'MEAN'
            summary_row = pd.Series(summary_row, name=summary_row_idx)

            results_df = pd.concat([results_df, summary_row], ignore_index=False)
            results_df.to_excel(writer)
