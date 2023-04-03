import matplotlib.pyplot as plt
import numpy as np
from config import SR, SHOW_PLOTS


def silence_removing(signal, filename="Test signal"):
    std = np.std(signal[:int(SR*0.5)])
    mean = np.mean(signal[:int(SR*0.5)])

    encoded_signal = np.zeros(signal.shape)

    for i in range(0, encoded_signal.shape[0]):
        if std == 0:
            encoded_signal[i] = 1 if (abs(signal[i] - mean)) > 3 else 0
        else:
            encoded_signal[i] = 1 if (abs(signal[i] - mean))/std > 3 else 0

    frame_size = int(SR / 1000)
    end = len(encoded_signal) - (len(encoded_signal) % frame_size)
    frames = np.split(encoded_signal[:end], frame_size)
    frames.append(encoded_signal[:end * -1])

    labels = []
    for window in frames:
        zeros, ones = 0, 0
        for i in window:
            zeros += 1 if i == 0 else 0
            ones += 1 if i == 1 else 0

        corrected_window = np.zeros_like(window) if zeros > ones else np.ones_like(window)
        labels.append(corrected_window)

    labels = np.concatenate(labels, axis=0)

    result = []
    for label, sample in zip(labels, signal):
        if label == 1:
            result.append(sample)

    if SHOW_PLOTS:
        plt.subplot(2, 1, 1)
        plt.plot([x / SR for x in range(0, len(signal))], signal)
        plt.title("Raw {} recording".format(filename))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot([x / SR for x in range(0, len(result))], result)
        plt.title("{} recording after silence removing".format(filename))
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    return result
