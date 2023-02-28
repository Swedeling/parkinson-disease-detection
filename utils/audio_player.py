import matplotlib as mpl
import matplotlib.pyplot as plt
import sounddevice as sd

from data_loader import SR

mpl.use('TkAgg')


def plot_voice_recording(signal, label):
    y = [x/SR for x in range(0, len(signal))]
    plt.plot(y, signal)

    plt.title("Raw {} recording".format(label))
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    plt.show()


def play_voice_recording(signal, label):
    sd.play(signal, SR)
    status = sd.wait()

