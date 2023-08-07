import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import SR, SHOW_PLOTS
import librosa
from sklearn.preprocessing import MinMaxScaler


def silence_removing(signal, filename="Test signal"):
    voice_segments = librosa.effects.split(signal, top_db=30)

    # Utwórz pusty sygnał, do którego dodane zostaną tylko fragmenty głosowe
    voice_audio = []

    # Przeiteruj przez segmenty i dodaj tylko fragmenty zawierające głos do nowego sygnału
    for segment_start, segment_end in voice_segments:
        voice_segment = signal[segment_start:segment_end]
        voice_audio.extend(voice_segment)
    colors = sns.color_palette('flare')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_signal = scaler.fit_transform(np.array(voice_audio).reshape(-1, 1)).flatten()

    if SHOW_PLOTS:
        n=9
        plt.subplot(3, 1, 1)
        plt.plot([x / SR for x in range(0, len(signal))], signal, color=colors[0])
        plt.grid(True)
        plt.title("Surowe nagranie", fontsize=n)
        plt.xlabel("Czas [s]",fontsize=8)
        plt.ylabel("Amplituda",fontsize=n)

        plt.subplot(3, 1, 2)
        plt.plot([x / SR for x in range(0, len(voice_audio))], voice_audio, color=colors[1])
        plt.grid(True)
        plt.title("Nagranie po usunięciu fragmentów ciszy", fontsize=8)
        plt.xlabel("Czas [s]",fontsize=8)
        plt.ylabel("Amplituda", fontsize=n)

        plt.subplot(3, 1, 3)
        plt.plot([x / SR for x in range(0, len(normalized_signal))], normalized_signal, color=colors[2])
        plt.grid(True)
        plt.title("Nagranie po usunięciu fragmentów ciszy i normalizacji", fontsize=8)
        plt.xlabel("Czas [s]", fontsize=8)
        plt.ylabel("Amplituda", fontsize=n)
        plt.subplots_adjust(hspace=0.8)
        plt.show()

    return voice_audio
