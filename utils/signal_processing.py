import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import SR, SHOW_PLOTS
import librosa
import random
from scipy.signal import butter, lfilter
from sklearn.preprocessing import MinMaxScaler


def silence_removing(signal, filename="Test signal"):
    voice_segments = librosa.effects.split(signal, top_db=30)
    voice_audio = []

    for segment_start, segment_end in voice_segments:
        voice_segment = signal[segment_start:segment_end]
        voice_audio.extend(voice_segment)
    colors = sns.color_palette('flare')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_signal = scaler.fit_transform(np.array(voice_audio).reshape(-1, 1)).flatten()

    if SHOW_PLOTS:
        n = 9
        plt.subplot(3, 1, 1)
        plt.plot([x / SR for x in range(0, len(signal))], signal, color=colors[0])
        plt.grid(True)
        plt.title(f"Surowe nagranie: {filename}", fontsize=n)
        plt.xlabel("Czas [s]", fontsize=8)
        plt.ylabel("Amplituda", fontsize=n)

        plt.subplot(3, 1, 2)
        plt.plot([x / SR for x in range(0, len(voice_audio))], voice_audio, color=colors[1])
        plt.grid(True)
        plt.title("Nagranie po usunięciu fragmentów ciszy", fontsize=8)
        plt.xlabel("Czas [s]", fontsize=8)
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


def random_roll(signal):
    shift_amount = np.random.randint(0, len(signal)) if len(signal) > 0 else 0
    rolled_signal = np.roll(signal, shift_amount)
    return rolled_signal


def pitch_change(y):
    time_stretch_factor = random.uniform(3, 5) * random.choice([-1, 1])
    stretched_audio = librosa.effects.pitch_shift(y, sr=SR, n_steps=time_stretch_factor)
    resampled_audio = librosa.resample(stretched_audio, orig_sr=SR, target_sr=SR)
    return resampled_audio


def speed_up(y):
    speed_factor = random.uniform(1.2, 1.5)
    y_fast = librosa.effects.time_stretch(y, rate=speed_factor)
    y_restored = librosa.resample(y_fast, orig_sr=int(SR / speed_factor), target_sr=SR)
    return y_restored


def slow_down(y):
    speed_factor = random.uniform(0.2, 0.8)
    y_slow = librosa.effects.time_stretch(y, rate=speed_factor)
    y_restored = librosa.resample(y_slow, orig_sr=int(SR / speed_factor), target_sr=SR)
    return y_restored


def color_noise(y):
    power_law_exponent = random.uniform(4, 6)
    noise = np.random.normal(0, 0.5, len(y))
    power_law_noise = noise ** power_law_exponent
    power_law_noise[np.isnan(power_law_noise)] = 0
    noisy_audio = y + power_law_noise
    return noisy_audio


def bandpass_filter(signal):
    low_freq = 500  # Hz
    high_freq = 1500  # Hz
    nyq = 0.5 * SR
    low = low_freq / nyq
    high = high_freq / nyq
    b, a = butter(5, [low, high], btype='bandpass')
    y = lfilter(b, a, signal)
    return y


def apply_audio_augmentation(y, augmentation):
    if augmentation == "pitch":
        y = pitch_change(y)
    elif augmentation == "speed":
        y = speed_up(y)
    elif augmentation == "slow":
        y = slow_down(y)
    elif augmentation == "noise":
        y = color_noise(y)
    elif augmentation == "rolled":
        y = random_roll(y)
    elif augmentation == "filtered":
        pass
    else:
        print(f"This augmentation technique ({augmentation}) is not supported. "
              f"Please choose from: 'filtered', 'pitch', 'slow', 'speed', 'rolled', 'noise'.")
        return None

    return bandpass_filter(y)
