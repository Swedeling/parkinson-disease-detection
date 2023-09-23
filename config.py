import os
from utils.utils import get_languages_to_load

# LANGUAGES AND VOWELS
LANGUAGES = ["polish", "italian", "spanish"]
VOWELS_TO_ANALYZE = ["a", "e", "i", "o", "u"]   # "a", "e", "i", "o", "u"

# LEARNING SETTINGS
BATCH_SIZE = 16
CLASSIFIERS_TO_TEST = ["VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"]    # "VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"
CROSS_VALIDATION_SPLIT = 10
DEVICE = "CPU"
EPOCHS = 50
LEARNING_RATE = 0.0005
LOSS_FUNCTION = 'binary_crossentropy'
OPTIMIZER = 'sgd'

# SPECTROGRAM SETTINGS
AUGMENTATION = ["filtered", "pitch", "slow", "speed", "rolled"]   # "filtered", "pitch", "slow", "speed", "rolled", "noise"
BINSIZE = [2048]
OVERLAP = [512]
N_MELS = 320
SPECTROGRAMS = False    # run analysis for spectrograms
MELSPECTROGRAMS = True  # run analysis for mel-spectrograms

# RECORDING SETTINGS
SR = 44100
SIGNAL_DURATION = 0.1   # time in [s]

# FEEDBACK SETTINGS
SHOW_PLOTS = False
SHOW_LEARNING_PROGRESS = False

# CONSTANT SETTINGS
HC = "HC"
PD = "PD"
CLASSES = {HC: 0, PD: 1}
GENDER_ENCODING = {'K': 0, 'M': 1}
NUM_CLASSES = len(CLASSES)

AVAILABLE_LANGUAGES = {"polish": "pl", "italian": "itl", "spanish": "es", "hungarian": "hu"}
AVAILABLE_OPTIMIZERS = ["adam", "sgd", "rmsprop"]

# PATHS
RECORDINGS_DIR = "data"
LANGUAGES = get_languages_to_load(LANGUAGES, AVAILABLE_LANGUAGES)

language_shorts = '+'.join(AVAILABLE_LANGUAGES[language] for language in LANGUAGES)
RESULTS_DIR = f'results_{language_shorts}'

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
