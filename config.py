import os
from utils.utils import get_languages_to_load

LANGUAGES = ["polish", "italian"]
VOWELS_TO_ANALYZE = ["a", "e"]   # "a", "e", "i", "o", "u"

# LEARNING SETTINGS
BATCH_SIZE = 16
CLASSIFIERS_TO_TEST = ["VGG16", "ResNet50"]    # "VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"
CROSS_VALIDATION_SPLIT = 2
DEVICE = "CPU"
EPOCHS = 5
LEARNING_RATE = 0.0005
LOSS_FUNCTION = 'binary_crossentropy'
OPTIMIZER = 'sgd'

# SPECTROGRAM SETTINGS
AUGMENTATION = ["filtered", "pitch", "slow", "speed", "rolled", "noise"]   # "filtered", "pitch", "slow", "speed", "rolled", "noise"
BINSIZE = [2048]
OVERLAP = [512]
N_MELS = 320
SPECTROGRAMS = False    # run analysis for spectrograms
MELSPECTROGRAMS = True  # run analysis for mel-spectrograms

# FEEDBACK SETTINGS
PRINT_DB_INFO = False
SAVE_PLOTS = False
SHOW_PLOTS = True
SHOW_LEARNING_PROGRESS = False

# CONSTANT SETTINGS
SR = 44100
SIGNAL_DURATION = 0.1   # time in [s]

AVAILABLE_LANGUAGES = {"polish": "pl", "italian": "itl", "spanish": "es", "hungarian": "hu"}
AVAILABLE_OPTIMIZERS = ["adam", "sgd", "rmsprop"]

HC = "HC"
PD = "PD"
CLASSES = {HC: 0, PD: 1}
GENDER_ENCODING = {'K': 0, 'M': 1}
NUM_CLASSES = len(CLASSES)

# PATHS
SUMMARY_PATH = "data/database_summary.xlsx"
RECORDINGS_DIR = "data"
LANGUAGES = get_languages_to_load(LANGUAGES, AVAILABLE_LANGUAGES)

language_shorts = '+'.join(AVAILABLE_LANGUAGES[language] for language in LANGUAGES)
RESULTS_DIR = f'results_{language_shorts}'

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


