import os

# EXPERIMENT SETTINGS
BATCH_SIZES = [16] # 8, 16, 32
BINSIZE = [512] # 1024
EPOCHS_NUMBER = 50
LOSS_FUNCTION = ['binary_crossentropy']
OPTIMIZERS = ['adam'] # 'sgd', 'rmsprop', 'adam'
OVERLAP = [320] # 0.1, 0.25,
LEARNING_RATE = 0.0005

LANGUAGE_TO_LOAD = "spanish"
VOWELS_TO_LOAD = ["a"]

CLASSIFIERS_TO_TEST = []    # "VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"
SPECTROGRAMS = False
MELSPECTROGRAMS = True
MFCC = False

DEVICE = "CPU"

# FEEDBACK SETTINGS
PRINT_DB_INFO = False
SAVE_PLOTS = False
SHOW_PLOTS = True
RUN_DATASET_ANALYSIS = False
COMPUTE_ADDITIONAL_FEATURES = False
RETRAIN_MODELS = False
USE_VALIDATION_DATASET = True

# CONSTANT VARIABLES
AVAILABLE_LANGUAGES = ["polish", "italian", "spanish", "hungarian"]
CLASS_ENCODING = {"PD": 1, "HC": 0}
CLASSES = ["HC", "PD"]
GENDER_ENCODING = {'K': 0, 'M': 1}
NUM_CLASSES = len(CLASSES)
SR = 44100
SUMMARY_PATH = "data/database_summary.xlsx"

RESULTS_DIR = 'results_{}'.format(LANGUAGE_TO_LOAD)
MODELS_DIR = 'models_{}'.format(LANGUAGE_TO_LOAD)

# RESULTS_DIR = 'results'
# MODELS_DIR = 'models'

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
RECORDINGS_DIR = "data"


def get_settings():
    settings = []
    for binsize in BINSIZE:
        for overlap in OVERLAP:
            if SPECTROGRAMS:
                settings.append("spectrogram_{}_{}".format(binsize, overlap))
            if MELSPECTROGRAMS:
                settings.append("melspectrogram_{}_{}".format(binsize, overlap))

    return settings


def get_languages_to_load():
    if LANGUAGE_TO_LOAD == "all":
        languages_to_load = ["italian", "spanish", "polish"]
    elif LANGUAGE_TO_LOAD == "it+pol":
        languages_to_load = ["italian", "polish"]
    elif LANGUAGE_TO_LOAD == "polish+italian+hungarian":
        languages_to_load = ["italian", "polish", "hungarian"]
    elif LANGUAGE_TO_LOAD == "spanish+hungarian":
        languages_to_load = ["spanish", "hungarian"]
    elif LANGUAGE_TO_LOAD == "spanish+italian":
        languages_to_load = ["spanish", "italian"]
    elif LANGUAGE_TO_LOAD not in AVAILABLE_LANGUAGES:
        print("Language not available. I am using default language --> polish")
        languages_to_load = ["polish"]
    else:
        languages_to_load = LANGUAGE_TO_LOAD if type(LANGUAGE_TO_LOAD) == list else [LANGUAGE_TO_LOAD]

    return languages_to_load
