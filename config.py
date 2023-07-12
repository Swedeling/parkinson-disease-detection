# EXPERIMENT SETTINGS
BATCH_SIZE = [8] # 8, 16, 32
BINSIZE = [512] # 1024
EPOCHS_NUMBER = 100
LOSS_FUNCTION = ['categorical_crossentropy']
OPTIMIZER = ['adam'] # 'sgd', 'rmsprop', 'adam'
OVERLAP = [0.1] # 0.1, 0.25,

LANGUAGE_TO_LOAD = "polish"
VOWELS_TO_LOAD = ["i"]
# "ResNet50", "VGGNet"
CLASSIFIERS_TO_TEST = ["LeNet-5", "AlexNet", "InceptionV3"] #  "AlexNet", "InceptionV3"
SPECTROGRAMS = True
MELSPECTROGRAMS = True

MANY_RECORDINGS = True

DEVICE = "GPU"

# FEEDBACK SETTINGS
PRINT_DB_INFO = False
SAVE_PLOTS = False
SHOW_PLOTS = True
RUN_DATASET_ANALYSIS = False
COMPUTE_ADDITIONAL_FEATURES = False
RETRAIN_MODELS = False
USE_VALIDATION_DATASET = True

# CONSTANT VARIABLES
AVAILABLE_LANGUAGES = ["polish", "italian"]
CLASS_ENCODING = {"PD": 1, "HS": 0}
CLASSES = ["HS", "PD"]
GENDER_ENCODING = {'K': 0, 'M': 1}
NUM_CLASSES = len(CLASSES)
SR = 44100
SUMMARY_PATH = "data/database_summary.xlsx"
RESULTS_DIR = 'results'
MODELS_DIR = 'models'
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
        languages_to_load = ["italian", "polish"]
    elif LANGUAGE_TO_LOAD not in AVAILABLE_LANGUAGES:
        print("Language not available. I am using default language --> polish")
        languages_to_load = ["polish"]
    else:
        languages_to_load = LANGUAGE_TO_LOAD if type(LANGUAGE_TO_LOAD) == list else [LANGUAGE_TO_LOAD]

    return languages_to_load
