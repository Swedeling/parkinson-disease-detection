# EXPERIMENT SETTINGS
BATCH_SIZE = [5]
BINSIZE = [512, 1024]
EPOCHS_NUMBER = [30]
LOSS_FUNCTION = ['categorical_cross-entropy']
OPTIMIZER = ['adam']
OVERLAP = [0.1, 0.25, 0.5]

LANGUAGE_TO_LOAD = "polish"
VOWELS_TO_LOAD = ["a"]

DEVICE = "GPU"

# FEEDBACK SETTINGS
PRINT_DB_INFO = False
SAVE_PLOTS = True
SHOW_PLOTS = False
RUN_DATASET_ANALYSIS = False
COMPUTE_ADDITIONAL_FEATURES = False
USE_VALIDATION_DATASET = False

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
    settings = ["melspectrogram"]
    for binsize in BINSIZE:
        for overlap in OVERLAP:
            settings.append("spectrogram_{}_{}".format(binsize, overlap))
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
