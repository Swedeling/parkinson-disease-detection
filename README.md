# Anomalies detection in speech signals in people with Parkinson’s

The project aims to conduct research in the field of automatic Parkinson's disease diagnostics using speech signal analysis. Parkinson's disease is a neurological disorder that affects speech functions, making voice analysis useful in the diagnosis and monitoring of patients.

The project focuses on analyzing voice signals, specifically vowels /a/, /e/, /i/, /o/, /u/, utilizing the technique of mel spectrograms and various neural network architectures. The results of experiments are automatically recorded and presented in the form of various metrics, such as accuracy, F1 score, specificity, precision, and confusion matrices.

The project code is available in this repository and can be used to perform similar analyses in other contexts or for further research in the field of Parkinson's disease diagnostics. The key project parameters can be customized in the [config.py](./config.py) file, allowing for easy configuration of the analysis to meet specific needs.

## Running the Program
To run this program, follow these steps:

1. Make sure you have all the necessary dependencies installed.

2. Open a terminal or command prompt.

3. Navigate to the project directory.

4. Run the program by executing the following command:

```bash
python main.py
```

## Data preparation
To get started with this project, you need to prepare the data appropriately. Below are the steps you should take to do so:

### Set the Data Path
The first step is to set the correct path to the folder containing the source data. You can do this by editing the [config.py](./config.py) file and changing the value of the RECORDINGS_DIR variable to point to the directory that contains the data. Here's an example:
```python
# PATHS
RECORDINGS_DIR = "data"
```
Make sure that this path is correctly configured to point to the directory where your source data is located.


### Database Structure
The project utilizes a database organized according to the following structure:

```bash
data
|-- polish
|   |-- HC_polish
|   |   |-- recordings
|   |   |   |-- a
|   |   |   |   |-- train
|   |   |   |   |   |-- p_1_K_56_a_001.wav
|   |   |   |   |   |-- p_1_K_56_a_002.wav
|   |   |   |   |-- test
|   |   |   |   |   |-- p_5_K_69_a_001
|   |   |   |-- e
|   |   |   |   |-- ...
|   |   |   |-- i
|   |   |   |   |-- ...
|   |   |   |-- ...
|   
|   |-- PD_polish
|   |   |-- recordings
|   |   |   |-- a
|   |   |   |-- ...
|
|-- spanish
|   |-- HC_spanish
|   |-- PD_spanish
|   ...
```
* `data`: The main data directory.
* `{language}`: A directory containing data in a specific language, e.g., "polish" for Polish.
  * `{health_condition}_{language}`: A directory containing data for a specific health condition (HC - healthy individuals, PD - individuals with Parkinson's disease).
    * `recordings`: A directory containing recordings.
      * `{vowel}`: A directory containing recordings of a specific vowel.
        * `train`: A directory containing training data.
        * `test`: A directory containing test data.


### Supported Languages
The project supports multiple languages, including:

* Polish (`polish`)
* Italian (`italian`)
* Spanish (`spanish`)
* Hungarian (`hungarian`)

It is possible to expand the list of supported languages by adding new entries to the AVAILABLE_LANGUAGES dictionary in the [config.py](./config.py) file.

```python
AVAILABLE_LANGUAGES = {"polish": "pl", "italian": "itl", "spanish": "es", "hungarian": "hu"}
```
### Data Split
The data is divided into training and test sets according to the *subject_wise* approach. This means that recordings from the same individuals cannot be present in both the training and test sets. This is important for conducting cross-validation correctly.

### File Naming Convention
File names in the database follow a specific naming convention:

*{language_id}{gender}{age}{vowel}{recording_number}.wav*

Example: p_1_K_56_a_001.wav

This naming scheme allows for the identification of the speaker and is important for cross-validation. It is not necessary for file names to include all of these elements, but it is crucial that the first part {language_id}{gender}{age} allows for the unambiguous identification of the speaker. This is important for cross-validation and data analysis.

## Setting Up Experiments

### Language and Vowels
To begin experiments, the first step is to configure the languages and vowels you want to analyze. You can choose from the available options and combine them in any way you prefer.

```python
# LANGUAGES AND VOWELS
LANGUAGES = ["polish", "italian", "spanish"]
VOWELS_TO_ANALYZE = ["a", "e", "i", "o", "u"]   # "a", "e", "i", "o", "u"
```

In the above example, three languages have been selected: Polish, Italian, and Spanish. You can customize this list by choosing languages defined in the *AVAILABLE_LANGUAGES* variable or by adding new languages to the list.

The vowels for analysis are selected from the available options in the database, organized according to the structure described earlier. The analysis is conducted separately for each selected vowel.

### Parameters When Loading Recordings
You can customize the length of the loaded recording and the sampling frequency in the [config.py](./config.py).file.

```python
# RECORDING SETTINGS
SR = 44100
SIGNAL_DURATION = 0.1   # time in [s]
```


### Spectrogram Parameters
Parameters related to spectrograms can be customized, which can significantly impact the results of experiments. You have the option to choose between spectrograms and mel-spectrograms and adjust the following parameters:

Selection of the type of spectrogram:

```python
SPECTROGRAMS = False    # Choose True to perform analysis for spectrograms
MELSPECTROGRAMS = True  # Choose True to perform analysis for mel-spectrograms
```
Spectrogram Parameters

For BINSIZE and OVERLAP, you can choose multiple values, and the analysis will be conducted iteratively for all combinations.

```python
BINSIZE = [2048]  # You can adjust the size of the spectrogram window (binsize)
OVERLAP = [512]   # You can adjust the overlap value of the spectrogram
N_MELS = 320      # Number of mel channels
```

Data Augmentation Techniques
```python
AUGMENTATION = ["filtered", "pitch", "slow", "speed", "rolled", "noise"]
# Choose from the available data augmentation techniques: "filtered," "pitch," "slow," "speed," "rolled," "noise"
```

### Training Parameters

The next crucial section pertains to training parameters that you can customize to suit your needs. The [config.py](./config.py) file allows you to control the following parameters:

Batch Size
```python
BATCH_SIZE = 16  # Specify the batch size used during training
```
Classifiers
```python
CLASSIFIERS_TO_TEST = ["VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"]
# Choose classifiers to test from the available options: "VGG16," "ResNet50," "Xception," "InceptionV3," "MobileNetV2"
```

Cross-validation
```python
CROSS_VALIDATION_SPLIT = 10  # Specify the cross-validation split
```

Device (CPU/GPU)
```python
DEVICE = "CPU"  # Choose the device on which you want to perform training: "CPU" or "GPU"
```
Learning epochs
```python
EPOCHS = 5  # Specify the number of training epochs
```

Learning Rate
```python
LEARNING_RATE = 0.0005  # Adjust the learning rate for the optimizer
```

Optimizer
```python
OPTIMIZER = 'sgd'  # Choose an optimizer for training the model from the available options: "adam," "sgd," "rmsprop"
```
Adjusting these parameters allows you to tailor the training process to your specific problem and enhance the effectiveness of your experiments.

## Results Visualization
After conducting experiments, the results are saved in the results_language-analysis folder, for example, results_pol or results_pol+es. You can customize the display of feedback information, such as charts and training progress, using the following settings:
```python
# FEEDBACK SETTINGS
SHOW_PLOTS = True
SHOW_LEARNING_PROGRESS = False
```
### Results Organization
Results are organized separately for each vowel and then for individual spectrogram settings and CNN architectures. The folder structure is as follows:

```bash
RESULTS
|-- results_pl
|   |-- a
|   |   |-- mel-spectrogram_320_2048_512
|   |   |   |-- VGG16
|   |   |   |-- ResNet50
|   |   |   |   |-- binary_crossentropy_sgd_16
|   |   |   |   |   |-- cm
|   |   |   |   |   |-- history
|   |   |   |   |   |   |-- cross_validation_history.xlsx
|   |   |   |   |   |   |-- mel-spectrogram_320_2048_512_accuracy_and_loss_cv1.png
|   |   |   |   |   |   |-- mel-spectrogram_320_2048_512_accuracy_and_loss_cv3.png
|   |   |   |   |   |   |-- ...
|   |   |   |   |   |-- models
|   |   |   |   |   |-- cross_validation_results.xlsx
|   |   |   |-- summary.xlsx
|   |   |-- mel-spectrogram_128_2048_512
|   |   |-- spectrogram_320_2048_512
|   |   |-- spectrograms_summary.xlsx

|   |-- e
|   ...
|
|-- results_pl+es+itl
|   |-- a
|   |-- e
|   ...
```



Thanks to this folder structure, you can easily manage and analyze the results of experiments for different vowels, languages, spectrogram settings, and neural network architectures. All information, such as confusion matrices, learning curves, and cross-validation results, is organized in a clear manner.

Results for individual cross-validation iterations are retained, as well as their collective summaries. This structured approach helps in effectively assessing the performance of your models and conducting in-depth analysis of the experiment outcomes.