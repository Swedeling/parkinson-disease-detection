# Detekcja anomalii w sygnale mowy u osób z chorobą Parkinsona


Projekt ma na celu przeprowadzenie badań w dziedzinie automatycznej diagnostyki choroby Parkinsona, wykorzystując analizę sygnału mowy. Choroba Parkinsona jest neurologicznym schorzeniem, które wpływa na funkcje mowy, dlatego analiza głosu może być przydatna w procesie diagnozy i monitorowania pacjentów.

Projekt skupia się na analizie sygnału głosu, konkretnie na samogłoskach /a/, /e/, /i/, /o/, /u/, wykorzystując technikę melspektrogramów i różne architektury sieci neuronowych. Wyniki eksperymentów są automatycznie zapisywane i przedstawiane w postaci różnych metryk, takich jak dokładność (accuracy), F1, specyficzność, precyzja oraz macierze pomyłek.

Kod projektu jest dostępny w tym repozytorium i może być używany do przeprowadzenia podobnych analiz w innych kontekstach lub do dalszego rozwoju badań w dziedzinie diagnostyki choroby Parkinsona. Najważniejsze parametry projektu można dostosować w pliku [config.py](./config.py), co pozwala na łatwą konfigurację analizy pod własne potrzeby.

## Uruchamianie programu
Aby uruchomić ten program, wykonaj następujące kroki:

1. Upewnij się, że masz zainstalowane wszystkie niezbędne zależności.

2. Otwórz terminal lub wiersz polecenia.

3. Przejdź do katalogu projektu.

4. Uruchom program, wykonując polecenie:

```bash
python main.py
```


## Przygotowanie danych
Aby rozpocząć pracę z tym projektem, należy odpowiednio przygotować dane. Poniżej znajdziesz kroki, które należy podjąć, aby to zrobić:

### Ustawienie ścieżki do danych
Pierwszym krokiem jest ustawienie odpowiedniej ścieżki do folderu, w którym znajdują się dane źródłowe. Możesz to zrobić, 
edytując plik [config.py](./config.py) i zmieniając wartość zmiennej `RECORDINGS_DIR`, tak aby wskazywała na katalog zawierający dane. Oto przykład:

```python
# PATHS
RECORDINGS_DIR = "data"
```
Upewnij się, że ścieżka ta jest poprawnie skonfigurowana, aby wskazywać na katalog, w którym znajdują się Twoje dane źródłowe.

### Struktura bazy danych
Projekt wykorzystuje bazę danych, która jest zorganizowana według następującej struktury:

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

Struktura ta jest opisana w następujący sposób:

* `data`: Główny katalog danych.
* `{język}`: Katalog zawierający dane w danym języku, np. "polish" dla polskiego.
  * `{stan zdrowia}_{język}`: Katalog zawierający dane dla określonego stanu zdrowia (HC - osoby zdrowe, PD - osoby chore na chorobę Parkinsona).
    * `recordings`: Katalog zawierający nagrania.
        * `{samogłoska}`: Katalog zawierający nagrania danej samogłoski.
            * `train`: Katalog zawierający dane treningowe.
                * `test`: Katalog zawierający dane testowe.
            * data: The main data directory.

### Obsługiwane Języki
Projekt obsługuje wiele języków, w tym:

* Polski (`polish`)
* Włoski (`italian`)
* Hiszpański (`spanish`)
* Węgierski (`hungarian`)

Możliwe jest rozszerzenie obsługiwanych języków, dodając nowe pozycje do słownika `AVAILABLE_LANGUAGES` w pliku [config.py](./config.py).

```python
AVAILABLE_LANGUAGES = {"polish": "pl", "italian": "itl", "spanish": "es", "hungarian": "hu"}
```

### Podział Danych
Dane są podzielone na zbiór treningowy i testowy zgodnie z podejściem `subject_wise`. Oznacza to, że nagrania od tych samych osób nie mogą znaleźć się zarówno w zbiorze treningowym, jak i testowym. Jest to istotne dla poprawnego przeprowadzenia krosswalidacji.

### Nazewnictwo Plików
Nazwy plików w bazie danych mają określony schemat nazewnictwa:

*{język_id}_{płeć}_{wiek}_{samogłoska}_{numer-nagrania}.wav*

Przykład: p_1_K_56_a_001.wav

Schemat ten pozwala na identyfikację osoby mówiącej oraz jest ważny przy krosswalidacji.
Nie jest konieczne, aby w nazwach plików znajdowały się wszystkie te elementy, ale ważne jest, aby pierwszy człon {język_id}_{płeć}_{wiek} pozwalał na jednoznaczną identyfikację osoby mówiącej. Jest to istotne przy przeprowadzaniu krosswalidacji i analizie danych.


## Ustawienie eksperymentów

### Język i samogłoski
Aby rozpocząć eksperymenty, pierwszym krokiem jest konfiguracja języków oraz samogłoszek, które chcesz analizować. Możesz wybierać spośród dostępnych opcji i łączyć je w dowolny sposób.
```python
# LANGUAGES AND VOWELS
LANGUAGES = ["polish", "italian", "spanish"]
VOWELS_TO_ANALYZE = ["a", "e", "i", "o", "u"]   # "a", "e", "i", "o", "u"
```

W powyższym przykładzie wybrane zostały trzy języki: polski, włoski i hiszpański. Możesz dostosować tę listę, wybierając języki zdefiniowane w zmiennej AVAILABLE_LANGUAGES lub rozszerzając tę listę o nowe języki.

Samogłoski do analizy są wybierane z dostępnych opcji w bazie danych, zorganizowanej zgodnie z wcześniej opisaną strukturą. Analiza jest przeprowadzana osobno dla każdej wybranej samogłoski.

### Parametry przy wczytywaniu nagrań
Można dostosować długość wczytywanego nagrania i częstotliwość próbkowania w pliku [config.py](./config.py).
```python
# RECORDING SETTINGS
SR = 44100
SIGNAL_DURATION = 0.1   # time in [s]
```

### Parametry spektrogramów
Parametry dotyczące spektrogramów można dostosować, co może znacząco wpłynąć na wyniki eksperymentów. Masz możliwość wyboru między spektrogramami i mel-spektrogramami oraz dostosowania następujących parametrów:

Wybór rodzaju spektrogramu

```python
SPECTROGRAMS = False    # Wybierz True, aby przeprowadzić analizę dla spektrogramów
MELSPECTROGRAMS = True  # Wybierz True, aby przeprowadzić analizę dla mel-spectrogramów
```
Parametry spektrogramu

W przypadku `BINSIZE` oraz `OVERLAP` można wybrać więcej wartości, a analiza zostanie przeprowadzona iteracyjnie dla wszystkich kombinacji. 

```python
BINSIZE = [2048]  # Możesz dostosować rozmiar okna (binsize) spektrogramu
OVERLAP = [512]   # Możesz dostosować wartość overlap spektrogramu
N_MELS = 320      # Liczba kanałów mel
```

Techniki augmentacji
```python
AUGMENTATION = ["filtered", "pitch", "slow", "speed", "rolled", "noise"]
# Wybierz spośród dostępnych technik augmentacji: "filtered", "pitch", "slow", "speed", "rolled", "noise"
```

### Parametry uczenia

Następna kluczowa sekcja dotyczy parametrów uczenia, które możesz dostosować do swoich potrzeb. 
Plik [config.py](./config.py) umożliwia kontrolę następujących parametrów:

Rozmiar Paczki (Batch Size)
```python
BATCH_SIZE = 16  # Określ rozmiar paczki danych używany podczas uczenia
```
Klasyfikatory do Testów
```python
CLASSIFIERS_TO_TEST = ["VGG16",  "ResNet50", "Xception", "InceptionV3", "MobileNetV2"]
# Wybierz klasyfikatory do przetestowania spośród dostępnych: "VGG16", "ResNet50", "Xception", "InceptionV3", "MobileNetV2"
```
Podział Kroswalidacji
```python
CROSS_VALIDATION_SPLIT = 10  # Określ podział krosswalidacji
```

Urządzenie (CPU/GPU)
```python
DEVICE = "CPU"  # Wybierz urządzenie, na którym chcesz przeprowadzić uczenie: "CPU" lub "GPU"
```
Epoki Uczenia
```python
EPOCHS = 5  # Określ liczbę epok treningu
```

Współczynnik Uczenia (Learning Rate)
```python
LEARNING_RATE = 0.0005  # Dostosuj współczynnik uczenia (learning rate) dla optymalizatora
```

Optymizator (Optimizer)
```python
OPTIMIZER = 'sgd'  # Wybierz optymizator (optimizer) do uczenia modelu spośród dostępnych: "adam", "sgd", "rmsprop"
```
Dostosowanie tych parametrów pozwoli Ci na dostosowanie procesu uczenia do konkretnego problemu i zwiększenie skuteczności eksperymentów.

## Wizualizacja wyników
Po przeprowadzeniu eksperymentów wyniki są zapisywane w folderze results_język-analizy, na przykład results_pol lub results_pol+es. Możesz dostosować wyświetlanie informacji zwrotnej, takich jak wykresy i postęp uczenia, za pomocą następujących ustawień:
```python
# FEEDBACK SETTINGS
SHOW_PLOTS = True
SHOW_LEARNING_PROGRESS = False
```
### Organizacja wyników
Wyniki są zorganizowane osobno dla każdej samogłoski, a następnie dla poszczególnych ustawień spektrogramów i architektur CNN. Struktura folderów jest następująca:

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


Dzięki tej strukturze folderów możesz łatwo zarządzać i analizować wyniki eksperymentów dla różnych samogłosk, języków oraz ustawień spektrogramów i architektur sieci neuronowych. Wszystkie informacje, takie jak macierze pomyłek, krzywe uczenia i wyniki kroswalidacji, są zorganizowane w czytelny sposób.
Zachowywane są wyniki dla poszczególnych iteracji krosswalidacji, a także ich zbiorcze podsumowanie. 