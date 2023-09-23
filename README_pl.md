# Detekcja anomalii w sygnale mowy u osób z chorobą Parkinsona


Celem projektu było przeprowadzenie badań porównawczych w zakresie automatycznej 
diagnostyki choroby Parkinsona na podstawie głosu.
Wykorzystano dane obrazowe sygnału głosu samogłosek /a/, /e/, /i/, /o/, /u/ w postaci 
melspektrogramów i zaproponowano sieci neuronowe CNN o różnej architekturze, w tym sieci VGG16, ˙
ResNet50, Xception, InceptionV3 ora MobileNetV2. 
Przedstawiony kod umożliwia przeprowadzenie takiej analizy dla różnych samogłosek, języków, architektur CNN i ustawień spektrogramów.
Wyniki i modele są zapisywane automatycznie.
Przedstawiono je w postaci metryki dokładnosci, F1, specyficzności i precyzji, a także monitorowania procesu uczenia (krzywe dokłądności i straty) oraz macierze poomyłek.




### Przygotowanie danych

Edit this [file](./config.py)
```python
# SPECTROGRAM SETTINGS
AUGMENTATION = ["filtered", "pitch", "slow", "speed", "rolled", "noise"]
BINSIZE = [2048]
OVERLAP = [512]
N_MELS = 320
```




### Ustawienie eksperymentów


### Wizualizacja wyników