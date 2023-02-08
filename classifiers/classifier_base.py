from abc import ABC, abstractmethod


class ClassifierBase(ABC):
    def __init__(self):
        self._name = None

    @abstractmethod
    def run_classifier(self):
        pass


