from abc import ABC, abstractmethod


class ClassifierBase(ABC):

    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def run_classifier(self):
        pass


