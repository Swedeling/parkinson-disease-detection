from classifiers.MyAlexNet import MyAlexNet
from classifiers.GNet import GNet
from classifiers.InceptionV3 import InceptionNet
from classifiers.ResNet50 import ResNet
from classifiers.VGGNet import VGGNet
from classifiers.LeNet5 import LeNet5
from classifiers.MyMobileNet import MyMobileNet


from config import CLASSIFIERS_TO_TEST


def initialize_classifiers(train, test, setting, settings_dir, val):
    classifiers = {}
    for cls in CLASSIFIERS_TO_TEST:
        if cls == "AlexNet":
            classifiers[cls] = MyAlexNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "GNet":
            classifiers[cls] = GNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "InceptionV3":
            classifiers[cls] = InceptionNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "LeNet-5":
            classifiers[cls] = LeNet5(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "ResNet50":
            classifiers[cls] = ResNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "VGGNet":
            classifiers[cls] = VGGNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "MobileNet":
            classifiers[cls] = MyMobileNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
    return classifiers
