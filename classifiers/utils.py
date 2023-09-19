from classifiers.MyXception import MyXception
from classifiers.MyInceptionV3 import MyInceptionV3
from classifiers.MyResNet50 import MyResNet50
from classifiers.MyVGG16 import MyVGG16
from classifiers.MyMobileNet import MyMobileNet
from config import CLASSIFIERS_TO_TEST


def initialize_classifiers(train, test, setting, settings_dir, val):
    classifiers = {}
    for cls in CLASSIFIERS_TO_TEST:
        if cls == "VGG16":
            classifiers[cls] = MyVGG16(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "ResNet50":
            classifiers[cls] = MyResNet50(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "Xception":
            classifiers[cls] = MyXception(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "MobileNetV2":
            classifiers[cls] = MyMobileNet(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )
        elif cls == "InceptionV3":
            classifiers[cls] = MyInceptionV3(
                train, test, settings=setting, results_dir=settings_dir, val_data=val
            )

    return classifiers
