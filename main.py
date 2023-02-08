from data_loader import MetadataLoader
from classifiers.metadata_classifiers.DecisionTree import DecisionTree

from sklearn.metrics import classification_report

CLASSIFIERS_TO_TEST = ["decision_tree"]


if __name__ == "__main__":
    print("Run Parkinson Disease detection \n")
    data_loader = MetadataLoader()

    decision_tree_clf = DecisionTree(data_loader)

    for cls in CLASSIFIERS_TO_TEST:
        print("Results for {} classifier: ".format(cls))
        if cls == "decision_tree":
            y_pred = decision_tree_clf.run_classifier()
            print(classification_report(data_loader.y_test, y_pred))
