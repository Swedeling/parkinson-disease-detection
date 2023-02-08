from sklearn.tree import DecisionTreeClassifier, export_graphviz

from classifiers.classifier_base import ClassifierBase


class DecisionTree(ClassifierBase):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self._name = "decision_tree"

    def run_classifier(self):

        tree_clf = DecisionTreeClassifier()
        tree_clf.fit(self.data_loader.x_train, self.data_loader.y_train)

        y_pred = tree_clf.predict(self.data_loader.x_test)

        export_graphviz(tree_clf,
                        out_file="results/decision_tree/tree_diagram.dot",
                        feature_names=self.data_loader.features_names,
                        class_names=self.data_loader.class_names,
                        rounded=True,
                        filled=True)

        return y_pred
