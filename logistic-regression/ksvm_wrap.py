import numpy as np
from sklearn import svm
import data
import matplotlib.pyplot as plt


class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        self.clf = svm.SVC(kernel='rbf', C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)

    def support(self):
        return self.clf.support_

if __name__ == '__main__':
    class_count = 2
    sample_count = 6
    example_count = 10

    X, Y_ = data.sample_gmm_2d(sample_count, class_count, example_count)
    while (len(set(Y_)) != class_count):
        X, Y_ = data.sample_gmm_2d(sample_count, class_count, example_count)

    model = KSVMWrap(X, Y_)
    Y_hat = model.predict(X)

    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_hat, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(model.get_scores, rect, offset=0)
    data.graph_data(X, Y_, Y=Y_hat, special=model.support())
    plt.show()