import numpy as np
import matplotlib.pyplot as plt
import binlogreg

# TODO: Add support for multiple classification
class Random2DGaussian:
    """Class used for generating normally distributed 2-dimensional data points"""
    minx = 0
    maxx = 10
    miny = 0
    maxy = 10

    def __init__(self):
        self.mean = np.random.random_sample(2) * [self.maxx - self.minx, self.maxy - self.miny] + [self.minx, self.miny]
        eigenvalues = np.random.random_sample(2) * np.array([self.maxx - self.minx, self.maxy - self.miny]) / 5 ** 2
        d = np.diag(eigenvalues)
        random_radian = np.random.random() * 2 * np.pi
        r = np.matrix([[np.cos(random_radian), -np.sin(random_radian)],
                       [np.sin(random_radian), np.cos(random_radian)]])
        self.sigma = np.linalg.multi_dot([r.T, d, r])

    def get_sample(self, no_of_samples):
        """Returns a generated normally distributed 2-dimensional data sample

        :param no_of_samples: Number of samples to be generated.
        :return: Generated samples as a 2-dimensional array of their coordinates.
        """
        return np.random.multivariate_normal(self.mean, self.sigma, no_of_samples)


def sample_gauss_2d(C, N):
    """Returns a C number of N-sized

    :param C: Number of samples to be generated
    :param N: Sample sizes
    :return X, Y: The X matrix contains the generated data and the Y matrix contains the respective class index.
    """
    for i in range(C):
        G = Random2DGaussian()
        if (i == 0):
            X = G.get_sample(N)
            Y = np.array([[0]] * N)
        else:
            X = np.vstack((X, G.get_sample(N)))
            Y = np.vstack((Y, np.array([[i]] * N)))

    return X, Y


def eval_perf_binary(Y, Y_):
    """A function that evaluates the performance of a binary logistic regression model.

    :param Y: Predicted class indices
    :param Y_: True class indices
    :return: Accuracy, precision and recall
    """
    true_positives = np.sum(np.logical_and(Y == 1, Y_ == 1))
    true_negatives = np.sum(np.logical_and(Y == 0, Y_ == 0))
    false_positives = np.sum(np.logical_and(Y == 1, Y_ == 0))
    false_negatives = np.sum(np.logical_and(Y == 0, Y_ == 1))

    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)

    return accuracy, precision, recall


def eval_AP(ranked_labels):
    """Function that evaluates the average precision for the provided ranked binary labels.

    :param ranked_labels: Binary class labes ranked by their predicted probability e.g. the certainty of the model in that particular label.
    :return: The average precision.
    """

    n = len(ranked_labels)
    positives = np.sum(ranked_labels)
    ap_sum = 0

    for i in range(len(ranked_labels)):
        if (ranked_labels[i] == 1):
            true_positives = np.sum(ranked_labels[i:])
            all_positives = len(ranked_labels) - i
            precision = true_positives / all_positives
            ap_sum += precision

    ap_sum /= positives
    return ap_sum


def graph_data(X, Y_, Y):
    """Function that creates a scatter plot based of the classifications made by a binary logistic regression model.

    :param X: Input data
    :param Y_: True classes of the respective data points
    :param Y: Predicted classes made by the model
    """
    correct = np.where(Y_ == Y)[0]
    incorrect = np.where(Y_ != Y)[0]

    colors = np.array(['grey' if y_ == 0 else 'white' for y_ in Y_])

    plt.scatter(X[correct][:, 0], X[correct][:, 1], marker='o', c=colors[correct], edgecolors='black')
    plt.scatter(X[incorrect][:, 0], X[incorrect][:, 1], marker='s', c=colors[incorrect], edgecolors='black')

def graph_surface(function, rect, offset=0.5, width=256, height=256):
    """Function that contours and paints the mesh grid created by the dataset.

    :param function: Decision function of the binary logistic regresssion model
    :param rect: The bounding box of the dataset. Given as a list of tuples: [(min_x, min_y), (max_x, max_y)]
    :param offset: The separating value of the two classes, e.g. 0 for models like the SVM and 0.5 for probabilistic models.
    :param width: Width of the created graph.
    :param height: Height of the created graph.
    :return: None
    """
    min_x, min_y = rect[0]
    max_x, max_y = rect[1]
    x_range = np.linspace(min_x, max_x, width)
    y_range = np.linspace(min_y, max_y, height)
    x, y = np.meshgrid(x_range, y_range)
    xy_grid = np.array(np.meshgrid(x_range, y_range)).flatten('F').reshape(-1, 2)
    f_values = np.reshape(function(xy_grid), x.shape, order='F')
    v_max = np.max(f_values) - offset
    v_min = np.min(f_values) - offset

    plt.pcolormesh(x, y, f_values, vmin=v_min, vmax=v_max, shading='auto')
    plt.contour(x, y, f_values, colors='black', levels=[offset])

if __name__ == '__main__':
    # get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg.binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probabilities = binlogreg.binlogreg_classify(X, w, b)

    # recover the predicted classes
    Y = np.where(probabilities > 0.5, 1, 0)

    # evaluate and print the performance measures
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[np.argsort(probabilities, 0)])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    dec_fun = binlogreg.binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(dec_fun, bbox, offset=0.5)

    # graph the data points
    graph_data(X, Y_, Y)
    # show the plot
    plt.show()
