import numpy as np
import data


def binlogreg_train(X, Y_, param_niter=1000, param_delta=0.2, verbose=False):
    """Function used to train a binary regression model.

    :param X: Training data, np.array NxD (N being the number of data points, D being the dimensionality)
    :param Y_: Class indices, np. array Nx1 (N being the number of data points)
    :param param_niter: The number of iterations the model will make
    :param param_delta: The learning rate of the model
    :param verbose: Boolean which controls diagnostic output
    :return: Parameters of the logistic regression model. The first returned value are the coefficients, the second is the intercept.
    """
    if (X.shape[0] != Y_.shape[0]):
        raise ValueError("The dimensions of your data are mismatched!")

    n = X.shape[0]
    d = X.shape[1]
    w = np.random.randn(d, 1)
    b = 0

    for i in range(param_niter):
        classification_scores = np.dot(X, w) + b
        probabilities = np.exp(classification_scores)/(1+np.exp(classification_scores))
        loss = np.sum(np.log(Y_*probabilities+(1-Y_)*(1-probabilities)))*(-1/n)

        if i % 10 == 0 and verbose:
            print("iteration {}: loss {}".format(i+1, loss))

        dL_dscores = probabilities - Y_

        grad_w = 1/n * np.dot(X.T, dL_dscores)
        grad_b = 1/n * np.sum(dL_dscores)

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    return w, b

def binlogreg_classify(X, w, b):
    """Function that uses binary logistic regression to classify a dataset.

    :param X: The supplied dataset, np.array NxD (N being the number of data points, D being the dimensionality)
    :param w: The coefficients of the binary logistic regression model
    :param b: The intercept of the binary logistic regression model
    :return: The respective probabilities of the data points belonging to the c_1 class.
    """
    classification_scores = np.dot(X, w) + b
    return np.exp(classification_scores)/(1 + np.exp(classification_scores))

def binlogreg_decfun(w, b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify

if __name__ == '__main__':
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(2, 100)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = np.where(probs > 0.5, 1, 0)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[np.argsort(probs, 0)])
    print(accuracy, recall, precision, AP)