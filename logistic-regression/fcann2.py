import numpy as np
import matplotlib.pyplot as plt
import data


def fcann2_train(X, Y_, param_niter=10000, param_delta=0.05, hidden_layer_size=5, verbose=False):
    if (X.shape[0] != Y_.shape[0]):
        raise ValueError("The dimensions of your data are mismatched!")

    data_count = X.shape[0]
    class_count = np.max(Y_) + 1
    feature_count = X.shape[1]

    W_1 = np.random.randn(hidden_layer_size, feature_count)  # HxD
    b_1 = np.zeros((hidden_layer_size, 1))  # Hx1
    W_2 = np.random.randn(class_count, hidden_layer_size)  # CxH
    b_2 = np.zeros((class_count, 1))  # Cx1

    Y_matrix = data.class_to_onehot(Y_)

    for i in range(param_niter):
        s_1 = np.dot(X, W_1.T) + b_1.T  # NxH
        h_1 = np.where(s_1 > 0, s_1, 0)  # NxH
        s_2 = np.dot(h_1, W_2.T) + b_2.T  # NxC
        h_2 = stable_softmax(s_2)  # NxC

        log_probabilities = np.log(np.sum(np.multiply(h_2, Y_matrix), axis=1)).reshape(-1, 1)
        loss = np.sum(log_probabilities + 1e-20) * (-1) / data_count

        # diagnostics
        if verbose and i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Gs_2 = h_2 - Y_matrix  # NxC
        Gs_2 /= data_count
        grad_W_2 = np.dot(Gs_2.T, h_1)  # CxH
        grad_b_2 = np.sum(Gs_2, axis=0).reshape(-1, 1)  # Cx1

        Gh_1 = np.dot(Gs_2, W_2)  # NxH
        Gs_1 = np.multiply(Gh_1, np.where(s_1 > 0, 1, 0))  # NxH
        grad_W_1 = np.dot(Gs_1.T, X)  # HxD
        grad_b_1 = np.sum(Gs_1, axis=0).reshape(-1, 1)  # Hx1

        W_1 += -param_delta * grad_W_1
        W_2 += -param_delta * grad_W_2
        b_1 += -param_delta * grad_b_1
        b_2 += -param_delta * grad_b_2

    return W_1, b_1, W_2, b_2


def fcann2_classify(X, W_1, b_1, W_2, b_2):
    data_count = X.shape[0]

    s_1 = np.dot(X, W_1.T) + b_1.T  # NxH
    h_1 = np.where(s_1 > 0, s_1, 0)  # NxH
    s_2 = np.dot(h_1, W_2.T) + b_2.T  # NxC
    h_2 = stable_softmax(s_2)  # NxC

    softmaxes_sum = np.sum(h_2, axis=1).reshape(data_count, 1)
    return h_2 / softmaxes_sum


def fcann2_decfun(W_1, b_1, W_2, b_2):
    def classify(X):
        return fcann2_classify(X, W_1, b_1, W_2, b_2)[:, 0]

    return classify


def stable_softmax(x):
    z = x - np.amax(x, axis=1).reshape(-1, 1)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1).reshape(-1, 1)
    return numerator / denominator


if __name__ == '__main__':
    class_count = 3
    sample_count = 6
    example_count = 10

    X, Y_ = data.sample_gmm_2d(sample_count, class_count, example_count)
    while (len(set(Y_)) != class_count):
        X, Y_ = data.sample_gmm_2d(sample_count, class_count, example_count)

    W_1, b_1, W_2, b_2 = fcann2_train(X, Y_)
    probabilities = fcann2_classify(X, W_1, b_1, W_2, b_2)
    Y = np.argmax(probabilities, axis=1)
    # performance review
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y, Y_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # plot results
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    decision = fcann2_decfun(W_1, b_1, W_2, b_2)

    data.graph_surface(decision, rect, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])
    plt.show()
