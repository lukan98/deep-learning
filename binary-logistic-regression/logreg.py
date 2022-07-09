import numpy as np
import data

# TODO: Write documentation
def logreg_train(X, Y_, param_niter=1000, param_delta=0.2, verbose=False):
    if (X.shape[0] != Y_.shape[0]):
        raise ValueError("The dimensions of your data are mismatched!")

    data_count = X.shape[0]
    class_count = np.max(Y_) + 1
    feature_count = X.shape[1]

    W = np.random.randn(class_count, feature_count)  # CxD
    b = np.zeros(class_count).reshape(-1, 1)  # Cx1

    Y_matrix = np.zeros((data_count, class_count)) # NxC
    for i in range(len(Y_matrix)):
        Y_matrix[i][Y_[i]] = 1

    for i in range(param_niter):
        classification_scores = np.dot(X, W.T) + b.T  # NxC
        softmaxes = stable_softmax(classification_scores)
        softmaxes_sum = np.sum(softmaxes, axis=1).reshape(data_count, 1)

        probabilities = softmaxes / softmaxes_sum # NxC
        log_probabilities = np.log(np.sum(np.multiply(probabilities, Y_matrix), axis=1)).reshape(-1, 1)
        loss = np.sum(log_probabilities) * (-1)/data_count

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        Gs = probabilities - Y_matrix
        grad_W = 1/data_count * np.dot(Gs.T, X) # CxD
        grad_b = np.sum(Gs, axis=0).reshape(-1, 1) # Cx1

        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b

def stable_softmax(x):
    z = x - np.max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    return numerator / denominator

def logreg_classify(X, W, b):
    classification_scores = np.dot(X, W.T) + b.T  # NxC
    softmaxes = stable_softmax(classification_scores)
    softmaxes_sum = np.sum(softmaxes, axis=1).reshape(len(X), 1)
    return softmaxes/softmaxes_sum

def logreg_decfun(W, b):
    def classify(X):
        return logreg_classify(X, W, b)
    return classify

if __name__ == '__main__':
    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 10)

    # train the model
    W, b = logreg_train(X, Y_, verbose=True)
    print (logreg_classify(X, W, b))
