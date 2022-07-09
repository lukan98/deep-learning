import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt


class PTLogreg(nn.Module):
    def __init__(self, D, C):
        """Arguments:
           - D: dimensions of each datapoint
           - C: number of classes
        """
        super().__init__()
        self.weights = nn.Parameter(torch.randn((C, D)))
        self.b = nn.Parameter(torch.zeros((C, 1)))

    def forward(self, X):
        classification_scores = torch.mm(X, torch.t(self.weights)) + torch.t(self.b)
        return torch.softmax(classification_scores, dim=1)

    def get_loss(self, X, Yoh_):
        correct_probabilites = torch.sum(torch.multiply(self.forward(X), Yoh_), dim=1)
        log_probabilities = torch.log(correct_probabilites + 1e-20)
        return -torch.mean(log_probabilities)


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0, verbose=False):
    """A method for training logistic regression models.

    The model passed as the parameter model is trained using the input data {X, Yoh_}.
    :param model: Model to be trained
    :param X: Input data
    :param Yoh_: True output values
    :param param_niter: Number of training epochs
    :param param_delta: Learning rate
    :param param_lambda: Regularization factor
    :param verbose: Boolean controling the diagnostic output
    """
    if (not torch.is_tensor(X) or not torch.is_tensor(Yoh_)):
        raise ValueError("X and Yoh_ both must be torch.Tensor data types!")

    optimizer = optim.SGD([model.weights, model.b], lr=param_delta)

    for i in range(int(param_niter)):
        loss = model.get_loss(X, Yoh_) + param_lambda * torch.linalg.norm(model.weights)
        loss.backward()
        optimizer.step()
        if verbose == True:
            print(f'step: {i}, loss:{loss}')
        optimizer.zero_grad()


def eval(model, X):
    """An evaluation method.

    :param model: type: PTLogreg
    :param X: actual datapoints [NxD], type: np.array
    :return: predicted class probabilites [NxC], type: np.array
    """
    X = torch.Tensor(X).detach()
    return torch.Tensor.numpy(model.forward(X).detach())


def decfun(model, X):
    def classify(X):
        return eval(model, X)[:, 0]

    return classify


if __name__ == "__main__":
    class_count = 3
    sample_count = 6
    example_count = 10

    # instanciraj podatke X i labele Yoh_
    X, Yoh_ = data.sample_gmm_2d(sample_count, class_count, example_count)
    while len(set(Yoh_)) != class_count:
        X, Yoh_ = data.sample_gmm_2d(sample_count, class_count, example_count)

    Y_matrix = data.class_to_onehot(Yoh_)

    # definiraj model:
    ptlr = PTLogreg(X.shape[1], Y_matrix.shape[1])

    # nauči parametre (X i Yoh_ moraju biti tipa torch.Tensor):
    train(ptlr, torch.Tensor(X), torch.Tensor(Y_matrix),
          param_niter=1000, param_delta=0.1, param_lambda=0)

    # dohvati vjerojatnosti na skupu za učenje
    probs = eval(ptlr, X)
    Y_hat = np.argmax(probs, axis=1)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_hat, Yoh_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun(ptlr, X), rect)
    data.graph_data(X, Yoh_, Y_hat)
    plt.show()
