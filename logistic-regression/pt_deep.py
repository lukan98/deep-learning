import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import data
import matplotlib.pyplot as plt


class PTDeep(nn.Module):
    def __init__(self, architecture, activation_function):
        """Arguments:
           - architecture: The architecture of the network expressed as a list of integers.
           - activation_function: The non-linear activation function used in the hidden layers.
        """
        super().__init__()
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.activation_function = activation_function
        for i in range(len(architecture) - 1):
            self.weights.append(nn.Parameter(torch.randn((architecture[i], architecture[i + 1]))))
            self.biases.append(nn.Parameter(torch.zeros(1, architecture[i + 1])))

    def forward(self, X):
        input = X
        for i in range(len(self.biases)):
            input = torch.mm(input, self.weights[i]) + self.biases[i]
            input = self.activation_function(input)
        return torch.softmax(input, dim=1)

    def get_loss(self, X, Yoh_):
        correct_probabilities = torch.sum(torch.multiply(self.forward(X), Yoh_), dim=1)
        log_probabilities = torch.log(correct_probabilities + 1e-20)
        return torch.mean(log_probabilities) * (-1)

    def count_params(self, verbose=False):
        parameter_counter = 0
        for name, parameter in self.named_parameters():
            parameter_counter += torch.numel(parameter)
            if verbose:
                print(f'Parameter name: {name}')
                print(f'Parameter values:\n{parameter.detach().numpy()}')
        print(f'Ukupan broj parametara: {parameter_counter}')


def train(model, X, Yoh_, param_niter, param_delta, param_lambda=0,
          optimizer=optim.SGD, schedule=False,
          verbose=False, print_step=1,):
    """A deep model training function

       - X: model inputs [NxD], type: torch.Tensor
       - Yoh_: ground truth [NxC] (in one-hot notation), type: torch.Tensor
       - param_niter: number of training iterations
       - param_delta: learning rate
    """
    if (print_step <= 0):
        raise ValueError('The print step must be at least 1')
    if (not torch.is_tensor(X) or not torch.is_tensor(Yoh_)):
        raise ValueError("X and Yoh_ both must be torch.Tensor data types!")

    optimizer = optimizer(model.parameters(), lr=param_delta)

    if schedule:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=1-1e-4)

    for i in range(int(param_niter)):
        norm_sum = 0
        if param_lambda > 0:
            for weight_index in range(len(model.weights)):
                norm_sum += torch.linalg.norm(model.weights[weight_index])
        loss = model.get_loss(X, Yoh_) + param_lambda * norm_sum
        loss.backward()
        optimizer.step()
        if schedule:
            scheduler.step()
        if verbose and i % print_step == 0:
            print(f'step: {i}, loss:{loss}')
        optimizer.zero_grad()


def eval(model, X):
    """
       - model: type: PTLogreg
       - X: actual datapoints [NxD], type: np.array
       Returns: predicted class probabilites [NxC], type: np.array
    """
    X = torch.Tensor(X).detach()
    return torch.Tensor.numpy(model.forward(X).detach())


def decfun(model, X):
    def classify(X):
        return eval(model, X)[:, 0]
    return classify


if __name__ == '__main__':
    class_count = 2
    sample_count = 6
    example_count = 10

    # instanciraj podatke X i labele Yoh_
    X, Yoh_ = data.sample_gmm_2d(sample_count, class_count, example_count)
    while len(set(Yoh_)) != class_count:
        X, Yoh_ = data.sample_gmm_2d(sample_count, class_count, example_count)

    Y_matrix = data.class_to_onehot(Yoh_)

    model = PTDeep([2, 10, 10, class_count], activation_function=torch.sigmoid)

    train(model, torch.Tensor(X), torch.Tensor(Y_matrix),
          param_niter=10000, param_delta=0.1, param_lambda=0.001)

    probs = eval(model, X)
    Y_hat = np.argmax(probs, axis=1)

    model.count_params()

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, precision, confusion_matrix = data.eval_perf_multi(Y_hat, Yoh_)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Confusion matrix:\n', confusion_matrix)

    # iscrtaj rezultate, decizijsku plohu
    rect = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun(model, X), rect)
    data.graph_data(X, Yoh_, Y_hat)
    plt.show()
