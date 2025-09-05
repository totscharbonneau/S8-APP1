import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    parameters = {}
    buffers = {}

    def __init__(self, input_count, alpha=0.1):
        self.parameters["gamma"] = np.ones(input_count)
        self.parameters["beta"] = np.zeros(input_count)

        self.buffers["global_mean"] = np.zeros(input_count)
        self.buffers["global_variance"] = np.zeros(input_count)

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        mu_b = np.mean(x, axis=0)
        sig_square_b = np.var(x, axis=0)
        x_hat = (x - mu_b) / np.sqrt(sig_square_b + 1e-12)
        y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
        return y, {"input": x, "output": y}


    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):

        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = np.maximum(x, 0)
        return y, {"input": x, "output": y}

    def backward(self, output_grad, cache):
        X = cache["input"]
        dev_relu = np.where(X > 0, 1, 0)
        return dev_relu * output_grad, {}
