import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        # self.w = np.random.normal(loc=0.0,
        #                   size=(output_count, input_count))
        w = np.zeros((output_count,input_count))
        b = np.zeros(output_count)
        self.parameters = {"w":w,"b":b}


    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        y = x @ self.parameters['w'].T + self.parameters['b']
        return (y, {"input":x, "output":y})

    def backward(self, output_grad, cache):
        deriv_W = output_grad.T @ cache["input"]
        deriv_b = np.sum(output_grad, axis=0)
        deriv_X = output_grad @ self.parameters["w"]
        return (deriv_X, {"w":deriv_W,"b":deriv_b})


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

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
        y = 1/(1+np.exp(-x))
        output = (y,{"input":x,"output":y})
        return output

    def backward(self, output_grad, cache):

        deriv_sigm = cache["output"] * (1 - cache["output"])
        return (deriv_sigm * output_grad , {})


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()
