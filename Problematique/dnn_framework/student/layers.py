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

    parameters = {}
    buffers = {}

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.input_count = input_count
        self.parameters["gamma"] = np.ones(input_count)
        self.parameters["beta"] = np.zeros(input_count)

        self.buffers["global_mean"] = np.zeros(input_count)
        self.buffers["global_variance"] = np.zeros(input_count)

    def get_parameters(self):
        return self.parameters

    def get_buffers(self):
        return self.buffers

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)


    def _forward_training(self, x):
        mu_b = np.mean(x, axis=0)
        sig_square_b = np.var(x, axis=0)
        x_hat = (x - mu_b) / np.sqrt(sig_square_b + 1e-12)
        y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
        return y, {"input": x, "output": y, "mu_b":mu_b, "sig_square_b":sig_square_b, "x_hat": x_hat}

    def _forward_evaluation(self, x):
        mu_b = self.buffers["global_mean"]
        sig_square_b = self.buffers["global_variance"]
        x_hat = (x - mu_b) / np.sqrt(sig_square_b + 1e-12)
        y = self.parameters["gamma"] * x_hat + self.parameters["beta"]
        return y, {"input": x, "output": y, "mu_b":mu_b, "sig_square_b":sig_square_b, "x_hat": x_hat}

    def backward(self, output_grad, cache):
        deriv_x_hat = output_grad * self.parameters["gamma"]
        deriv_gamma = np.sum(output_grad*cache["output"],axis=0)
        deriv_beta = np.sum(output_grad,axis=0)


        M = cache["input"].shape[0]
        deriv_x = (1/M) * (1/np.sqrt(cache["sig_square_b"] + 1e-12)) * (M * deriv_x_hat - np.sum(deriv_x_hat,axis=0) - cache["x_hat"]* np.sum(deriv_x_hat*cache["x_hat"],axis=0))

        # deriv_sig_square_b = np.sum(deriv_x_hat*(cache["input"]-cache["mu_b"])*-0.5*((cache["sig_square_b"] + 1e-12)**(-3/2)))
        #
        # deriv_mu_b = - np.sum(deriv_x_hat/np.sqrt(cache["sig_square_b"]+1e-12))
        # deriv_x = deriv_x_hat / np.sqrt(cache["sig_square_b"]+1e-12) + 2/self.input_count * deriv_sig_square_b * (cache["input"]-cache["mu_b"]) + 1/self.input_count * deriv_mu_b

        return (deriv_x, {"gamma":deriv_gamma,"beta":deriv_beta})


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
        y = np.maximum(x, 0)
        return y, {"input": x, "output": y}

    def backward(self, output_grad, cache):
        X = cache["input"]
        dev_relu = np.where(X > 0, 1, 0)
        return dev_relu * output_grad, {}
