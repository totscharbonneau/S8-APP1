import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        probs = softmax(x)

        # Compute cross-entropy loss
        N = x.shape[0]
        log_likelihood = -np.log(probs[np.arange(N), target] + 1e-12)  # avoid log(0)
        loss = np.mean(log_likelihood)

        # Backward: gradient wrt logits
        grad = probs.copy()
        grad[np.arange(N), target] -= 1
        grad /= N

        return loss, grad

def softmax(x):
    """
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        loss = np.mean((x-target)**2)
        grad = (2/x.size) * (x-target)
        return (loss,grad)
