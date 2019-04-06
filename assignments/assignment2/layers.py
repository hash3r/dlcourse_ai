import numpy as np

def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient
    Arguments:
      W, np array - weights
      reg_strength - float value
    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    # raise Exception("Not implemented!")
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * W * reg_strength
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    # TODO implement softmax
    x = predictions.copy()
    if len(predictions.shape) == 1:
        x -= np.max(x)
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x -= np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    if len(probs.shape) == 1:
        return -np.log(probs[target_index])
    else:
        n_samples = probs.shape[0]
        return np.mean(-np.log(probs[np.arange(n_samples), target_index]))

def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (N, batch_size) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probes = softmax(preds)
    loss = cross_entropy_loss(probes, target_index)
    dprediction = probes.copy()

    if len(preds.shape) == 1:
        dprediction[target_index] -= 1
    else:
        n_samples = probes.shape[0]
        dprediction[np.arange(n_samples), target_index] -= 1
        dprediction /= n_samples

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.dX = np.array(X > 0).astype(int)
        return np.maximum(X, 0) #relu imp
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = self.dX * d_out
#         print("d_out.shape", d_out.shape)
#         print("self.dX", self.dX.shape)
#         print(d_result)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
#         res = (X @ self.W.value) + self.B.value
        self.X = np.copy(X)
        return np.dot(X, self.W.value) + self.B.value
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
#         print("d_out.shape", d_out.shape)
#         print("self.W", self.W.value.shape)
#         print("self.X", self.X.shape)
        
#         np.ones_like(output) * output_weight
#         dX = d_out @ self.W.value.T #.reshape(x.shape)
        #self.W.grad = dX ???
#         self.W.grad = self.W.value.T @ d_out #x.reshape(self.X.shape[0], self.W.value.shape[0]).T.dot(d_out)
#         self.B.grad = np.sum(d_out, axis=1) # 1 for sure???
#         print("dX", dX)
#   dW2 = np.dot(hidden_layer.T, dscores)
#   db2 = np.sum(dscores, axis=0, keepdims=True)
#         return dX

        d_w = np.dot(self.X.T, d_out)
        d_b = d_out.sum(axis=0)[None, :]
        self.W.grad += d_w
        self.B.grad += d_b
        d_X = np.dot(d_out, self.W.value.T)
        return d_X

    def params(self):
        return {'W': self.W, 'B': self.B}
