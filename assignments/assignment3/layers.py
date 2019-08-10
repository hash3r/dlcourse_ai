import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
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
    
    
def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.dX = np.array(X > 0).astype(int)
        return np.maximum(X, 0) #relu imp

    def backward(self, d_out):
        d_result = self.dX * d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = np.copy(X)
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        d_w = np.dot(self.X.T, d_out)
        d_b = d_out.sum(axis=0)[None, :]
        self.W.grad += d_w
        self.B.grad += d_b
        d_X = np.dot(d_out, self.W.value.T)
        return d_X

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = height - self.filter_size + 2*self.padding + 1
        out_width = width - self.filter_size + 2*self.padding + 1
#         out_height = height - (self.filter_size - 1) + self.padding * 2
#         out_width = width - (self.filter_size - 1) + self.padding * 2

        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        self.X = np.copy(X)
        X_out = np.full((batch_size, out_height, out_width, self.out_channels), 0)
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_res = np.reshape(X[:, y: y + self.filter_size, x: x + self.filter_size, :], 
                                  (batch_size, self.filter_size * self.filter_size * channels))
                W_res = np.reshape(self.W.value[y: y + self.filter_size, x: x + self.filter_size, :, :], 
                                  (self.filter_size * self.filter_size * channels, self.out_channels))
                X_out[:, y, x, :] = X_res @ W_res + self.B.value 
        return X_out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        d_res = np.full((batch_size, height, width, channels), 0)
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                d_out_res = np.reshape(d_out[:, y, x, :], (batch_size, out_height * out_width * out_channels))

                W_res = np.reshape(self.W.value[:, y: y + self.filter_size, x: x + self.filter_size, :], 
                                   (self.filter_size * self.filter_size * channels, self.out_channels))
                
                d_res[:, y: y + self.filter_size, x: x + self.filter_size, :] = np.reshape(d_out_res @ W_res.T, 
                                                                                          (batch_size, height, width, channels))
                
                self.W.grad[y: y + self.filter_size, x: x + self.filter_size, :, :] += np.reshape(d_out_res @ self.X.T,
                                                                                                  (self.filter_size, self.filter_size, channels, out_channels))
        
        self.B.grad += np.reshape(d_out, (out_channels, batch_size)).sum(axis=0)
        
        return d_res

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
