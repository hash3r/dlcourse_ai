import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        # TODO Create necessary layers

        self.l1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.l2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.reg = reg
        self.w1 = self.l1.params()['W']
        self.w2 = self.l2.params()['W']
        self.b1 = self.l1.params()['B']
        self.b2 = self.l1.params()['B']

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
#         [self.params()[param].grad.fill(0) for param in self.params().keys()]

        fout = self.l1.forward(X)
        fout = self.relu.forward(fout)
        fout = self.l2.forward(fout)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        loss, d_out = softmax_with_cross_entropy(fout, y)
        
        bout = self.l2.backward(d_out)
        bout = self.relu.backward(bout)
        bout = self.l1.backward(bout)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        for param in self.params().values():
            reg_loss, reg_grad = l2_regularization(param.value, self.reg)
            loss += reg_loss
            param.grad += reg_grad
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        pred = np.argmax(softmax(self.l2.forward(self.relu.forward(self.l1.forward(X)))), 1)
        return pred

    def params(self):
        # TODO Implement aggregating all of the params
#         result = {'B1': self.l1.B, 'W1': self.l1.W, 'W2': self.l2.W, 'B2': self.l2.B}
        result = {"W1": self.w1, "W2": self.w2, "B1": self.b1, "B2": self.b2}
        return result
