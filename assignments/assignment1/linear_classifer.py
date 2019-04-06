import numpy as np


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
    # Your final implementation shouldn't have any loops

#     maxes = np.max(predictions, axis=1)[:, np.newaxis]
#     predictions_normalized = predictions - maxes
#     predictions_exp = np.exp(predictions_normalized)
#     predictions_sum = np.sum(predictions_exp, axis=1)[:, np.newaxis]

#     return predictions_exp/predictions_sum

    predictions_limited = predictions.copy()
    if len(predictions.shape) == 2:
        predictions_limited -= np.amax(predictions, axis = 1)[:, np.newaxis]
    else:
        predictions_limited -= np.max(predictions)
    exps = np.exp(predictions_limited)
    print(exps)
    print(np.sum(exps, axis = 1))
    print(np.sum(exps, axis = 1)[:, np.newaxis])
    print(np.sum(exps))
    print(exps / np.sum(exps))
    if len(predictions.shape) == 2:
        return exps / np.sum(exps, axis = 1)[:, np.newaxis]
    else:
        return exps / np.sum(exps)



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
    # Your final implementation shouldn't have any loops
#     y = target_index.copy()
#     X = probs.copy()
    
#     log_likelihood = np.log(probs[range(m), y.T])
#     loss = -np.mean(np.sum(log_likelihood))# / m
    
    if len(probs.shape) == 2:
        m = target_index.shape[0]
        return -np.mean(np.log(probs[range(m), target_index.T]))
    else:
        return -np.log(probs[target_index])
    

#     rows_count = target_index.shape[0]
#     return np.sum(-np.log(probs[range(rows_count), target_index]))/rows_count

#     print("log_likelihood", log_likelihood, "loss", loss)#, "loss2", loss2)
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dpred = probs
    if len(dpred.shape) > 1:
        m = predictions.shape[0]
        dpred[np.arange(m), target_index.T] -= 1
        dpred /= m
    else:
        dpred[target_index] -= 1
    return loss, dpred


#     probs = softmax(predictions)
#     loss = cross_entropy_loss(probs, target_index)

#     y = target_index.copy()
#     X = predictions.copy();
#     m = y.shape[0]

#     probs[range(m), y.T] -= 1 #this is a derivation of function softmax and cross entropy
#     probs /= m


#     rows_count = target_index.shape[0]
#     probas = softmax(predictions)
#     loss = cross_entropy_loss(probas, target_index)
#     probas[range(rows_count), target_index] -= 1
#     probas /= rows_count
#     return loss, probas

#     print("sftmax \w ce", "probs", probs, target_index, probs - target_index, "grad", grad)
        
    return loss, probs


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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W*W);
    grad = reg_strength * 2 * W;
    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    predictions = X@W
    loss, dW = softmax_with_cross_entropy(predictions, target_index)
    grad = X.T@dW
#     print("loss\n", loss)
#     print("dW\n", dW)
#     print("grad\n", grad)
    return loss, grad


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

#             print(shuffled_indices)
#             print(sections)
#             print(batches_indices)
            
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch in batches_indices:
                loss, grad = linear_softmax(X[batch], self.W, y[batch])
                l2_loss, l2_grad = l2_regularization(self.W, reg)
                loss += l2_loss
                grad += l2_grad
                self.W -= learning_rate * grad
                
            loss_history.append(loss) #how???
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        probs = softmax(X @ self.W)
        y_pred = np.argmax(probs, axis = 1)

        return y_pred



                
                                                          

            

                
