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
    grad = 2 * reg_strength * W

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
    probs = []

    if len(predictions.shape) == 1:
        probs = predictions - np.max(predictions)
        probs = np.exp(probs) / np.sum(np.exp(probs))
    else:
        probs = predictions - np.max(predictions, axis=1).reshape(-1, 1)
        probs = np.exp(probs) / np.sum(np.exp(probs), axis=1).reshape(-1, 1)
    return probs


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
    loss = 0

    if len(probs.shape) == 1:
        loss = - np.log(probs[target_index])
    else:
        loss = - \
            np.sum(
                np.log(probs[np.arange(probs.shape[0]), target_index.flatten()]))
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
    dprediction = softmax(predictions)
    loss = cross_entropy_loss(dprediction, target_index)
    if len(predictions.shape) == 1:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(predictions.shape[0]),
                    target_index.flatten()] -= 1

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
        self.is_X_big = X > 0
        return X * self.is_X_big

    def backward(self, d_out):

        return d_out * self.is_X_big

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        self.B.grad += np.sum(d_out, axis=0).reshape(1, -1)
        self.W.grad += self.X.T.dot(d_out)
        d_result = d_out.dot(self.W.value.T)
        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}


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
        self.X = X
        out_height = height - self.filter_size + 2 * self.padding + 1
        out_width = width - self.filter_size + 2 * self.padding + 1

        result = np.zeros(
            (batch_size, out_height, out_width, self.out_channels))

        X_with_padding = np.zeros(
            (batch_size, height + 2 * self.padding, width + 2 * self.padding, self.in_channels))
        X_with_padding[:, self.padding:height + self.padding,
                       self.padding:width + self.padding, :] = X
        self.X_with_padding = X_with_padding

        W = self.W.value.reshape(
            self.filter_size ** 2 * self.in_channels, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                slice = X_with_padding[:, y:y + self.filter_size, x:x + self.filter_size,
                                       :].reshape(batch_size, self.filter_size ** 2 * self.in_channels)
                result[:, y, x, :] = slice.dot(W) + self.B.value
        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        self.B.grad += np.sum(d_out, axis=(0, 1, 2))
        W = self.W.value.reshape(
            self.filter_size ** 2 * self.in_channels, self.out_channels)

        result = np.zeros_like(self.X_with_padding)

        for y in range(out_height):
            for x in range(out_width):
                X_slice = self.X_with_padding[:, y:y + self.filter_size,
                                              x:x + self.filter_size, :].reshape(batch_size, self.filter_size ** 2 * self.in_channels)
                d_out_slice = d_out[:, y, x, :].reshape(
                    batch_size, self.out_channels)
                self.W.grad += X_slice.T.dot(d_out_slice).reshape(self.filter_size, self.filter_size,
                                                                  self.in_channels, self.out_channels)
                grad_slice = d_out_slice.dot(W.T).reshape(batch_size, self.filter_size,
                                                          self.filter_size, self.in_channels)
                result[:, y:y+self.filter_size, x:x +
                       self.filter_size, :] += grad_slice

        return result[:, self.padding:height + self.padding, self.padding:width+self.padding, :]

    def params(self):
        return {'W': self.W, 'B': self.B}


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
        self.X = X
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1

        result = np.zeros(
            (batch_size, out_height, out_width, channels))

        for y in range(out_height):
            for x in range(out_width):
                slice = X[:, y:y + self.pool_size, x:x + self.pool_size, :]
                result[:, y, x] = slice.max(axis=(2, 1))
        return result

    def backward(self, d_out):
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        result = np.zeros_like(self.X)

        for y in range(out_height):
            for x in range(out_width):
                slice = self.X[:, y:y + self.pool_size,
                               x:x + self.pool_size, :]
                max_value = slice.max(axis=(2, 1))
                filter_values = (
                    slice == max_value[:, np.newaxis, np.newaxis, :])
                values = (d_out[:, y, x, :])[:, np.newaxis, np.newaxis, :]
                result[:, y:y + self.pool_size, x:x +
                       self.pool_size, :] += values * filter_values

        return result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
