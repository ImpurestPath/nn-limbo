import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
)


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """

    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        self.layers = [
            ConvolutionalLayer(input_shape[2], conv1_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            ConvolutionalLayer(conv1_channels, conv2_channels, 3, 1),
            ReLULayer(),
            MaxPoolingLayer(4, 4),
            Flattener(),
            FullyConnectedLayer(4 * conv2_channels, n_output_classes)
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        for layer in self.layers:
            X = layer.forward(X)

        loss, grad = softmax_with_cross_entropy(X, y)
        for i in range(len(self.layers)):
            grad = self.layers[len(self.layers) - 1 - i].backward(grad)
        return loss

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for layer in self.layers:
            X = layer.forward(X)
        pred = X.argmax(axis=1)

        return pred

    def params(self):
        result = {
            'C1W': self.layers[0].params()['W'],
            'C1B': self.layers[0].params()['B'],
            'C2W': self.layers[3].params()['W'],
            'C2B': self.layers[3].params()['B'],
            'FW': self.layers[7].params()['B'],
            'FB': self.layers[7].params()['B'],
        }

        return result
