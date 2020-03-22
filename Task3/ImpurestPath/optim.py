import numpy as np


class SGD:
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate


class MomentumSGD:
    '''
    Implements Momentum SGD update
    '''
    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.v = 0
    
    def update(self, w, d_w, learning_rate):
        '''
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        '''
        self.v = self.momentum * self.v - learning_rate * d_w 
        w = w + self.v
        return w
