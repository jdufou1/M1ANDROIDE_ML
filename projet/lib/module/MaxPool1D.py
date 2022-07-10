"""
MaxPool1D module
"""

import numpy as np

from lib.module.Module import Module

class MaxPool1D(Module):

    def __init__(self, k_size=3, stride=1):
        self._parameters = None
        self._gradient = None
        self._k_size = k_size
        self._stride = stride

    def zero_grad(self):
        pass

    def forward(self, X):
        size = ((X.shape[1] - self._k_size) // self._stride) + 1
        outPut = np.zeros((X.shape[0], size, X.shape[2]))
        for i in range(0, size, self._stride):
            outPut[:,i,:]=np.max(X[:,i:i+self._k_size,:],axis=1)
        self._forward=outPut
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        size = ((input.shape[1] - self._k_size) // self._stride) + 1
        outPut=np.zeros(input.shape)
        batch=input.shape[0]
        chan_in=input.shape[2]
        for i in range(0,size,self._stride):
            indexes_argmax = np.argmax(input[:, i:i+self._k_size,:], axis=1) + i
            outPut[np.repeat(range(batch),chan_in),indexes_argmax.flatten(),list(range(chan_in))*batch]=delta[:,i,:].reshape(-1)
        self._delta=outPut
        return self._delta