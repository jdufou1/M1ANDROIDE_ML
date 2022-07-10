import numpy as np

from lib.module.Module import Module

class ReLU(Module):

    def __init__(self , threshold = 0):
        self._parameters = None
        self._gradient = None
        self._threshold = threshold

    def zero_grad(self):
        pass

    def forward(self, X):
        self._forward =  np.where(X>self._threshold,X,0.)
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        pass

    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        derive = (input > self._threshold).astype(float)
        self._delta= delta * derive
        return self._delta