import numpy as np

from lib.loss.Loss import Loss

class CELogSoftmax(Loss):

    def forward(self, y, yhat,eps = 1e-3):
        """Calculer le cout en fonction de deux entrees"""
        return np.log(np.sum(np.exp(yhat), axis=1) + eps) - np.sum(y * yhat,axis = 1)

    def backward(self, y, yhat):
        """calcul le gradient du cout par rapport yhat"""
        exp = np.exp(yhat)
        return exp / np.sum(exp, axis=1).reshape((-1,1)) - y  