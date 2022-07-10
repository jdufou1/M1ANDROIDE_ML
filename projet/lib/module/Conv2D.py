"""
Conv2D module
"""

import numpy as np

from lib.module.Module import Module

class Conv2D(Module):

    def __init__(self,k_size, chan_in, chan_out , stride = 1, biais = True):
        super().__init__()
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self._biais = biais
        facteur=1 / np.sqrt(chan_in*k_size)
        self._parameters = np.random.uniform(-facteur,facteur,(k_size,k_size , chan_in , chan_out))
        self._grad = np.zeros(self._parameters.shape)
        if biais : 
            self._grad_biais = np.random.uniform(-facteur, facteur, chan_out)
            self._biais_value = np.zeros((chan_out))

    def zero_grad(self):
        self._grad = np.zeros(self._parameters.shape)
        if (self._biais):
            self._grad_biais = np.zeros(self._grad_biais.shape)

    def forward(self, X):
        """
        X : batch * width * height * c_in
        params : kernel_size * c_in * c_out
        output : batch * length_width * length_height * c_out
        """
        length_width = int(np.floor((X.shape[1]-self._k_size)/self._stride) + 1)
        length_height = int(np.floor((X.shape[2]-self._k_size)/self._stride) + 1)
        output = np.zeros((X.shape[0] , length_height , length_width  , self._chan_out))
        for i in range(0, X.shape[2] - self._k_size + 1,self._stride) :
            for j in range(0,X.shape[1] - self._k_size + 1,self._stride) :
                output[: , i , j , :] = X[: , i : i + self._k_size , j : j + self._k_size , : ].reshape(X.shape[0],-1) @ self._parameters.reshape(-1,self._parameters.shape[3])
        if (self._biais):
            output += self._biais_value
        self._forward = output
        return self._forward

    def update_parameters(self, gradient_step=1e-3):
        self._parameters -= gradient_step*self._grad
        if self._biais : 
            self._biais_value -= gradient_step*self._grad_biais

    def backward_update_gradient(self, input, delta):
        length_width = int(np.floor((input.shape[1]-self._k_size)/self._stride) + 1)
        length_height = int(np.floor((input.shape[2]-self._k_size)/self._stride) + 1)
        for i in range(0, input.shape[2] - self._k_size + 1,self._stride) :
            for j in range(0,input.shape[1] - self._k_size + 1,self._stride) :
                self._grad += (input.transpose(3,1,2,0)[ : , i : i + self._k_size , j : j + self._k_size , :].reshape(-1 , input.shape[0]) @ delta[: , i , j , :]).reshape(self._parameters.shape)
        self._grad /= delta.shape[0]
        if self._biais:
            self._grad_biais=delta.mean((0,1,2))

    def backward_delta(self, input, delta):
        length_width = int(np.floor((input.shape[1]-self._k_size)/self._stride) + 1)
        length_height = int(np.floor((input.shape[2]-self._k_size)/self._stride) + 1)
        output = np.zeros(input.shape)
        for i in range(0, input.shape[2] - self._k_size + 1,self._stride) :
            for j in range(0,input.shape[1] - self._k_size + 1,self._stride) :    
                output[: , i : i + self._k_size, j : j + self._k_size, :] = (delta[: , i , j , :] @ self._parameters.transpose(3,0,1,2).reshape(self._parameters.shape[3],-1)).reshape(input.shape[0],self._k_size,self._k_size,input.shape[3])
        self._delta=output
        return self._delta