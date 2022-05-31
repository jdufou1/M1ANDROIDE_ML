"""
Optim : optimizer
DUFOURMANTELLE JEREMY
"""

import numpy as np

from tqdm import tqdm

class Optim :
    def __init__(self,net, loss, eps=1e-3):
        self.net=net
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
        pass_forward=self.net.forward(batch_x)
        loss = self.loss.forward(batch_y,pass_forward).mean()
        backward_loss=self.loss.backward(batch_y,pass_forward)
        self.net.backward_delta(batch_x,backward_loss)
        self.net.backward_update_gradient(batch_x,backward_loss)
        self.net.update_parameters(self.eps)
        self.net.zero_grad()
        return loss

    def update(self):
        """abstract method"""
        pass

class SGD(Optim):
    
    def __init__(self, net, loss,datax,datay,batch_size=20,nbIter=100, eps=1e-3):
        self.net=net
        self.loss=loss
        self.eps=eps
        self.datax=datax
        self.datay=datay
        self.batch_size=batch_size
        self.nbIter=nbIter

    def creation_dataset_minibatch(self):
        size=len(self.datax)
        values = np.arange(size)
        np.random.shuffle(values)
        nb_batch = size // self.batch_size
        if (size % self.batch_size != 0):
            nb_batch += 1
        for i in range(nb_batch):
            index=values[i * self.batch_size:(i + 1) * self.batch_size]
            yield self.datax[index],self.datay[index]

    def update(self):
        list_loss = []
        for _ in tqdm(range(self.nbIter)):
            list_loss_batch = []
            for batch_x,batch_y in self.creation_dataset_minibatch():
                list_loss_batch.append( self.step(batch_x,batch_y) )
            list_loss.append(np.array(list_loss_batch).mean())
        return list_loss