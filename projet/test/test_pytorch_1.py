"""
Test Conv1D
"""
from cmath import tanh
import sys
from unittest import result
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

import time

from lib.module.Sequential import Sequential

from lib.module.Conv2D import Conv2D

from lib.module.Linear import Linear

from lib.module.Flatten import Flatten

from lib.module.MaxPool2D import MaxPool2D

from lib.module.ReLU import ReLU

from lib.module.TanH import TanH

from lib.module.Softmax import Softmax

from lib.loss.CELogSM import CELogSoftmax

from lib.module.Optim import SGD

def OneHotEncoding(y):
    onehot = np.zeros((y.size,y.max()+1))
    onehot[np.arange(y.size),y]=1
    return onehot

def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

uspsdatatrain = "../data/USPS_train.txt"
uspsdatatest = "../data/USPS_test.txt"
alltrainx, alltrainy = load_usps(uspsdatatrain)
alltestx, alltesty = load_usps(uspsdatatest)

# taille couche
output = len(np.unique(alltesty))
alltrainy_oneHot = OneHotEncoding(alltrainy)
alltesty_oneHot = OneHotEncoding(alltesty)

alltrainx /= 2
alltestx /= 2


alltrainx = alltrainx.reshape((alltrainx.shape[0],16,16,1))
alltestx = alltestx.reshape((alltestx.shape[0],16,16,1))

print(alltrainx.shape)

l1=Conv2D(3,1,32, biais = False) 
l2=MaxPool2D(2,2) 
l3=Flatten()
l4=Linear(1568,100) 
l5=ReLU(0) 
l6=Linear(100,10)

model = Sequential(l1, l2, l3, l4, l5, l6)
loss = CELogSoftmax()

iteration = 2
gradient_step = 1e-3
batch_size = 50

opt = SGD(model, loss, alltrainx, alltrainy_oneHot, batch_size, nbIter=iteration, eps=gradient_step)

list_loss_mynetwork = opt.update()

# Prediction
predict = model.forward(alltrainx)
predict = np.argmax(predict, axis=1)

predict_test = model.forward(alltestx)
predict_test = np.argmax(predict_test, axis=1)

print(predict.shape)
print(alltrainy.shape)

print("Precision sur l'ensemble d'entrainement",((np.sum(np.where(predict == alltrainy, 1, 0)) / len(predict))*100),"%")
print("Precision sur l'ensemble de test",((np.sum(np.where(predict_test == alltesty, 1, 0)) / len(predict_test))*100),"%")


"""
TEST PYTORCH
"""

print("Test pytorch")

import torch
import torch.nn as nn
import random

alltrainx_t = torch.as_tensor(alltrainx, dtype=torch.float32).transpose(1,3)


alltrainy_oneHot_t = torch.as_tensor(alltrainy_oneHot, dtype=torch.float32)

class Network(nn.Module) :

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,32,3),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1568,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self,x):
        return self.net(x)


network = Network()

optimizer = torch.optim.SGD(network.parameters(), lr=gradient_step)

n_batches = int(len(alltrainx) / batch_size) 
print("n_batches : ",n_batches)
loss_network_pytorch = list()

for _ in tqdm(range(iteration)) :

    list_loss = list()

    for batch in range(n_batches):

        batch_X, batch_y = alltrainx[batch*batch_size:(batch+1)*batch_size,], alltrainy_oneHot[batch*batch_size:(batch+1)*batch_size,]

        batch_X_t = torch.as_tensor(batch_X, dtype = torch.float32).transpose(1,3)
        batch_y_t = torch.as_tensor(batch_y, dtype = torch.float32)

        loss_function = torch.nn.CrossEntropyLoss()

        y_pred = network(batch_X_t)

        loss = loss_function(y_pred, batch_y_t)

        list_loss.append(loss.detach().numpy())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    
    avg_loss = sum(list_loss) / len(list_loss)
    loss_network_pytorch.append(avg_loss)


# prediction train
alltrainx_t = torch.as_tensor(alltrainx, dtype=torch.float32).transpose(1,3)
alltrainy_oneHot_t = torch.as_tensor(alltrainy_oneHot, dtype=torch.float32)

prediction_train = torch.argmax(network(alltrainx_t), dim=1)
y_train = torch.argmax(alltrainy_oneHot_t, dim=1)
result_train = (torch.sum((prediction_train == y_train))/alltrainx_t.shape[0] * 100).detach().numpy()
print(f"resultat pytorch entrainement : {result_train}")


# prediction test
alltestx_t = torch.as_tensor(alltestx, dtype=torch.float32).transpose(1,3)
alltesty_oneHot_t = torch.as_tensor(alltesty_oneHot, dtype=torch.float32)

prediction_test = torch.argmax(network(alltestx_t), dim=1)
y_test = torch.argmax(alltesty_oneHot_t, dim=1)
result_test = (torch.sum((prediction_test == y_test))/alltestx_t.shape[0] * 100).detach().numpy()
print(f"resultat pytorch test : {result_test}")


plt.figure()
plt.title("Comparaison between my network and pytorch")
plt.plot(list_loss_mynetwork,label="loss my network")
plt.plot(loss_network_pytorch,label="loss pytorch network")
plt.xlabel("iteration")
plt.ylabel("CELogSoftmax")
plt.legend()
plt.show()

