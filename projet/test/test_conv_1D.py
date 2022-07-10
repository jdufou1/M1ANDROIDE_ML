"""
Test Conv1D
"""
from cmath import tanh
import sys
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import numpy as np

import time

from lib.module.Sequential import Sequential

from lib.module.Conv1D import Conv1D

from lib.module.Linear import Linear

from lib.module.Flatten import Flatten

from lib.module.MaxPool1D import MaxPool1D

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
input = len(alltrainx[0])
output = len(np.unique(alltesty))
alltrainy_oneHot = OneHotEncoding(alltrainy)
alltesty_oneHot = OneHotEncoding(alltesty)

# Standardisation
# scaler = StandardScaler()
# alltrainx = scaler.fit_transform(alltrainx)
# alltestx = scaler.fit_transform(alltestx)

alltrainx /= 2
alltestx /= 2


alltrainx = alltrainx.reshape((alltrainx.shape[0],alltrainx.shape[1],1))
alltestx = alltestx.reshape((alltestx.shape[0],alltestx.shape[1],1))

print(alltrainx.shape)

# l1 = Conv1D(5, 1, 6,biais=True)
# l2 = ReLU()
# l3 = MaxPool1D(2, 2)
# l4 = Conv1D(5, 6, 16,biais=True)
# l5 = ReLU()
# l6 = MaxPool1D(2, 2)
# l7 = Conv1D(5, 16, 120,biais=True)
# l8 = ReLU()
# l9 = Flatten()
# l10 = Linear(6840, 84)
# l11 = TanH()
# l12 = Linear(84, alltrainy_oneHot.shape[1])
# l13 = Softmax()


l1=Conv1D(3,1,64, biais = False) 
l2=MaxPool1D(2,2) 
l3=Flatten()
l4=Linear(8128,100) 
l5=ReLU(0) 
l6=Linear(100,10)


model = Sequential(l1, l2, l3, l4, l5, l6)
loss = CELogSoftmax()

iteration = 10
gradient_step = 1e-3
batch_size = 50

opt = SGD(model, loss, alltrainx, alltrainy_oneHot, batch_size, nbIter=iteration, eps=gradient_step)
    

list_loss = opt.update()

# Prediction
predict = model.forward(alltrainx)
predict = np.argmax(predict, axis=1)

predict_test = model.forward(alltestx)
predict_test = np.argmax(predict_test, axis=1)

print(predict.shape)
print(alltrainy.shape)

print("Precision sur l'ensemble d'entrainement",((np.sum(np.where(predict == alltrainy, 1, 0)) / len(predict))*100),"%")
print("Precision sur l'ensemble de test",((np.sum(np.where(predict_test == alltesty, 1, 0)) / len(predict_test))*100),"%")
    