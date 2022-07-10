"""
Test Sequential
Jérémy DUFOURMANTELLE
"""

import sys
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import numpy as np

import time

from mltools import gen_arti

from lib.module.Sequential import Sequential

from lib.module.Optim import SGD,AdaGrad

from lib.module.Linear import Linear

from lib.module.TanH import TanH

from lib.module.Sigmoide import Sigmoide

from lib.loss.MSEloss import MSEloss

X, y = gen_arti(data_type=2, epsilon=0.001)

if y.ndim == 1 :
    y = y.reshape((-1,1))

# prediction

def prediction(X,model) : 
    return model.forward(X)

## MODEL 1

# Parameters

nbIter = 200
nbNeurons = 6
learning_rate = 0.01
batch_size = 20

linear1 = Linear(X.shape[1],nbNeurons)
activation1 = TanH()
linear2 = Linear(nbNeurons,y.shape[1])
activation2 = Sigmoide()

loss = MSEloss()

train_loss_SGD = []

model1 = Sequential(linear1 , activation1 , linear2 , activation2)

optim = SGD(model1 , loss , X , y ,batch_size=batch_size,nbIter=nbIter, eps=learning_rate)

# model training
start = time.time()

train_loss_SGD = optim.update()

end = time.time()
duration = end - start

pred = prediction(X,model1)
pred = np.where(pred > 0.5 , 1 , -1)
accuracy = (pred == y).mean() * 100




## MODEL 2

learning_rate = 0.01
batch_size = 20
linear1 = Linear(X.shape[1],nbNeurons)
activation1 = TanH()
linear2 = Linear(nbNeurons,y.shape[1])
activation2 = Sigmoide()

model2 = Sequential(linear1 , activation1 , linear2 , activation2)

optim = AdaGrad(model2 , loss , X , y ,batch_size=batch_size,nbIter=nbIter, eps=learning_rate)

train_loss_Adagrad = optim.update()



# Print information

print("----- Test sgd vs adagrad -----")
print("DATA : XOR")
print("number of sample : ",len(X))
print("MODEL : Linear - TanH - Linear - Sigmoide - MSE")
print("Learning rate : ",learning_rate)
print("number of iteration : ",nbIter)
print("number of neurons : ",nbNeurons)
print("shape X : ",X.shape)
print("shape y : ",y.shape)
print("Loss reached : ",train_loss_SGD[-1])
print("Accuracy : ",accuracy,"%")
print("train duration : ",duration,"sec")

# Plot the results

plt.figure()
plt.title("Loss evolution")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.plot(train_loss_SGD,label="SGD")
plt.plot(train_loss_Adagrad,label="Adagrad")
plt.legend()
plt.show()
