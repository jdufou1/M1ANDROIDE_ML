"""
Test Activation and Binary classification
Jérémy DUFOURMANTELLE
"""

import sys
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import numpy as np

import time

from tqdm import tqdm

from mltools import plot_data, plot_frontiere, make_grid, gen_arti

from lib.module.Linear import Linear

from lib.module.TanH import TanH

from lib.module.Sigmoide import Sigmoide

from lib.loss.MSEloss import MSEloss

from sklearn.datasets import make_classification


X, y = gen_arti(data_type=2, epsilon=0.001)

if y.ndim == 1 :
    y = y.reshape((-1,1))

# Parameters

nbIter = 20000
nbNeurons = 64
learning_rate = 1e-3

linear1 = Linear(X.shape[1],nbNeurons)
activation1 = TanH()
linear2 = Linear(nbNeurons,y.shape[1])
activation2 = Sigmoide()
loss = MSEloss()

train_loss = []

# model training
start = time.time()
for _ in tqdm(range(nbIter)):
            
    # forward
    res1 = linear1.forward(X)
    res2 = activation1.forward(res1)
    res3 = linear2.forward(res2)
    res4 = activation2.forward(res3)

    train_loss.append(np.mean(loss.forward(y, res4))) # loss

    # backward

    last_delta = loss.backward(y, res4)
    delta_sig = activation2.backward_delta(res3, last_delta)
    delta_lin = linear2.backward_delta(res2, delta_sig)
    delta_tan = activation1.backward_delta(res1, delta_lin)

    # update gradient

    linear1.backward_update_gradient(X, delta_tan)
    linear2.backward_update_gradient(res2, delta_sig)

    # upadte parameters
    linear1.update_parameters(learning_rate)
    linear2.update_parameters(learning_rate)

    # reinitialisation parameters
    linear1.zero_grad()
    linear2.zero_grad()

end = time.time()
duration = end - start
# prediction

def prediction(X) : 
    res1 = linear1.forward(X)
    res2 = activation1.forward(res1)
    res3 = linear2.forward(res2)
    return activation2.forward(res3)

pred = prediction(X)
pred = np.where(pred > 0.5 , 1 , -1)
accuracy = (pred == y).mean() * 100

# Print information

print("----- Test activation 5 -----")
print("DATA : Chessboard")
print("number of sample : ",len(X))
print("MODEL : Linear - TanH - Linear - Sigmoide - MSE")
print("Learning rate : ",learning_rate)
print("number of iteration : ",nbIter)
print("number of neurons : ",nbNeurons)
print("shape X : ",X.shape)
print("shape y : ",y.shape)
print("Loss reached : ",train_loss[-1])
print("Accuracy : ",accuracy,"%")
print("train duration : ",duration,"sec")

# Plot the results

plt.figure()
plt.title("Loss evolution")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.plot(train_loss)
plt.show()

# Plot decision surface

plt.figure()
plt.title(f"Decision surface | acc = {accuracy}%")
plt.xlabel("x1")
plt.ylabel("x2")


# define bounds of the domain
min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 0.1)
x2grid = np.arange(min2, max2, 0.1)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))

yhat = prediction(grid)
yhat = np.where(yhat > 0.5 , 1  , 0)

zz = yhat.reshape(xx.shape)
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='Paired')
plt.scatter(X[:,0],X[:,1],c=y)

plt.show()