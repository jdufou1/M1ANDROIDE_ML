"""
Test Linear
Jérémy DUFOURMANTELLE
"""

import sys
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import time

from tqdm import tqdm

from lib.module.Linear import Linear

from lib.loss.MSEloss import MSEloss

from sklearn.datasets import make_regression

# Data generation

X, y = make_regression(n_samples=100, n_features=1,bias=1,noise=10,n_targets=1, random_state=0)

if y.ndim == 1 :
    y = y.reshape((-1,1))

# Parameters

nbIter = 50
nbNeurons = 1
learning_rate = 1e-3

# Model

model = Linear(X.shape[1],1)
loss = MSEloss()

train_loss = []

# Model training
start = time.time()
for _ in tqdm(range(nbIter)):
    # forward pass
    y_hat = model.forward(X)

    # Backpropagation
    forward_loss = loss.forward(y,y_hat)
    
    # MSE
    train_loss.append(forward_loss.mean()) 

    backward_loss = loss.backward(y,y_hat)

    model.backward_update_gradient(X,backward_loss)

    # Update
    model.update_parameters(learning_rate)

    model.zero_grad()

end = time.time()
duration = end - start

# Print information

print("----- Test linear 1 -----")
print("DATA : ")
print("noise : ",10)
print("number of sample : ",100)
print("MODEL : ")
print("Learning rate : ",learning_rate)
print("number of iteration : ",nbIter)
print("shape X : ",X.shape)
print("shape y : ",y.shape)
print("Loss reached : ",train_loss[-1])
print("train duration : ",duration,"sec")


# Plot the data

plt.figure()
plt.title("Data")
plt.xlabel("X")
plt.ylabel("y")
plt.scatter(X,y)
plt.legend()
plt.show()


# Plot the results

plt.figure()
plt.title("Loss evolution")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.plot(train_loss)
plt.show()

# Plot the decision border of the linear model

plt.figure()
plt.title("Decision border")
plt.xlabel("X")
plt.ylabel("y")
plt.scatter(X,y)
plt.plot(X , model.forward(X), color="r" , label="decision border")
plt.legend()
plt.show()