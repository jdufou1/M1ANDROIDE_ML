"""
Test multiclasse
Jérémy DUFOURMANTELLE
"""

import sys
sys.path.insert(1, '../')

import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import confusion_matrix

import seaborn as sns

import time

from tqdm import tqdm

from lib.module.Linear import Linear

from lib.module.Sequential import Sequential

from lib.module.Optim import Optim , SGD

from lib.module.TanH import TanH

from lib.module.Softmax import Softmax

from lib.module.Sigmoide import Sigmoide

from lib.loss.MSEloss import MSEloss

from lib.loss.CEloss import CEloss

from sklearn.datasets import make_classification



def onehot(y):
    onehot = np.zeros((y.size, y.max() + 1))
    onehot[np.arange(y.size), y] = 1
    return onehot

# Load digits from USPS database

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

# hot encoding for multiclasse purpose

y = onehot(alltrainy)

# Model

linear1 = Linear(alltrainx.shape[1], 256)
activation1 = TanH()
linear2 = Linear(256, 128)
activation2 = TanH()
linear3 = Linear(128, y.shape[1])
activation3 = Softmax()
loss = CEloss()

model = Sequential(Linear(alltrainx.shape[1], 4) , TanH(), Linear(4, y.shape[1]), Softmax())

# Hyperparameters

maxIter = 500
eps = 1e-2
batch_size = 10

# Training

optimizer = SGD(model, loss, alltrainx, y, batch_size=batch_size, eps=eps, nbIter=maxIter)
list_loss = optimizer.update()

# Results

taux_train = ((np.argmax( optimizer.net.forward(alltrainx),axis = 1) == alltrainy).mean()*100)
taux_test = ((np.argmax( optimizer.net.forward(alltestx),axis = 1) == alltesty).mean()*100)
print("Taux de bonne classification en train : ",taux_train,"%")
print("Taux de bonne classification en test : ",taux_test,"%")

# Plot

plt.figure()
plt.xlabel("iterations")
plt.ylabel("CELoss")
plt.title("Loss evolution with USPS digits")
plt.plot(list_loss,label="Erreur")
plt.legend()
plt.show()


predict = model.forward(alltrainx)
predict = np.argmax(predict, axis=1)


"""
MATRICE DE CONFUSION TRAIN
"""

plt.figure()
confusion = confusion_matrix(predict, alltrainy)



ax = sns.heatmap(confusion, annot=True, cmap='Blues')

ax.set_title(f"Confusion matrix for USPS Train \ acc = {taux_train}%\n\n")
ax.set_xlabel('\ndigit predicted')
ax.set_ylabel('true digit ')

ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))

plt.show()

"""
MATRICE DE CONFUSION TEST
"""


predict = model.forward(alltestx)
predict = np.argmax(predict, axis=1)

plt.figure()
confusion = confusion_matrix(predict, alltesty)

ax = sns.heatmap(confusion, annot=True, cmap='Blues')

ax.set_title(f"Confusion matrix for USPS Test \ acc = {taux_test}%\n\n")
ax.set_xlabel('\ndigit predicted')
ax.set_ylabel('true digit ')

ax.xaxis.set_ticklabels(np.arange(10))
ax.yaxis.set_ticklabels(np.arange(10))

plt.show()