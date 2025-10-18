import pandas as pd
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def svw_train(X, y,C1, C2, E=1000, alpha=0.1, B=10, lamda = 0.001):

    w = np.random.rand(X.shape[1])
    y = np.where(y == C1, 1, -1)

    for _ in range(E):
        for i in range(B, X.shape[0], B):
            X_batch = X[i-B:i]
            Y_batch = y[i-B:i]

            f = np.sign(X_batch@w)
            err = np.maximum(0, 1-Y_batch*f)
            mask = err>0
            w -= 2*lamda*alpha*w - alpha*np.sum(Y_batch[mask][:, None]*X_batch[mask], axis=0)
    return w

def predict(X, w, C1, C2):
    pred = np.sign(X@w)
    return C1 if pred>=0 else C2

def all_vs_all_predict(x, classificator : dict):
    votes = {c:0 for c in [0, 1, 2]}
    for (c1,c2), w in classificator.items():
        pred = predict(x, w, c1, c2)
        votes[pred] += 1
    return max(votes, key=votes.get)

d = load_iris()
data = pd.DataFrame(d.data)
data['target'] = d.target

# plt.scatter(data[2], data[3], c=data['target'])    # see classes

X = np.array(data.iloc[:, :-1])
X = np.insert(X, 0, [1 for _ in range(X.shape[0])], axis=1)
target = np.array(data['target'])
w = np.random.rand(3, X.shape[1])

classes = np.unique(target)

classifiers = dict()
pairs = list(it.combinations(classes, 2))

for C1, C2 in pairs:
    mask = (target == C1) | (target==C2)
    X_pair = X[mask]
    Y_pair = target[mask]

    classifiers[(C1, C2)] = svw_train(X_pair,Y_pair, C1, C2)




y_pred = np.array([all_vs_all_predict(xi, classifiers) for xi in X])
print(y_pred)
data_final = pd.DataFrame({"target":target, "predict":y_pred})
print(data_final)
print(sum(data_final['target']==data_final['predict']), len(data['target']))

