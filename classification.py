from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt
import  numpy as np

d = load_breast_cancer(as_frame=True)
data = pd.DataFrame(d.data)

data['target'] = d.target
data['target'] = data['target'].replace(0,-1)
X = np.array(data.iloc[:, :-1])
X = np.insert(X, 0, [1 for _ in range(X.shape[0])], axis=1)
target = np.array(data['target'])
alpha = 0.1
lamda = 0.001
E = 1000
B = 100

w = np.random.rand(X.shape[1])

for _ in range(E):
    for i in range(B, X.shape[0], B):
        X_batch = X[i-B:i]
        Y_batch = target[i-B:i]

        f = np.sign((X_batch@w))
        err = np.maximum(0, 1-Y_batch*f)
        mask = err>0
        w -= 2*alpha*lamda*w - alpha*np.sum(Y_batch[mask][:, None]*X_batch[mask], axis=0)


data_pred = pd.DataFrame({"target":target, "preds":np.sign(X@w)})
data_pred['equal'] = data_pred['target'] == data_pred['preds']
print(data_pred)

