from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = load_diabetes(as_frame=True)
diabets = data.data
diabets["target"] = data.target

target = np.array(diabets['target'])
X = np.array(diabets.iloc[:, :-1])
X = np.insert(X, 0, [1 for _ in range(X.shape[0])], axis=1)
alpha = 0.1
w = np.random.rand(11)
B = 20
MSE = []
for _ in range(1000): # Количество эпох 1000
    for i in range(B, X.shape[0]-1, B):
        X_batch = X[i-B:i]
        Y_batch = target[i-B:i]
        
        f = X_batch.dot(w)
        err = f - Y_batch
        w -= 2*alpha*X_batch.T.dot(err)/B
    MSE.append(sum(err**2/B))
    # График MSE от количества эпох строится обязательно для подбора эпох и когда кривая начинает быть близка к горизонтальной смысла нету увеличивать эпохи
    
print(1)
print(len(MSE))
plt.plot([i for i in range(1000)], MSE)
plt.show()
print(X @ w)
np.save("coefBatch", w)
# w = np.load("coefBatch.npy")


dataP = pd.DataFrame({"Ответ": target, "Предсказание": X@w})
print(f"MSE: {sum((dataP["Ответ"] - dataP["Предсказание"])**2/len(dataP['Ответ']))}")
print(dataP)
