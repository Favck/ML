from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = datasets.load_diabetes()
table = pd.DataFrame(data.data, columns=data.feature_names) # Тут нету таргета внимательно!
table['target'] = data.target

X = np.array(table.iloc[:,:-1])

target = np.array(table['target'])

tolerance = 10**(-6)
alpha = 0.1

w = np.array(np.random.rand(11)).T
w_new = np.array([0,0,0,0,0,0,0,0,0,0,0]).T

def grad(err):
    return (2/len(target))*(X.T@err)

X = np.insert(X, 0, [1 for _ in range(442)], axis=1)

t=0
while np.linalg.norm(w_new-w) > tolerance:
    f = X.dot(w) # Предсказание
    err = f - target  # Ошибка
    if t>0:
        w=w_new
    w_new = w - alpha*grad(err)
    t+=1

print(t)
w_new = np.load("weithAnal.npy")
print(w_new)
np.save("weithAnal", w_new)

w_new = np.load("coefAnalytic.npy")

table2 = pd.DataFrame({"Ответ": target, "Предсказание": X@w_new.T})
table2["Ошибка"] = abs(table2["Ответ"] - table2["Предсказание"])
print(f"MSE:{np.sum((target - X@w_new.T)@(target - X@w_new.T))/len(target)}")
table2.to_csv("tableAnalgrad")

plt.scatter(table2["Ответ"], table2["Ответ"], color="red")
plt.scatter(table2["Ответ"], table2["Предсказание"], color='blue')
plt.grid(True)

plt.show()
