from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = datasets.load_diabetes()
table = pd.DataFrame(data.data, columns=data.feature_names) # Тут нету таргета внимательно!
table['target'] = data.target


one_column = np.array([1 for _ in range(442)])

target = np.array(table["target"])

X = np.array(table.iloc[:, :-1], dtype=float) 
X = np.insert(X, 0, one_column, axis=1)
w = np.array((np.linalg.inv((X.T @ X))) @ (X.T @ target))
np.savetxt("weith.txt", w)

table2 = pd.DataFrame({"Ответ":target, "Предсказания": X @ w})
table2["Квадрат разности"] = ((table2["Ответ"] - table2["Предсказания"])**2)
table2.to_csv("table")
table2 = pd.read_csv("table", index_col=0)
w = np.load("weith.txt.npy")
print(w)
print(table2.head())
plt.scatter(table2["Ответ"], table2["Предсказания"])
plt.scatter(table2["Ответ"], table2["Ответ"], color="red")

plt.legend()
plt.grid(True)
plt.show()
print(["w0"] + table.columns.tolist()[:-1])
plt.bar([i for i in range(1, len(w)+1)],w)
plt.xticks([i for i in range(1, len(w)+1)], ["w0"] + table.columns.tolist())

plt.show()
