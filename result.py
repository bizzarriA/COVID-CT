import numpy as np
import pandas as pd


csv = pd.read_csv('result.csv')
patients = csv.groupby(['id']).mean()
y_true = []
y_pred = []
patients = []
print(patients)
for row in patients:
    y_pred.append(int(row[1]))
    y_true.append(int(row[2][1]))

y_pred=np.array(y_pred)
y_true=np.array(y_true)
T0 = sum((y_true == 0) & (y_true==y_pred))
T1 = sum((y_true == 1) & (y_true==y_pred))
T2 = sum((y_true == 2) & (y_true==y_pred))
F01 = sum((y_pred == 0) & (y_true==1))
F02 = sum((y_pred == 0) & (y_true==2))
F10 = sum((y_pred == 1) & (y_true==0))
F12 = sum((y_pred == 1) & (y_true==2))
F20 = sum((y_pred == 2) & (y_true==0))
F21 = sum((y_pred == 2) & (y_true==1))
matrice = np.array([[T0, F01, F02],[F10, T1, F12],[F20, F21, T2]])
print(matrice)
print(T0, F01, F02)
print(F10, T1, F12)
print(F20, F21, T2)

