import numpy as np
import pandas as pd

csv = pd.read_csv('result.csv')
pazienti, _ = np.unique(np.array(csv['id']), return_counts=True)
labels = []
y_preds = []
for p in pazienti:
    label, c = np.unique(csv[csv['id']==p]['y_t'], return_counts=True)
    label = label[np.argmax(c)]
    y_pred, c = np.unique(csv[csv['id']==p]['y_p'], return_counts=True)
    y_pred = y_pred[np.argmax(c)]
    labels.append(label)
    y_preds.append(y_pred)

print(pazienti[0], labels[0], y_preds[0])
new_csv = pd.DataFrame({'id': pazienti, 'label': labels, 'prediction': y_preds})
new_csv.to_csv('result_per_paziente.csv')