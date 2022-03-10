import numpy as np
import pandas as pd
from tqdm import tqdm

def format_csv(csv):
    names = []
    labels = []
    y_preds = []
    for _, row in csv.iterrows():
        name = row['id']
        if name[0] == 'N':
            names.append(name[10:22])
        else:
            names.append(name[:6])
        labels.append(row['y_true'])
        y_preds.append(row['y_pred'])
    return pd.DataFrame({'id':names, 'y_t': labels, 'y_p':y_preds})
            
csv = pd.read_csv('result.csv')
new_df = format_csv(csv)
pazienti, _ = np.unique(np.array(new_df['id']), return_counts=True)
labels = []
y_preds = []
for p in tqdm(pazienti):
    label, c = np.unique(new_df[new_df['id']==p]['y_t'], return_counts=True)
    label = label[np.argmax(c)]
    y_pred, c = np.unique(new_df[new_df['id']==p]['y_p'], return_counts=True)
    y_pred = y_pred[np.argmax(c)]
    labels.append(label)
    y_preds.append(y_pred)

print(pazienti[0], labels[0], y_preds[0])
new_csv = pd.DataFrame({'id': pazienti, 'label': labels, 'prediction': y_preds})
new_csv.to_csv('result_per_paziente.csv')
