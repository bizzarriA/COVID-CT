import pandas as pd
import os 
import numpy as np

val_df = pd.read_csv('total_val_data.csv')
val_df = val_df.iloc[:, 1:]
val_df = val_df[val_df['label']==2]
n_pat = 0
current = 'nessuno'
for _, row in val_df.iterrows():
    name = row[0].split('_')[0]
    if name != current:
        n_pat+=1
        current = name
        print(current)
        if n_pat==20:
            break
    os.system(f"mv \"val/{row[0]}\" ex_val/")
