import pandas as pd
import os 
import numpy as np

val_df = pd.read_csv('total_val_data.csv')
val_df = val_df.iloc[:, 1:]
val_df = np.array(val_df)
val_df = val_df[:1471]
for img in val_df:
    os.system(f"mv val/{img} ex_val/")