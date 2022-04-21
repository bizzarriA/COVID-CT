import cv2
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv
from utils.preprocessing import auto_body_crop
from utils.gradcam import gradcam_main

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, _ = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    return roc_auc_score(y_test, y_pred, average=average)

def multiclass_precision_recall_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        precision, recall, _ = precision_recall_curve(y_test[:,idx].astype(int), y_pred[:,idx])
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    return pr_display


        


ISIZE = 256
if __name__=='__main__':
    crop = False
    base_path = 'dataset/'
    # _, _, _, test_df = read_csv()
    test_df = pd.read_csv('total_test_data.csv')
    test_df = test_df.iloc[:, 1:]
    print(test_df)
    x_test = []
    y_test = []
    filename = []
    # test_df = test_df[test_df['filename'].str.contains('HUST')]
    # test_df = test_df.sample(frac=1)
    test_df = test_df.sample(n=100)
    test_df = np.array(test_df)
    model_names = os.listdir('model/')
    # for model_name in model_names:
    model_name = "model/model_split_20220406-192229"
        # try: 
    # tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_name)
    print(np.shape(test_df))
    val_ds = pd.read_csv('total_val_data.csv')
    val_ds['filename'] = 'test/' + val_ds['filename']
    val_ds = np.array(val_ds['filename'])
    for row in tqdm(test_df):
        name = row[0]
        try:
            #print("[INFO] immagine utilizzabile: ", name)
            if name not in val_ds:
                if os.path.exists(name):
                    img = cv2.imread(name, 0) 
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    x_test.append(img/255.)
                    y_test.append(row[1])
                    filename.append(name)
                    # gradcam_main(model, img, name.split('/')[-1], row[1])
        except:
            continue
    x_test = np.array(x_test)
    
    y_true = np.array(y_test)
    print(np.shape(x_test))
    print("lettura DS finita")
    model_names = os.listdir('model/')
    model_name = "model/model_split_20220406-123204" 
    model = tf.keras.models.load_model(model_name)
    y_pred = []
    scores = []

    predictions = model.predict(x_test, verbose=1, batch_size=1)
    for prediction in tqdm(predictions):
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
        scores.append(prediction)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    print("[INFO] MODEL NAME: ", model_name)
    test_acc = sum(y_pred == y_true) / len(y_true)
    print("[INFO] normal accuracy: ")
    print(test_acc)
   
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    
    target= ['normal', 'common', 'covid']
    
    # set plot figure size
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    
    print('ROC AUC score:', multiclass_roc_auc_score(y_true, y_pred))

    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    plt.show()
    pr_display = multiclass_precision_recall_score(y_true, y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # roc_display.plot(ax=ax1)
    # pr_display.plot(ax=ax2)
    plt.show()

    # result = pd.DataFrame({'filename': filename, 'y_pred':y_pred, 'y_true':y_true})
    # result.to_csv('result_3_class.csv')

