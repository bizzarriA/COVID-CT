import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv
from utils.preprocessing import auto_body_crop

ISIZE = 256
if __name__=='__main__':
    crop = True
    base_path = 'dataset/'
    _, _, _, test_df = read_csv()
    # test_df = pd.read_csv('test_data.csv')
    # test_df = test_df.iloc[:, 1:]
    # print(test_df)
    x_test = []
    y_test = []
    filename = []
    # test_df = test_df[test_df['filename'].str.contains('HUST')]
    # test_df = test_df.sample(frac=1)
    # test_df = test_df.sample(n=1000)
    test_df = np.array(test_df)
    print(np.shape(test_df))
    for row in tqdm(test_df):
        name = row[0]
        try:
            # print("[INFO] immagine utilizzabile: ", name)
            img = cv2.imread(name, 0) 
            if crop:
                img, validate = auto_body_crop(img)
                if validate:
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    x_test.append(img / 255.)
                    y_test.append(row[1])
            else:
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                x_test.append(img / 255.)
                y_test.append(row[1])
        except:
            continue
    x_test = np.array(x_test)
    y_true = np.array(y_test)
    print(np.shape(x_test))
    print("lettura DS finita")
    #for model_name in model_names:
    model_name = "model/model_3class_Adam_20220322-105724"
    #try: 
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    # model.summary()    
    y_pred = []
    scores = []
    predictions = model.predict(x_test, verbose=1, batch_size=1)
    for prediction in predictions:        
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
        scores.append(prediction)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    print(np.shape(y_pred), np.shape(y_true))
    print(y_true[:3], y_pred[:3])
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    # result = pd.DataFrame({'id': filename, 'y_pred':y_pred, 'y_true':y_true})
    # result.to_csv('result_3_class.csv')
    # except:
    #         print("[ERRORE] modello: ", model_name)
