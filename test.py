import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv
from utils.preprocessing import auto_body_crop
from utils.gradcam import gradcam_main

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
    test_df = test_df.sample(frac=1)
    # test_df = test_df.sample(n=10000)
    test_df = np.array(test_df)
    model_names = os.listdir('model/')
    # for model_name in model_names:
    model_name = "model/model_split_20220406-192229"
        # try: 
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_name)
    print(np.shape(test_df))
    val_ds = pd.read_csv('total_val_data.csv')
    val_ds['filename'] = 'test/' + val_ds['filename']
    val_ds = np.array(val_ds['filename'])
    print(val_ds[:10])
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
            # else:
            #     print("immagine in validazione: ", name)
        except:
            continue
    # print((x_test))
    x_test = np.array(x_test)
    
    # x_normal = x_test / 255.
    y_true = np.array(y_test)
    print(np.shape(x_test))
    print("lettura DS finita")
    model_names = os.listdir('model/')
    # for model_name in model_names:
    model_name = "model/model_split_20220406-123204"
        # try: 
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_name)
    y_pred = []
    scores = []
    
    # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # y_test = np.array(tf.keras.utils.to_categorical(y_test, 3))
    # result = model.evaluate(x_test, y_test)
    # print(dict(zip(model.metrics_names, result)))
    predictions = model.predict(x_test, verbose=1, batch_size=1)
    for prediction in tqdm(predictions):
        # print(np.shape(x))
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
    result = pd.DataFrame({'filename': filename, 'y_pred':y_pred, 'y_true':y_true})
    result.to_csv('result_3_class.csv')
    # except:
    #         print("[ERRORE] modello: ", model_name)
