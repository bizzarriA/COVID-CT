import cv2
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.preprocessing import auto_body_crop
from utils.data import read_csv
from utils.gradcam import gradcam_main


ISIZE = 256

if __name__=='__main__':
    crop = False
    base_path = 'dataset/'
    # _, _, _, test_df = read_csv()
    test_df = pd.read_csv('total_test_data.csv')
    # test_df = test_df.loc[test_df['filename'].str.contains('Patient 124_')]
    test_df = test_df.iloc[:, 1:]
    print(test_df)
    x_test = []
    y_test = []
    filename = []
 
    model_name = "model/model_256_total_20220331-171249"
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    optimizer = tf.keras.optimizers.Adam(0.001) 
    # print("[INFO] Model compile")
    # model.compile(
    #     loss="categorical_crossentropy",
    #     optimizer=optimizer,
    #     metrics=['acc'],
    # )
    model.summary()
    # test_df = test_df[test_df['filename'].str.contains('HUST')]
    # test_df = test_df.sample(frac=1)
    # test_df = test_df.sample(n=10)
    test_df = np.array(test_df)
    #test_df = test_df[:1000]
    print(np.shape(test_df))
    scans = []
    y_real = []
    pred_per_paz = []
    count_ric_per_paz = []
    paz_name = []
    # test_df = np.sort(test_df, axis=0)
    start = 0 # 20246
    current = test_df[start,0].split('_')[0][5:]
    y = test_df[start, 1]
    y_real.append(y)
    paz_name.append(current)
        
    for row in tqdm(test_df[start:]):
        # try:
            name = row[0]
            if os.path.exists(name):
                # print(name)
                paziente = name.split('_')[0][5:]
                # print(current, paziente)
                if paziente == current:
                    # print(name)
                    img = cv2.imread(name, 0) 
                    # print(img)
                    img = cv2.resize(img, (ISIZE, ISIZE)) / 255.
                    img = np.expand_dims(img, axis=-1)
                    scans.append(img)
                    # gradcam_main(model, img, name.split('/')[-1], y)
                    # print(np.shape(scans))
                else:
                    ### cambio paziente!
                    # scans = np.expand_dims(scans, axis=0)
                    # print(current, paziente, name)
                    y_pred = []
                    scans = np.array(scans)
                    # print(np.shape(scans))
                    predictions =  model.predict(scans, verbose=0, batch_size=1)
                    # print(np.shape(predictions))
                    for prediction in predictions:
                        classes = np.argmax(prediction)
                        prob = prediction[classes]
                        y_pred.append(classes)
                    y_pred = np.array(y_pred)
                    count_ric = np.bincount(y_pred)
                    y_max = count_ric.argmax()
                    # if len(count_ric)>=2 and count_ric[1]>len(scans)*0.2:
                    #     y_max = 1
                    # elif len(count_ric)==3 and count_ric[2]>len(scans)*0.3:
                    #     y_max = 2
                    # else:
                    #     y_max = count_ric.argmax()
                    pred_per_paz.append(y_max)
                    count_ric_per_paz.append(count_ric)
                    
                    
                    current = row[0].split('_')[0][5:]
                    paz_name.append(current)
                    y = row[1]
                    y_real.append(y)
                    scans = []
                    img = cv2.imread(name, 0) 
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)/255.
                    # gradcam_main(model, img, name.split('/')[-1], y)
                    scans.append(img)
                
        # except:
        #     # print("ERRORE img:", name)
        #     continue
            
    print(current, paziente)
    y_pred = []
    scans = np.array(scans)
    print(np.shape(scans))
    
    predictions =  model.predict(scans, verbose=0, batch_size=1)
    # print(np.shape(predictions))
    for prediction in predictions:
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
    y_pred = np.array(y_pred)
    count_ric = np.bincount(y_pred)
    y_max = count_ric.argmax()
    print("prediction:", y_pred)
    # if len(count_ric)==3 and count_ric[2]>len(scans)*0.1:
    #     y_max = 2
    # elif len(count_ric)>=2 and count_ric[1]>len(scans)*0.3:
    #     y_max = 1
    # else:
    #     y_max = count_ric.argmax()
    pred_per_paz.append(y_max)
    count_ric_per_paz.append(count_ric)
    
        # except:
        #     continue
        
    print(np.shape(y_real), np.shape(pred_per_paz))
    y_pred = np.array(pred_per_paz)
    print(y_pred)
    y_true = np.array(y_real)
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    print(np.shape(paz_name), np.shape(y_real), np.shape(pred_per_paz), np.shape(count_ric_per_paz))
    pd.DataFrame({'filename': paz_name, 'real': y_real, 'pred': pred_per_paz, 'count': count_ric_per_paz}).to_csv('ris.csv')

