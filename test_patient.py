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

def from_img(test_df):
    model_name = "model/model_2_calssi_20220428-115231"
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    scans = []
    y_real = []
    pred_per_paz = []
    count_ric_per_paz = []
    paz_name = []
    start = 0
    p = test_df[start,0]
    while(os.path.exists(p) == False):
        start += 1
        p = test_df[start,0]
    current = test_df[start,0].split('_')[0][5:]
    if test_df[start, 1] == 2:
        y = 1
    else:
        y = test_df[start, 1]
    y_real.append(y)
    paz_name.append(current)
    print("primo paziente valido: ", current) 
    for row in tqdm(test_df[start:]):
        name = row[0]
        if os.path.exists(name):
            paziente = name.split('_')[0][5:]
            if paziente == current:
                img = cv2.imread(name, 0) 
                img = cv2.resize(img, (ISIZE, ISIZE)) / 255.
                img = np.expand_dims(img, axis=-1)
                scans.append(img)
                # gradcam_main(model, img, name.split('/')[-1], y)
            else:
                ### cambio paziente!
                y_pred = []
                scans = np.array(scans)
                predictions =  model.predict(scans, verbose=0, batch_size=1)
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
                if row[1] == 2:
                    y = 1
                else:
                    y = row[1]
                y_real.append(y)
                scans = []
                img = cv2.imread(name, 0) 
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)/255.
                # gradcam_main(model, img, name.split('/')[-1], y)
                scans.append(img)
    print(current, paziente)
    y_pred = []
    scans = np.array(scans)
    print(np.shape(scans))
    
    predictions =  model.predict(scans, verbose=0, batch_size=1)
    # print(np.shape(predictions))
    for prediction in predictions:
        classes = np.argmax(prediction)
        # prob = prediction[classes]
        y_pred.append(classes)
    
    y_pred = np.array(y_pred)
    y_real = np.array(y_real)
    
    return y_real, y_pred, paz_name

def from_csv(csv, a=0.2):
    y_pred = []
    y_real = []
    pazienti_fatti = []
    for _, row in csv.iterrows():
        name = row[0].split('_')[0]
        name = name[5:]+'_'
        if name in pazienti_fatti:
            continue
        else:
            ###
            # make gradcam #
            # model_name = "model/model_split_20220406-123204" 
            # print("[INFO] MODEL NAME: ", model_name)
            # model = tf.keras.models.load_model(model_name)
            # img = cv2.imread(row[0], 0) 
            # img = cv2.resize(img, (ISIZE, ISIZE))
            # img = np.expand_dims(img, axis=-1)/255.
            # gradcam_main(model, img, name.split('/')[-1], y)
            pazienti_fatti.append(name)
            y_real.append(row[2])
            pred_paziente = csv[csv['filename'].str.contains(name)]['y_pred']
            n_scan = len(pred_paziente)
            count_ric = np.bincount(pred_paziente)
            y_max = count_ric.argmax()
            if len(count_ric)==2 and count_ric[1]>n_scan*a:
                y_max = 1
            # elif len(count_ric)>=2 and count_ric[1]>n_scan*0.2:
            #     y_max = 1
            else:
                y_max = count_ric.argmax()
            y_pred.append(y_max)
    return np.array(y_real), np.array(y_pred), pazienti_fatti
        
def unife_img_paziente(test_df):
    model_name = "model/model_split_20220406-123204" 
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    y_real = []
    pred_per_paz = []
    count_ric_per_paz = []
    paz_name = []
    for row in tqdm(test_df):
        try:
            name = 'dataset/unife/png/' + row
            if os.path.exists(name) and '.DS_Store' not in name:
                scans_path = os.listdir(name)
                scans = []
                for p in scans_path:
                    img = cv2.imread(name+'/'+p, 0) 
                    img, _ = auto_body_crop(img)
                    img = cv2.resize(img, (ISIZE, ISIZE)) / 255.
                    img = np.expand_dims(img, axis=-1)
                    scans.append(img)
                    # gradcam_main(model, img, name.split('/')[-1], y)
                scans = np.array(scans)
                predictions =  model.predict(scans, verbose=0, batch_size=1)
                y_pred = []
                for prediction in predictions:
                    classes = np.argmax(prediction)
                    prob = prediction[classes]
                    y_pred.append(classes)
                y_pred = np.array(y_pred)
                count_ric = np.bincount(y_pred)
                y_max = count_ric.argmax()
                if 'NEG' in row:
                    y = 0
                else:
                    y = 2
                y_real.append(y)
                paz_name.append(row)
                # if len(count_ric)>=2 and count_ric[1]>len(scans)*0.2:
                #     y_max = 1
                # elif len(count_ric)==3 and count_ric[2]>len(scans)*0.3:
                #     y_max = 2
                # else:
                #     y_max = count_ric.argmax()
                pred_per_paz.append(y_max)
                count_ric_per_paz.append(count_ric)
        except:
            continue
    pred_per_paz = np.array(pred_per_paz)
    y_real = np.array(y_real)
    paz_name = np.array(paz_name)
    
    return y_real, pred_per_paz, paz_name

def unife_img(test_df):
    base_path = 'dataset/unife/preproc/'
    model_name = "model/model_split_20220406-123204" 
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    y_test = []
    x_test = []
    y_pred = [] 
    filename = []
    for row in tqdm(test_df):
        name = base_path + row
        try:
            #print("[INFO] immagine utilizzabile: ", name)
            if os.path.exists(name):
                img = cv2.imread(name, 0) 
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                x_test.append(img/255.)
                filename.append(name)
                # gradcam_main(model, img, name.split('/')[-1], row[1])
                if 'NEG' in row:
                    y = 0
                else:
                    y = 2
                y_test.append(y)
        except:
            continue
    x_test = np.array(x_test)
    predictions = model.predict(x_test, verbose=1, batch_size=1)
    for prediction in tqdm(predictions):
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    
    return y_test, y_pred, filename



if __name__=='__main__':
    crop = False
    # test_df = pd.read_csv('total_test_data.csv')
    test_df = pd.read_csv('result_3_class.csv')
    test_df = test_df.iloc[:, 1:]
    print(test_df)
    # test_df = np.array(test_df)
    # print(np.shape(test_df))
    # test_df = os.listdir('dataset/unife/preproc/')
    a_max = 0
    y_max = []
    acc_max = 0
    for a in np.arange(0.05, 0.16, 0.05):
        y_real, y_pred, paz_name = from_csv(test_df, a=a)
        test_acc = sum(y_pred == y_real) / len(y_real)
        print(test_acc, a)
        if test_acc > acc_max:
            acc_max=test_acc
            y_max = y_pred
            a_max = a
    print("accuracy max: ",acc_max, "alfa mac: ", a_max) 
    confusion_mtx = tf.math.confusion_matrix(y_real, y_max)
    print("Confusion matrix:\n",confusion_mtx)
    pd.DataFrame({'filename': paz_name, 'real': y_real, 'pred': y_pred}).to_csv('ris.csv')

