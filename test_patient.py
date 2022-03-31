import cv2
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

from utils.preprocessing import auto_body_crop
from utils.data import read_csv
from utils.gradcam import gradcam_main


ISIZE = 256

def read_scan(path_paziente, crop=False):
    lista_scan = os.listdir('test/')
    lista_scan = np.where(path_paziente in lista_scan)
    print(lista_scan)
    # print("[INFOOOO] ",np.shape(lista_scan))
    scans = []
    lista_scan.sort()
    # paz_name = path_paziente.split('/')
    paz_name = path_paziente
    # print(lista_scan[:5])
    for s in lista_scan:
        try:
            # print("[INFO] immagine utilizzabile: ", name)
            img = cv2.imread(path_paziente + s, 0) 
            if crop:
                img, validate = auto_body_crop(img)
                if validate:
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    scans.append(img)
                    
            else:
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                scans.append(img)
        except:
            continue
            # print("ERRORE ", s)
    # centro = int(len(scans)/2)
    # raggio = int(centro*0.60)
    # new_scans = []
    # for i, s in enumerate(scans[centro-raggio:centro+raggio]):
    #     cv2.imwrite(f"test/{paz_name}_{i}.png", s)
    #     new_scans.append(s/255.)
    return scans
        
if __name__=='__main__':
    crop = False
    base_path = 'dataset/'
    _, _, _, test_df = read_csv()
    # test_df = pd.read_csv('test_data.csv')
    # test_df = test_df.iloc[:, 1:]
    print(test_df)
    x_test = []
    y_test = []
    filename = []
 
    model_name = "model/model_ft_cropped_20220329-122916"
    print("[INFO] MODEL NAME: ", model_name)
    model = tf.keras.models.load_model(model_name)
    # test_df = test_df[test_df['filename'].str.contains('HUST')]
    # test_df = test_df.sample(frac=1)
    # test_df = test_df.sample(n=100)
    test_df = np.array(test_df)
    print(np.shape(test_df))
    scans = []
    y_real = []
    pred_per_paz = []
    # test_df = np.sort(test_df, axis=0)
    start = 0 # 20246
    current = test_df[start,0].split('_')[0][5:]
    y = test_df[start, 1]
    y_real.append(y)
        
    for row in tqdm(test_df[start:]):
        try:
            name = row[0]
            if os.path.exists(name):
                paziente = name.split('_')[0][5:]
                # print(current, paziente)
                if paziente == current:
                    # print(name)
                    img = cv2.imread(name, 0) 
                    # print(img)
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    scans.append(img/255.)
                    # print(np.shape(scans))
                else:
                    ### cambio paziente!
                    # scans = np.expand_dims(scans, axis=0)
                    # print(current, paziente, name)
                    y_pred = []
                    scans = np.array(scans)
                    # print(np.shape(scans))
                    predictions =  model.predict(scans, verbose=0, batch_size=1)
                    for prediction in predictions:
                        classes = np.argmax(prediction)
                        prob = prediction[classes]
                        y_pred.append(classes)
                    y_pred = np.array(y_pred)
                    count_ric = np.bincount(y_pred)
                    y_max = count_ric.argmax()
                    # if len(count_ric)==3 and count_ric[2]>len(scans)*0.1:
                    #     y_max = 2
                    # elif len(count_ric)>=2 and count_ric[1]>len(scans)*0.3:
                    #     y_max = 1
                    # else:
                    #     y_max = count_ric.argmax()
                    pred_per_paz.append(y_max)
                    
                    current = row[0].split('_')[0][5:]
                    y = row[1]
                    y_real.append(y)
                    scans = []
                    img = cv2.imread(name, 0) 
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    # gradcam_main(model, img, current, y)
                    scans.append(img)
                
        except:
            # print("ERRORE img:", name)
            continue
            
    print(current, paziente)
    y_pred = []
    scans = np.array(scans)
    print(np.shape(scans))
    
    for scan in scans:
        scan = np.expand_dims(scan, axis=0)
        prediction = model.predict(scan, verbose=0)
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
    y_pred = np.array(y_pred)
    count_ric = np.bincount(y_pred)
    y_max = count_ric.argmax()
    pred_per_paz.append(y_max)
    
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

