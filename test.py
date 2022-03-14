import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv

if __name__=='__main__':
    current_path = '' #'/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    base_path = current_path + 'dataset/'
    _, test_df, _ = read_csv(base_path)
    # csv = pd.read_csv(base_path + 'unife.csv')
    # test_df = csv[csv['split']=='test']
    # test_df = test_df[test_df['label']!=0]
    x_test = []
    y_test = []
    filename = []
    # test_df = test_df.sample(frac=1)
    print(test_df.values.tolist()[0])
    test_df = np.array(test_df)
    test_df = [row for row in test_df if "HUST" in row[0]]
    print(test_df[-5:])
    n =  len(test_df)
    print(n)
        # n = 10
    for row in tqdm(test_df):
        name = row[0]
        if "HUST-Patient" in name:
            try:
                # print("[INFO] immagine utilizzabile: ", name)
                img = cv2.imread(name, 0) / 255
                bbox = ([int(row[2]), int(row[3]), int(row[4]), int(row[5])])
                img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                img = cv2.resize(img, (256, 256))
                img = np.expand_dims(img, axis=-1)
                x_test.append(img)
                y_test.append(row[1])
                filename.append(name)
            except:
                print("ERRORE ", name)
        # else:
            # print("[INFO] immagine inutile: ", name)
            # break
    x_test = np.array(x_test)
    y_true = np.array(y_test)
    print(np.shape(x_test))
    print("lettura DS finita")
    model = tf.keras.models.load_model('model/model_3class_ft_20220311-173216')
    # model.summary()    
    y_pred = []
    scores = []
    predictions = model.predict(x_test, verbose=0)
    for prediction in predictions:        
        classes = np.argmax(prediction)
        prob = prediction[classes]
        y_pred.append(classes)
        scores.append(prediction)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    result = pd.DataFrame({'id': filename, 'y_pred':y_pred, 'y_true':y_true})
    result.to_csv('result_3_class.csv')
