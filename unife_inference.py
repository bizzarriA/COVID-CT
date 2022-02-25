from utils.data import read_slice

import numpy as np
import tensorflow as tf
from tqdm import tqdm

if __name__=="__main__":
    base_path="dataset/unife/"
    current_path ='' # '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    x_patient, y_patient, x_tot, y_tot = read_slice(current_path + base_path)    
    print("lettura DS finita")
    model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220224-203030/')
    model.summary()
    y_pred = []
    scores = []

    print("[INFO] PROVIAMO TOTALE:")
    x_test = x_tot
    y_true = y_tot
    for i in range(len(x_test)):
        prediction = model.predict(x_test[i], batch_size=1, verbose=1)
        classes = np.argmax(prediction, axis=1)
        prob = prediction[0, classes]
        y_pred.append(classes[0])
        scores.append(prediction)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    test_acc = sum(y_pred[0] == y_true[0]) / len(y_true)
    print("[info] ", test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    
    
####### FINE PRIMA PARTE #####


    # print("[INFO] PROVIAMO per paziente:")
    # x_test = x_patient
    # y_true = y_patient
    # n = len(x_test)
    # for i in range(len(n)):
    #     print(np.shape(x_test[i]))
    #     prediction = [model.predict(x_test[i][k], verbose=0) for k in range(len(x_test[i]))]
    #     print(prediction)
    #     classes = np.mean(np.argmax(prediction, axis=1))
    #     prob = prediction[0, classes]
    #     y_pred.append(classes[0])
    #     scores.append(prediction)
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print(test_acc)
    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # print("Confusion matrix:\n",confusion_mtx)