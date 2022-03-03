import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv

if __name__=='__main__':
    current_path = '' #'/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    base_path = current_path + 'dataset/'
    _, _, test_df = read_csv(base_path)
    # csv = pd.read_csv(base_path + 'unife.csv')
    # test_df = csv[csv['split']=='test']
    # test_df = test_df[test_df['label']!=0]
    x_test = []
    y_true = []
    name = []
    test_df = test_df.sample(frac=1)
    print(test_df.values.tolist()[0])
    n =  len(test_df)
    for i, path in enumerate(test_df["filename"][:n]):
        if path[:22] == 'dataset/2A_images/HUST':
            img = cv2.imread(path, 0) / 255.
            img = cv2.resize(img, (256, 256))
            img = np.expand_dims(img, axis=-1)
            img = np.expand_dims(img, axis=0)
            x_test.append(img)
            y_true.append(test_df["label"][i])
            name.append(path)
    print(np.shape(x_test))
    print("lettura DS finita")
    model = tf.keras.models.load_model(current_path + 'model/model_bin_ft_20220303-113909')
    model.summary()
    y_pred = []
    scores = []
    for i in range(len(x_test)):
        prediction = model.predict(x_test[i], verbose=0)
        classes = np.argmax(prediction, axis=1)
        prob = prediction[0, classes]
        y_pred.append(classes[0])
        scores.append(prediction)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true) - 1
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
    result = pd.DataFrame({'id': name, 'y_pred':y_pred, 'y_true':y_true})
    result.to_csv('result.csv')

