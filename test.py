import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv

if __name__=='__main__':
    current_path = '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    base_path = current_path + 'dataset/'
    _, _, test_df = read_csv(base_path)
    x_test = []
    test_df = test_df.sample(frac=1)
    print(test_df.values.tolist()[0])
    n =  len(test_df)
    for path in tqdm(test_df["filename"][:n]):
        img = cv2.imread(path, 0) / 255.
        img = cv2.resize(img, (256, 256))
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        x_test.append(img)
    y_true = test_df["label"][:n]
    print(np.shape(x_test))
    print("lettura DS finita")
    model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220223-200225/')
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
    y_true = np.array(y_true)
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)

