from utils.data import read_slice

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import coremltools as ct


if __name__=="__main__":
    base_path="dataset/unife/"
    current_path ='' # '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    name, x_tot, y_tot = read_slice(current_path + base_path)    
    print("lettura DS finita")
    keras_model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220223-200225')
    keras_model.summary()
    y_pred = []
    scores = []

    # Define the input type as image, 
    # set pre-processing parameters to normalize the image 
    # to have its values in the interval [-1,1] 
    # as expected by the mobilenet model
    image_input = ct.ImageType(shape=(1, 256, 256, 1,),
                            bias=[-1,-1,-1], scale=1/127)

    # set class labels
    class_labels = [0, 1, 2]
    classifier_config = ct.ClassifierConfig(class_labels)

    # Convert the model using the Unified Conversion API
    model = ct.convert(
        keras_model, inputs=[image_input], classifier_config=classifier_config,
    )

    print("[INFO] PROVIAMO TOTALE:")
    x_test = x_tot
    y_true = y_tot
    for i in tqdm(range(len(x_test))):
        prediction = model.predict(x_test[i], batch_size=1, verbose=0)
        classes = np.argmax(prediction, axis=1)
        prob = prediction[0, classes]
        y_pred.append(classes[0])
        scores.append(prediction)
    result = pd.DataFrame({'id': name, 'y_pred':y_pred, 'y_true':y_true})
    result.to_csv('result.csv')
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    test_acc = sum(y_pred == y_true[0]) / len(y_true)
    print("[info] ", test_acc)
    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n",confusion_mtx)
