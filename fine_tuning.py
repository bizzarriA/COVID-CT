import cv2
import datetime
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_slice
from utils.model import get_model

if __name__=="__main__":
    base_path="dataset/unife/"
    current_path ='' # '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    csv = pd.read_csv(base_path+'unife.csv')
    train_df = csv[csv['split'] == 'train']
    val_df = csv[csv['split'] == 'val']
    test_df = csv[csv['split'] == 'test']
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    print(n_train, n_val, n_test)
    n_train, n_val, n_test = 100, 13, 10
    print("read train images")
    x_train = []
    for name in tqdm(train_df["filename"][:n_train]):
        try:
            # print(name)
            img = cv2.imread(name, 0) / 255
            # img = cv2.resize(img, (256, 256))
            img = np.expand_dims(img, axis=-1)
            x_train.append(img)
        except:
            print("ERRORE ", name)
    x_train = np.array(x_train)
    y_train = tf.keras.utils.to_categorical(train_df["label"][:n_train], 3)
    print("read val images")
    x_val = []
    for name in tqdm(val_df["filename"][:n_val]):
        try:
            # print(name)
            img = cv2.imread(name, 0) / 255
            img = cv2.resize(img, (256, 256))
            img = np.expand_dims(img, axis=-1)
            x_val.append(img)
        except:
            print("ERRORE ", name)
    x_val = np.array(x_val)
    y_val = tf.keras.utils.to_categorical(val_df["label"][:n_val], 3)
    print(np.shape(x_train[0]))
    # model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220223-200225')
    model = get_model(width=256, height=256)
    model.summary()
    fine_tune_at = -7
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
        print(layer)
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_ft_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
                 tensorboard_callback,
                 checkpoint_cb,
                 early_stopping_cb
                 ]

    optimizer = tf.keras.optimizers.Adam(0.001)  # * hvd.size())
    print("[INFO] Model compile")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['acc'],
    )
    #print("Shape x and y train ",np.shape(x_train), np.shape(y_train))
    #print("Shape x and y val ",np.shape(x_val), np.shape(y_val))
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        callbacks=callbacks,
        batch_size=2,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_ft_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # y_true = y_test
    # pd.DataFrame({'id':name_test}).to_csv('test_set.csv')
    # for i in tqdm(range(len(x_test))):
    #     prediction = model.predict(x_test[i], batch_size=1, verbose=0)
    #     classes = np.argmax(prediction, axis=1)
    #     prob = prediction[0, classes]
    #     y_pred.append(classes[0])
    #     scores.append(prediction)
    # y_pred = tf.keras.utils.to_categorical(y_pred, 3)
    # result = pd.DataFrame({'id': name, 'y_pred':y_pred, 'y_true':y_true})
    # result.to_csv('result.csv')
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # print(y_true)
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print("[info] ", test_acc)
    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # print("Confusion matrix:\n",confusion_mtx)
