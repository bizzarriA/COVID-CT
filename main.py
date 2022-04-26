from fileinput import filename
import cv2
import datetime
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from utils.data import read_csv
from utils.model import get_model
from utils.preprocessing import auto_body_crop


if __name__ == '__main__':
    ISIZE = 256
    N_CALSSI = 2
    base_path = 'dataset/'
    crop = True
    print("[INFO] Read Train - Val - Test:")

    train_df = pd.read_csv('total_train_data.csv')
    train_df = train_df.iloc[:, 2:]
    old_train = pd.read_csv(base_path + 'train_COVIDx_CT-2A.txt', sep=" ", header=None)
    old_train.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    old_train = old_train.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    old_train['filename']='train/'+old_train['filename']
    # old_train = old_train.sample(n=20000)
    train_df = train_df.append(old_train, ignore_index=True)
    train_df = train_df.sample(frac=1)
    print(train_df)
    x_train = []
    y_train = []
    for _, row in tqdm(train_df.iterrows()):
        try:
            name = row[0]
            if os.path.exists(name) and row[1] != 0:  
                img = cv2.imread(name, 0)
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                x_train.append(img / 255.)
                y = row[1] - 1
                y_train.append(y)

        except:
            continue
    print("[INFO] immagini train per etichetta: ", np.bincount(y_train))
    x_train = np.array(x_train)
    y_train = np.array(tf.keras.utils.to_categorical(y_train, N_CALSSI))
    val_df = pd.read_csv('total_val_data.csv')
    val_df = val_df.iloc[:, 1:]
    val_df['filename'] = 'val/'+val_df['filename']
    old_val = pd.read_csv(base_path + 'val_COVIDx_CT-2A.txt', sep=" ", header=None)
    old_val.columns = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
    old_val = old_val.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    val_df = val_df.append(old_train, ignore_index=True)
    val_df = val_df.sample(frac=1)
    print(val_df)
    x_val = []
    y_val = []
    for _, row in tqdm(val_df.iterrows()):
        try:
            name = row[0]
            if os.path.exists(name) and row[1] != 0:
                img = cv2.imread(name, 0)
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                x_val.append(img / 255.)
                y = row[1] - 1
                y_val.append(y)

        except:
            continue
    print("[INFO] immagini validazione per etichetta: ", np.bincount(y_val))
    x_val = np.array(x_val)
    y_val = np.array(tf.keras.utils.to_categorical(y_val, N_CALSSI))



    mirrored_strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 16
    global_batch_size = (BATCH_SIZE_PER_REPLICA *
                         mirrored_strategy.num_replicas_in_sync)

    print(mirrored_strategy.num_replicas_in_sync)
    with mirrored_strategy.scope():
        model = get_model(width=256, height=256, n_class=N_CALSSI)
    # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_binaty_totale_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
                 tensorboard_callback,
                #  checkpoint_cb,
                 early_stopping_cb
                 ]

    optimizer = tf.keras.optimizers.Adamax(0.001)  # * hvd.size())
    print("[INFO] Model compile")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=['acc'],
    )
    print("Shape x and y train ",np.shape(x_train), np.shape(y_train))
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=callbacks,
        batch_size=global_batch_size,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_binary_totale_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # y_pred = []
    # scores = []
    # predictions = model.predict(x, verbose=1, batch_size=1)
    # for prediction in tqdm(predictions):
    #     classes = np.argmax(prediction)
    #     prob = prediction[classes]
    #     y_pred.append(classes)
    #     scores.append(prediction)
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_test)
    # # print(np.shape(y_pred), np.shape(y_true))
    # print(y_true[:5], y_pred[:5])
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print("[INFO] normal accuracy: ")
    # print(test_acc)
   
    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # print("Confusion matrix:\n",confusion_mtx)
