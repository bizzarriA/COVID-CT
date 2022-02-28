import cv2
import datetime
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_slice
from utils.model import get_model

if __name__=="__main__":
    base_path="dataset/unife/"
    current_path ='' # '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    patientes_path = os.listdir(base_path+'png/')
    random.shuffle(patientes_path)
    print(len(patientes_path))
    csv = pd.read_csv(base_path+'test_set_unife.csv')
    names = []
    y_tot = []
    x_tot= []
    for path in tqdm(patientes_path):
        try:
            # names.append(path)
            label = csv.loc[csv["filename"]==path]["label"].item()
            scans_path = os.listdir(base_path +'png/'+ path)
            centro = len(scans_path) // 2
            if label == 0:
                n = 60
            elif label == 1:
                n = 60
            elif label == 2:
                n = 60
            for scan_path in scans_path[centro-n:centro+n]:
                scan = cv2.imread("{}png/{}/{}".format(base_path,path,scan_path), 0)
                scan = cv2.resize(scan, (256, 256))
                scan = np.expand_dims(scan, axis=-1)
                scan = scan.astype('float')/255.
                x_tot.append(scan)
                y_tot.append(label)
                names.append(path+'_'+scan_path)
        except:
            print("[ERROR] read path:", path)        
    print("[INFO] Numero pazienti: {} - Numero totale immagini: {} - Numero totale etichette: {}".format(len(patientes_path), len(x_tot), len(y_tot)))
    print("lettura DS finita")
    # model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220223-200225')
    model = get_model(width=256, height=256)
    model.summary()
    n_tot = len(x_tot)
    n_train = int(n_tot*0.70)
    n_test = int(n_tot*0.15) 
    print(n_tot, n_train, n_test)
    y_tot = tf.keras.utils.to_categorical(y_tot, 3)
    print("Shape x and y:", np.shape(x_tot), np.shape(y_tot))
    batch_size = 2
    # train_set = zip(x_tot[:n_train],y_tot[:n_train])
    # val_set = zip(x_tot[n_train:-n_test],y_tot[n_train:-n_test])
    x_train, y_train = x_tot[:n_train],y_tot[:n_train]
    x_val, y_val = x_tot[n_train:-n_test],y_tot[n_train:-n_test]
    x_test = x_tot[-n_test:]
    y_test = y_tot[-n_test:]
    print("Shape train x and y:", np.shape(x_train), np.shape(y_train))
    print("Shape val x and y:", np.shape(x_val), np.shape(y_val))
 
    # fine_tune_at = -7
    # # Freeze all the layers before the `fine_tune_at` layer
    # for layer in model.layers[:fine_tune_at]:
    #     layer.trainable = False
    #     print(layer)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_jpeg_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
    model.save("model/model_jpeg_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
