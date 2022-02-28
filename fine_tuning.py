import cv2
import datetime
import numpy as np
import os
import tensorflow as tf

from utils.data import read_slice

if __name__=="__main__":
    base_path="dataset/unife/"
    current_path ='' # '/Users/alicebizzarri/PycharmProjects/COVID-CT/'
    name, x_tot, y_tot = read_slice(current_path + base_path, shuffle=True)    
    print("lettura DS finita")
    model = tf.keras.models.load_model(current_path + 'model/model_jpeg_20220223-200225')
    model.summary()
    print("Shape x and y:", np.shape(x_tot), np.shape(y_tot))
    n_tot = len(x_tot)
    n_train = int(n_tot*0.70)
    n_test = int(n_tot*0.15) 
    y_tot = tf.keras.utils.to_categorical(y_tot, 3)
    batch_size = 16
    train_set = tf.data.Dataset.from_tensor_slices((x_tot[:n_train],y_tot[:n_train])).shuffle(n_train).batch(batch_size, drop_remainder=True).prefetch(2)
    val_set = tf.data.Dataset.from_tensor_slices((x_tot[n_train:-n_test],y_tot[n_train:-n_test])).batch(batch_size, drop_remainder=True).prefetch(2)
    x_test = x_tot[-n_test:]
    y_test = y_tot[-n_test:]
    fine_tune_at = -7
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in model.layers[:fine_tune_at]:
        layer.trainable = False
        print(layer)
    
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
        train_set,
        validation_data=val_set,
        epochs=10,
        callbacks=callbacks,
        batch_size=32,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_jpeg_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
