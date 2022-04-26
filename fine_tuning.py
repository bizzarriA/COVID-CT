import cv2
import datetime
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv
from utils.model import get_model
from utils.preprocessing import auto_body_crop

N_CLASSI = 2
CLASSI = ['Common_pneuma', 'Covid_19']
ISIZE = 256
crop = False

if __name__=="__main__":
    
    train_df = pd.read_csv('total_train_data.csv')
    train_df = train_df.iloc[:, 2:]
    print("read train images")
    # train_df = train_df.sample(n=10)
    x_train = []
    y_train = []
    # valide = os.listdir(base_path)
    print(len(train_df))
    print(train_df)
    for i, row in tqdm(train_df.iterrows()):
        # name = row[0]
    #    if i < n_train: # and name.split('/')[-1] in valide:
            try:
                name = row[0]
                if row[1] != 0:
                    if os.path.exists(name):
                        img = cv2.imread(name, 0)
                        img = cv2.resize(img, (ISIZE, ISIZE))
                        img = np.expand_dims(img, axis=-1)
                        x_train.append(img/255.)
                        y = row[1]-1
                        y_train.append(y)
            except:
                continue
                #print("ERRORE ", name)
     #   else:
        #     if i>=n_train:
      #      break
        #     continue
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("Shape x and y train ",np.shape(x_train), np.shape(y_train))
    print("read val images")
    val_df = pd.read_csv('total_val_data.csv')
    val_df = val_df.iloc[:, 1:]
    val_df['filename'] = 'val/' + val_df['filename']
    x_val = []
    y_val = []
    for _, row in tqdm(val_df.iterrows()):
        try:
            name = row[0]
            if row[1] != 0:
                if os.path.exists(name):
                    img = cv2.imread(name, 0)
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    x_val.append(img / 255.)
                    y = row[1]-1
                    y_val.append(y)

        except:
            continue
    x_val = np.array(x_val)
    y_val = np.array(tf.keras.utils.to_categorical(y_val, N_CLASSI))
    y_train = np.array(tf.keras.utils.to_categorical(y_train, N_CLASSI)) 
    print(np.shape(x_train), np.shape(y_train), np.shape(x_val), np.shape(y_val))
    ### Mirrored strategy ###
    mirrored_strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 16
    global_batch_size = (BATCH_SIZE_PER_REPLICA *
                         mirrored_strategy.num_replicas_in_sync)
    
    print(mirrored_strategy.num_replicas_in_sync)
    # model = tf.keras.models.load_model(current_path + 'model/model_bin_20220302-191317')
    # dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(global_batch_size)#.repeat()
    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # dataset = dataset.with_options(options)
    # dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    with mirrored_strategy.scope():
        model = get_model(width=ISIZE, height=ISIZE, n_class=N_CLASSI)
        # model = tf.keras.models.load_model("model/model_split_20220406-192229")
    
    model.summary()
    # fine_tune_at = -7

    # Freeze all the layers before the `fine_tune_at` layer
    # for layer in model.layers[:fine_tune_at]:
      #   layer.trainable = False
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_3class.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_2_calssi_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
                 tensorboard_callback,
                 #checkpoint_cb,
                 early_stopping_cb
                 ]

    optimizer = tf.keras.optimizers.Adam(0.001) 
    print("[INFO] Model compile")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['acc'],
    )
    
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=callbacks,
        # steps_per_epoch = 100,
        batch_size=global_batch_size,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_2_calssi_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
