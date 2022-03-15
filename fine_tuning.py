import cv2
import datetime
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from utils.data import read_csv
from utils.model import get_model
from utils.preprocessing import auto_body_crop

N_CLASSI = 3
CLASSI = ['Normal', 'Common_pneuma', 'Covid_19']
ISIZE = 256

if __name__=="__main__":
    base_path="dataset/"
    train_df, test_df, val_df = read_csv(base_path)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    # n_train, n_val, n_test = 100, 13, 10
    print("read train images")
    x_train = []
    y_train = []
    for i, row in tqdm(train_df.iterrows()):
        if i < n_train:
            try:
                name = row[0]
                img = cv2.imread(name, 0) 
                img, validate = auto_body_crop(img)
                if validate:
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    x_train.append(img)
                    y_train.append(tf.keras.utils.to_categorical(row[1], N_CLASSI))
            except:
                print("ERRORE ", name)
        else:
            break
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print("Shape x and y train ",np.shape(x_train), np.shape(y_train))
    print("read val images")
    x_val = []
    y_val = []
    for i, row in tqdm(val_df.iterrows()):
        if i < n_val:
            try:
                name = row[0]
                img = cv2.imread(name, 0)
                img, validate = auto_body_crop(img)
                if validate:
                    img = cv2.resize(img, (ISIZE, ISIZE))
                    img = np.expand_dims(img, axis=-1)
                    x_val.append(img)
                    y_val.append(tf.keras.utils.to_categorical(row[1], N_CLASSI))
            except:
                print("ERRORE ", name)
        else:
            break
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    print(np.shape(x_train[0]))
    print("Shape x and y val ",np.shape(x_val), np.shape(y_val))
    ### Mirrored strategy ###
    mirrored_strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 16
    global_batch_size = (BATCH_SIZE_PER_REPLICA *
                         mirrored_strategy.num_replicas_in_sync)
    
    print(mirrored_strategy.num_replicas_in_sync)
    # model = tf.keras.models.load_model(current_path + 'model/model_bin_20220302-191317')
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(global_batch_size)#.repeat()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)
    # dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

    with mirrored_strategy.scope():
        model = get_model(width=ISIZE, height=ISIZE)
    
    model.summary()
    # fine_tune_at = -7

    # Freeze all the layers before the `fine_tune_at` layer
    # for layer in model.layers[:fine_tune_at]:
    #     layer.trainable = False
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_3class.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_3_class_tf_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [#tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
                 tensorboard_callback,
                 #checkpoint_cb,
                 early_stopping_cb
                 ]

    optimizer = tf.keras.optimizers.SGD() 
    print("[INFO] Model compile")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=['acc'],
    )
    
    model.fit(
        dataset,
        validation_data=(x_val, y_val),
        epochs=100,
        callbacks=callbacks,
        # steps_per_epoch = 100,
        batch_size=global_batch_size,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_3class_ft_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    x_test = []
    y_test = []
    filename = []
    test_df = test_df.sample(frac=1)
    print(test_df.values.tolist()[0])
    for i, row in tqdm(test_df.iterrows()):
        if i < n_test:
            try:
                name = row[0]
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
        else:
            break
    x_test = np.array(x_test)
    y_true = np.array(y_test)
    print(np.shape(x_test))
    print("lettura DS finita")
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
    result = pd.DataFrame({'id': filename, 'y_pred':y_pred, 'y_true':y_true})
    result.to_csv('result_3_class.csv')
