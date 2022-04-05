from fileinput import filename
import cv2
import datetime
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from utils.data import read_csv
from utils.model import get_model
from utils.preprocessing import auto_body_crop

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ISIZE = 256
    base_path = 'dataset/2A_images/'
    crop = True
    print("[INFO] Read Train - Val - Test:")
    train_df, test_df, val_df, new_df = read_csv(img_path='train/')
    print(len(train_df))
    train_df = train_df.sample(n=10000)
    # new_df = new_df.sample(n=3)
    train_df = train_df.append(new_df, ignore_index=True)
    print(len(train_df))
    train_df = train_df.sample(frac=1)
    train_df.to_csv('use_for_training.csv')
    # train_df = pd.read_csv('total_train_data.csv')
    # train_df = train_df.iloc[:, 2:]
    # train_df = train_df.sample(n=10)
    x_train = []
    y_train = []
    print(train_df)
    file_name =[]
    for _, row in train_df.iterrows():
        try:
                # print(row[0], row[1])
                name = row[0]
                img = cv2.imread(name, 0)
                img = cv2.resize(img, (ISIZE, ISIZE))
                img = np.expand_dims(img, axis=-1)
                x_train.append(img / 255.)
                y_train.append(tf.keras.utils.to_categorical(row[1], 3))
                filename.append(name)
                # print(y_train[-1])
        except:
            continue
    # print(np.bincount(y_train))
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x, y, names = shuffle(x_train, y_train, file_name)
    n = int(len(y)*0.15)
    x_train = x[:-(2*n)]
    y_train = y[:-(2*n)]
    train_name = names[:-(2*n)]
    x_val = x[-(2*n):-n]
    y_val = y[-(2*n):-n]
    val_name = names[-(2*n):-n]
    x_test = x[-n:]
    y_test = y[-n:]
    test_name = names[-n:]
    pd.DataFrame({'name': train_name, 'label': y_train}).to_cvs('train.csv')
    pd.DataFrame({'name': val_name, 'label': y_val}).to_cvs('val.csv')
    pd.DataFrame({'name': test_name, 'label': y_test}).to_cvs('test.csv')
    # y_train = tf.keras.utils.to_categorical(y_train, 3)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    BATCH_SIZE_PER_REPLICA = 16
    global_batch_size = (BATCH_SIZE_PER_REPLICA *
                         mirrored_strategy.num_replicas_in_sync)

    print(mirrored_strategy.num_replicas_in_sync)
    with mirrored_strategy.scope():
        model = get_model(width=256, height=256)
    # checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("model_jpeg_.h5", save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,
                                                         restore_best_weights=True)
    log_dir = "log/model_prova_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1),
                 tensorboard_callback,
                #  checkpoint_cb,
                 early_stopping_cb
                 ]

    optimizer = tf.keras.optimizers.Adam(0.001)  # * hvd.size())
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
    model.save("model/model_prova_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # print("[INFO] Test Phase: ")
    # y_pred = []
    # scores = []
    # for i in range(len(x_test)):
    #     prediction = model.predict(x_test[i], verbose=0)
    #     classes = np.argmax(prediction, axis=1)
    #     prob = prediction[0, classes]
    #     y_pred.append(classes[0])
    #     scores.append(prediction)
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_test)
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print(test_acc)
    # confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    # print("Confusion matrix:\n",confusion_mtx)

