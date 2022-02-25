import cv2
import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.segmentation import segment_cv2
from utils.data import read_csv, load_and_process
from utils.model import get_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_path = 'dataset/'
    print("[INFO] Read Train - Val - Test:")
    train_df, test_df, val_df = read_csv(base_path)
    x_train = [load_and_process(row) for row in tqdm(train_df[:5])]
    x_val = [load_and_process(row) for row in tqdm(val_df[:5])]
    # x_test, y_test = np.transpose([load_and_process(row) for row in tqdm(test_df[:50])])
    y_train = [row[1] for row in train_df[:5]]
    y_val = [row[1] for row in val_df[:5]]
    y_train = tf.keras.utils.to_categorical(y_train, 3)
    y_val = tf.keras.utils.to_categorical(y_val, 3)
    model = get_model(width=256, height=256)
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
        batch_size=32,
        shuffle=True,
        verbose=1
    )
    model.save("model/model_jpeg_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
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

