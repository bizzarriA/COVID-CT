import cv2
import datetime
import numpy as np
import tensorflow as tf

from utils.segmentation import segment_cv2
from utils.data import read_csv, plot_img
from utils.model import get_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_path = 'dataset/'
    train_df, test_df, val_df = read_csv(base_path)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    print(n_train, n_val, n_test)
    n_train, n_val, n_test = 10, 10, 10
    x_train = [cv2.imread(train_df["filename"][i], cv2.IMREAD_GRAYSCALE)/255. for i in range(n_train)]
    segment_cv2(train_df["filename"][8])
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = tf.keras.utils.to_categorical(train_df["label"][:n_train], 3)
    print(np.shape(x_train), np.shape(y_train))
    x_val = [cv2.imread(val_df["filename"][i], 0)/255. for i in range(n_val)]
    y_val = tf.keras.utils.to_categorical(val_df["label"][:n_val], 3)
    x_val = np.expand_dims(x_val, axis=-1)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    batch_size = 32
    train_dataset = (
        train_loader.shuffle(len(x_train)).batch(batch_size, drop_remainder=True).prefetch(2)
    )
    validation_dataset = (
        validation_loader.shuffle(len(x_val)).batch(batch_size).prefetch(2)
    )
    model = get_model(width=np.shape(x_train[0])[0], height=np.shape(x_train[0])[0])
    # model.summary()
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
    print("Shape x and y train ", np.shape(x_train), np.shape(y_train))
    # model.fit(
    #     x_train, y_train,
    #     validation_data=(x_val, y_val),
    #     epochs=5,
    #     callbacks=callbacks,
    #     # steps_per_epoch=50,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     verbose=1
    # )
