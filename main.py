import cv2
import numpy as np
import tensorflow as tf

from utils.segmentation import get_segmented_lungs
from utils.data import read_csv, plot_img
from utils.model import get_model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    base_path = '/Users/alicebizzarri/Downloads/archive/'
    train_df, test_df, val_df = read_csv(base_path)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    print(n_train, n_val, n_test)
    n_train, n_val, n_test = 10, 10, 10
    x_train = [cv2.imread(train_df["filename"][i], 0) for i in range(n_train)]
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = tf.one_hot(train_df["label"][:n_train], 3)
    print(np.shape(x_train), np.shape(y_train))
    x_val = [cv2.imread(val_df["filename"][i], 0) for i in range(n_val)]
    y_val = tf.one_hot(val_df["label"][:n_val], 3)
    model = get_model()
    model.summary()

    optimizer = tf.keras.optimizers.Adam(0.001)  # * hvd.size())
    # optimizer = hvd.DistributedOptimizer(optimizer)
    print("[INFO] Model compile")
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=optimizer,
        metrics=['acc'],
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=2,
        shuffle=True,
        verbose=1
    )