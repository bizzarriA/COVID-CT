import cv2
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
    x_train = [cv2.imread(train_df["filename"][i], 1) for i in range(n_train)]
    y_train = tf.one_hot(train_df["label"][:n_train], 3)
    print(np.shape(x_train), np.shape(y_train))
    x_val = [cv2.imread(val_df["filename"][i], 0) for i in range(n_val)]
    y_val = tf.one_hot(val_df["label"][:n_val], 3)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    batch_size = 1
    # Augment the on the fly during training.
    train_dataset = (
        train_loader  # .shuffle(len(x_train))
            # .repeat()
            .batch(batch_size, drop_remainder=True)
            .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
            .batch(batch_size)
            .prefetch(2)
    )
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
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )
