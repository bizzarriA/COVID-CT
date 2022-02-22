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
    n_train, n_val, n_test = 100, 10, 10
    x_train = [cv2.imread(train_df["filename"][i], cv2.IMREAD_GRAYSCALE) for i in range(n_train)]
    x_train = np.expand_dims(x_train, axis=-1)
    y_train = tf.keras.utils.to_categorical(train_df["label"][:n_train], 3)
    print(np.shape(x_train), np.shape(y_train))
    x_val = [cv2.imread(val_df["filename"][i], 0) for i in range(n_val)]
    y_val = tf.keras.utils.to_categorical(val_df["label"][:n_val], 3)
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    batch_size = 1
    train_dataset = (
        train_loader.shuffle(len(x_train)).batch(batch_size, drop_remainder=True).prefetch(2)
    )
    validation_dataset = (
        validation_loader.shuffle(len(x_val)).batch(batch_size).prefetch(2)
    )
    model = get_model(width=np.shape(x_train[0])[0], height=np.shape(x_train[0])[0])
    # model.summary()

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
        epochs=5,
        # steps_per_epoch=50,
        batch_size=batch_size,
        shuffle=True,
        verbose=1
    )
