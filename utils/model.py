# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def semple_model(width, height, n_class = 3):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=n_class, activation="softmax"))
    
    return model


def get_model(width=512, height=512, n_class = 3):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, 1))

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool2D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=n_class, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="covid-net")
    return model
