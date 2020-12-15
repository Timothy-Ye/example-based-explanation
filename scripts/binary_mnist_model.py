import numpy as np
import tensorflow as tf

mnist_dataset = tf.keras.datasets.mnist
(full_train_images, full_train_labels), (
    full_test_images,
    full_test_labels,
) = mnist_dataset.load_data()

train_images = full_train_images[(full_train_labels == 1) | (full_train_labels == 7)]
train_labels = full_train_labels[(full_train_labels == 1) | (full_train_labels == 7)]

test_images = full_test_images[(full_test_labels == 1) | (full_test_labels == 7)]
test_labels = full_test_labels[(full_test_labels == 1) | (full_test_labels == 7)]

train_images = train_images / 255.0
test_images = test_images / 255.0

categorical_train_labels = (train_labels == 1).astype(np.float64).reshape((-1, 1))
categorical_test_labels = (test_labels == 1).astype(np.float64).reshape((-1, 1))

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1, kernel_regularizer="l2", bias_regularizer="l2"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)


model.fit(train_images, categorical_train_labels, epochs=10, validation_split=0.1)

model.save_weights("./output/binary_mnist_checkpoint")
