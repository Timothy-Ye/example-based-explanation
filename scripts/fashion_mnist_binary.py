import numpy as np
import tensorflow as tf

fashion_mnist_dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataset.load_data()

binary_train_images = train_images[(train_labels == 0) | (train_labels == 2)]
binary_train_labels = train_labels[(train_labels == 0) | (train_labels == 2)]

binary_test_images = test_images[(test_labels == 0) | (test_labels == 2)]
binary_test_labels = test_labels[(test_labels == 0) | (test_labels == 2)]

binary_train_images = binary_train_images / 255.0
binary_test_images = binary_test_images / 255.0

categorical_train_labels = (binary_train_labels == 0).astype(np.float64).reshape((-1, 1))
categorical_test_labels = (binary_test_labels == 0).astype(np.float64).reshape((-1, 1))

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(1, kernel_regularizer="l2", use_bias=False),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    binary_train_images, categorical_train_labels, epochs=10
)

model.save_weights("./output/fashion_mnist_binary_checkpoint")
