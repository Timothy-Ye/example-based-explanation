import numpy as np
import tensorflow as tf

num_training_points = 600
num_test_points = 1000

fashion_mnist_dataset = tf.keras.datasets.fashion_mnist
(full_train_images, full_train_labels), (full_test_images, full_test_labels) = fashion_mnist_dataset.load_data()

train_images = full_train_images[:num_training_points]
train_labels = full_train_labels[:num_training_points]

test_images = full_test_images[:num_test_points]
test_labels = full_test_labels[:num_test_points]

categorical_train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=10)
categorical_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=10)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(2, 3, activation="relu", use_bias=False, kernel_regularizer="l2", input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(2, 3, activation="relu", use_bias=False, kernel_regularizer="l2"),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer="l2", use_bias=False),
    tf.keras.layers.Dense(10, kernel_regularizer="l2", use_bias=False),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    tf.reshape(train_images, [-1, 28, 28, 1]), categorical_train_labels, epochs=500
)

model.save_weights("./output/fashion_mnist_checkpoint")
