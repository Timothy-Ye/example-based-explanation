import math

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel

num_training_points = 1200
num_test_points = 1200

fashion_mnist_dataset = tf.keras.datasets.fashion_mnist
(full_train_images, full_train_labels), (full_test_images, full_test_labels) = fashion_mnist_dataset.load_data()

train_mask = (full_train_labels == 0) | (full_train_labels == 2) | (full_train_labels == 4) | (full_train_labels == 6)
test_mask = (full_test_labels == 0) | (full_test_labels == 2) | (full_test_labels == 4) | (full_test_labels == 6)

train_images = full_train_images[train_mask][:num_training_points]
train_labels = full_train_labels[train_mask][:num_training_points]

test_images = full_test_images[test_mask][:num_test_points]
test_labels = full_test_labels[test_mask][:num_test_points]

train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = train_labels / 2
test_labels = test_labels / 2

categorical_train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=4)
categorical_test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=4)

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(2, 3, activation="relu", use_bias=False, kernel_regularizer="l2", input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Conv2D(2, 3, activation="relu", use_bias=False, kernel_regularizer="l2"),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu", kernel_regularizer="l2", use_bias=False),
    tf.keras.layers.Dense(4, kernel_regularizer="l2", use_bias=False),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.load_weights("./output/fashion_mnist_4class_checkpoint")

influence_values = np.zeros((num_training_points, num_test_points))
theta_relatif_values = np.zeros((num_training_points, num_test_points))
l_relatif_values = np.zeros((num_training_points, num_test_points))

influence_model = InfluenceModel(
    model,
    tf.reshape(train_images, [-1, 28, 28, 1]),
    categorical_train_labels,
    tf.reshape(test_images, [-1, 28, 28, 1]),
    categorical_test_labels,
    model.loss,
    damping=0.4,
    dtype=np.float64,
    cg_tol=1e-05,
    parameters=[model.trainable_variables[-1]]
)

for i in range(num_training_points):

    print("Computing influence values for training point", i, "out of", num_training_points)
    for j in range(num_test_points):
        influence_values[i, j] = influence_model.get_influence_on_loss(i, j)
        theta_relatif_values[i, j] = influence_model.get_theta_relatif(i, j)
        l_relatif_values[i, j] = influence_model.get_l_relatif(i, j)

np.savez(
    "./output/influence_model_on_fashion_mnist_4class.npz",
    influence_values=influence_values,
    theta_relatif_values=theta_relatif_values,
    l_relatif_values=l_relatif_values,
)
