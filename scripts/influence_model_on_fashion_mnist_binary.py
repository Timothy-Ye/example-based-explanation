import math

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel

fashion_mnist_dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist_dataset.load_data()

binary_train_images = train_images[(train_labels == 0) | (train_labels == 2)]
binary_train_labels = train_labels[(train_labels == 0) | (train_labels == 2)]

binary_test_images = test_images[(test_labels == 0) | (test_labels == 2)]
binary_test_labels = test_labels[(test_labels == 0) | (test_labels == 2)]

binary_train_images = binary_train_images / 255.0
binary_test_images = binary_test_images / 255.0

categorical_train_labels = ((binary_train_labels == 1).astype(np.float64).reshape((-1, 1)))
categorical_test_labels = (binary_test_labels == 1).astype(np.float64).reshape((-1, 1))

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

model.load_weights("./output/fashion_mnist_binary_checkpoint")

num_training_points = 12000
num_test_points = 2000

influence_values = np.zeros((num_training_points, num_test_points))
theta_relatif_values = np.zeros((num_training_points, num_test_points))
l_relatif_values = np.zeros((num_training_points, num_test_points))

influence_model = InfluenceModel(
    model,
    binary_train_images,
    categorical_train_labels,
    binary_test_images,
    categorical_test_labels,
    model.loss,
    damping=0.2,
    dtype=np.float64,
    cg_tol=1e-05,
)

for i in range(num_training_points):

    print("Computing influence values for training point", i, "out of", num_training_points)
    for j in range(num_test_points):
        influence_values[i, j] = influence_model.get_influence_on_loss(i, j)
        theta_relatif_values[i, j] = influence_model.get_theta_relatif(i, j)
        l_relatif_values[i, j] = influence_model.get_l_relatif(i, j)

np.savez(
    "./output/influence_model_on_fashion_mnist_binary.npz",
    influence_values=influence_values,
    theta_relatif_values=theta_relatif_values,
    l_relatif_values=l_relatif_values,
)
