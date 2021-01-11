import math

import numpy as np
import tensorflow as tf

from influence.influence_with_s_test import InfluenceWithSTest

mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

categorical_train_labels = tf.keras.utils.to_categorical(train_labels)
categorical_test_labels = tf.keras.utils.to_categorical(test_labels)

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, kernel_regularizer="l2", bias_regularizer="l2"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.load_weights("./output/mnist_checkpoint")

# Number of training points = 54000
# Number of validation points = 6000
# Number of test points = 10000

num_training_points = 54000
num_test_points = 10

influence_values = np.zeros((num_training_points, num_test_points))

influence_model = InfluenceWithSTest(
    model,
    train_images,
    categorical_train_labels,
    test_images,
    categorical_test_labels,
    model.loss,
    damping=0.2,
    dtype=np.float64,
    cg_tol=1e-05,
)

for j in range(num_test_points):

    print("Computing influence of test point", j, "out of", num_test_points)
    for i in range(num_training_points):
        influence_values[i, j] = influence_model.get_influence_on_loss(i, j)

np.savez(
    "./output/influence_with_s_test_on_full_mnist.npz",
    influence_values=influence_values,
)
