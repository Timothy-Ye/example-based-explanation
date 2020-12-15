import math

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel

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

model.load_weights("./output/binary_mnist_checkpoint")

# Number of training points = 11706
# Number of validation points = 1301
# Number of test points = 2163

num_training_points = 1000
num_test_points = 200

if_values = np.zeros((num_training_points, num_test_points))
theta_relatif_values = np.zeros((num_training_points, num_test_points))
l_relatif_values = np.zeros((num_training_points, num_test_points))

for i in range(num_training_points):

    print("Computing influence of training point", i)

    influence_model = InfluenceModel(
        model,
        train_images,
        categorical_train_labels,
        model.loss,
        i,
        damping=0.2,
        dtype=np.float64,
        cg_tol=1e-05,
    )

    for j in range(num_test_points):
        if_values[i, j] = influence_model.get_influence_on_loss(
            test_images[j], categorical_test_labels[j]
        )

    theta_relatif_values[i] = if_values[i] / np.linalg.norm(
        influence_model.get_inverse_hvp()
    )

    flat_training_gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in influence_model.get_training_gradient()]
    )
    l_relatif_values[i] = if_values[i] / math.sqrt(
        np.dot(influence_model.get_inverse_hvp(), flat_training_gradient)
    )

np.savez("./output/relatif_comparison_on_binary_mnist")
