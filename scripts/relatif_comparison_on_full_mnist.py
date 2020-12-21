import math

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel

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

num_training_points = 54
num_test_points = 10

influence_values = np.zeros((num_training_points, num_test_points))
theta_relatif_values = np.zeros((num_training_points, num_test_points))
l_relatif_values = np.zeros((num_training_points, num_test_points))

for i in range(num_training_points):

    print("Computing influence of training point", i, "out of", num_training_points)

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
        influence_values[i, j] = influence_model.get_influence_on_loss(
            test_images[j], categorical_test_labels[j]
        )

    theta_relatif_values[i] = influence_values[i] / np.linalg.norm(
        influence_model.get_inverse_hvp()
    )

    flat_training_gradient = np.concatenate(
        [tf.reshape(t, [-1]) for t in influence_model.get_training_gradient()]
    )
    l_relatif_values[i] = influence_values[i] / math.sqrt(
        np.dot(influence_model.get_inverse_hvp(), flat_training_gradient)
    )

np.savez(
    "./output/relatif_comparison_on_full_mnist.npz",
    influence_values=influence_values,
    theta_relatif_values=theta_relatif_values,
    l_relatif_values=l_relatif_values,
)
