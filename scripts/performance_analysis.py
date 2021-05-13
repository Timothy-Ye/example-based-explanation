import time

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel
from representer.representer_model import RepresenterModel

mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

binary_train_images = train_images[(train_labels == 1) | (train_labels == 7)]
binary_train_labels = train_labels[(train_labels == 1) | (train_labels == 7)]

binary_test_images = test_images[(test_labels == 1) | (test_labels == 7)]
binary_test_labels = test_labels[(test_labels == 1) | (test_labels == 7)]

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

model.load_weights("./output/binary_mnist_checkpoint")

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

feature_model = model.get_layer(index=0)
prediction_network = model.get_layer(index=1)

representer_model = RepresenterModel(
    feature_model,
    prediction_network,
    binary_train_images,
    categorical_train_labels,
    binary_test_images,
    model.loss
)

training_idxs = np.arange(0, 1000).reshape((10, 100))
test_idxs = np.arange(0, 100).reshape((10, 10))

influence_times = []
representer_times = []

for i in range(10):

    print("Starting trial {} of {}.".format(i, 10))

    start_time = time.time()

    for j in training_idxs[i]:
        for k in test_idxs[i]:
            influence_model.get_influence_on_loss(j, k)

    end_time = time.time()
    influence_times.append(end_time - start_time)

    start_time = time.time()

    for j in training_idxs[i]:
        for k in test_idxs[i]:
            representer_model.get_representer_value(j, k)

    end_time = time.time()
    representer_times.append(end_time - start_time)

print("Influence functions: {:.3f}s +- {:.3f}s".format(np.mean(influence_times), np.std(influence_times)))
print("Representer values: {:.3f}s +- {:.3f}s".format(np.mean(representer_times), np.std(representer_times)))