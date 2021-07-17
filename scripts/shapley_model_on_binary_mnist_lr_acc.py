import numpy as np
import tensorflow as tf

from shapley.shapley_model import ShapleyModel

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

model.save_weights("./output/binary_mnist_untrained_lr")

num_training_points = 13007
num_test_points = 400

def performance_metric(model):
    return model.evaluate(binary_test_images[:num_test_points],
                          categorical_test_labels[:num_test_points],
                          verbose=0)[1]

def model_reset_fn(model):
    model.load_weights("./output/binary_mnist_untrained_lr")

def model_train_fn(model, idxs):
    model.fit(binary_train_images[idxs],
        categorical_train_labels[idxs], verbose=0, shuffle=False)

shapley_model = ShapleyModel(
    model,
    num_training_points,
    performance_metric,
    model_reset_fn,
    model_train_fn,
    max_iters=1000,
    truncate_threshold=0.01,
    verbose=True
)

shapley_values = shapley_model.get_shapley_values()

np.savez(
    "./output/shapley_model_on_binary_mnist_lr_acc.npz",
    shapley_values=shapley_values
)
