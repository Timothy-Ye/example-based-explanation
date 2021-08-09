import numpy as np
import tensorflow as tf

from data_shapley import get_shapley_values

# Using Fashion MNIST dataset.
dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Using a subset of training and test points.
num_train_points = 6000
num_test_points = 400

# Normalise image values.
train_x = train_images[:num_train_points] / 255.0
test_x = test_images[:num_test_points] / 255.0

# Outputs are in categorical form.
train_y = tf.keras.utils.to_categorical(train_labels[:num_train_points], dtype="float64")
test_y = tf.keras.utils.to_categorical(test_labels[:num_test_points], dtype="float64")

tf.keras.backend.set_floatx("float64")

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, kernel_regularizer="l2", use_bias=False),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.save_weights("./output/untrained_lr_on_fashion_mnist")

def performance_metric(model):
    return model.evaluate(
        test_x, test_y, verbose=0
    )[0]

def model_reset_fn(model):
    model.load_weights("./output/untrained_lr_on_fashion_mnist")

def model_train_fn(model, idxs):
    model.fit(train_x[idxs], train_y[idxs], verbose=0, shuffle=False)

shapley_values = get_shapley_values(
    model,
    num_train_points,
    performance_metric,
    model_reset_fn,
    model_train_fn,
    max_iters=500,
    truncate_threshold=0.1,
    verbose=True,
)

np.savez("./output/lr_on_fashion_mnist_shapley", shapley_values=shapley_values)
