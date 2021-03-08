import numpy as np
import tensorflow as tf

from representer.representer_model import RepresenterModel

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
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, kernel_regularizer="l2", use_bias=False),
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.load_weights("./output/fashion_mnist_lr_checkpoint")

feature_model = model.get_layer(index=0)
prediction_network = model.get_layer(index=1)

# Note there's a representer value for each class.
representer_values = np.zeros((num_training_points, num_test_points, 10))

representer_model = RepresenterModel(
    feature_model,
    prediction_network,
    train_images,
    categorical_train_labels,
    test_images,
    model.loss
)

for i in range(num_training_points):
    print("Computing representer values of training point", i, "out of", num_training_points)
    for j in range(num_test_points):
        representer_values[i, j] = representer_model.get_representer_value(i, j)

np.savez(
    "./output/representer_model_on_fashion_mnist_lr.npz",
    representer_values=representer_values,
)