import numpy as np
import tensorflow as tf

from explainable_model import ExplainableModel


# Using MNIST digits dataset.
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Filter to binary (2-class) problem.
binary_train_images = train_images[(train_labels == 1) | (train_labels == 7)]
binary_train_labels = train_labels[(train_labels == 1) | (train_labels == 7)]
binary_test_images = test_images[(test_labels == 1) | (test_labels == 7)]
binary_test_labels = test_labels[(test_labels == 1) | (test_labels == 7)]

# Normalise image values.
train_x = binary_train_images / 255.0
test_x = binary_test_images / 255.0

# Outputs need to be of the form of a probability distribution.
train_y = (binary_train_labels == 1).astype(np.float64).reshape((-1, 1))
test_y = (binary_test_labels == 1).astype(np.float64).reshape((-1, 1))

num_train_points = len(train_x)
num_test_points = len(test_x)

# float64 gives better CG speed.
tf.keras.backend.set_floatx("float64")

# Simple LR model.
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

model.fit(train_x, train_y, epochs=10, shuffle=False)

# Instantiate ExplainableModel
e = ExplainableModel(
    model=model,
    feature_model=(lambda x: x),
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    loss_fn=model.loss,
    l2=0.01,
    ihvp_method="cg",
    cg_damping=0.01,
)

# Calculate and saves all intermediate values.

for i in range(num_train_points):
    print(
        "Calculating values for training point {} out of {}.".format(
            i, num_train_points
        )
    )

    e.get_train_grad(i)
    e.get_train_ihvp(i)
    e.get_train_feat(i)
    e.get_alpha_val(i)

for j in range(num_test_points):
    print("Calculating values for test point {} out of {}.".format(j, num_test_points))

    e.get_test_grad(j)
    e.get_test_ihvp(j)
    e.get_test_feat(j)

e.save_train_grads("./output/lr_on_binary_mnist_train_grads")
e.save_train_ihvps("./output/lr_on_binary_mnist_train_ihvps")
e.save_train_feats("./output/lr_on_binary_mnist_train_feats")
e.save_alpha_vals("./output/lr_on_binary_mnist_alpha_vals")

e.save_test_grads("./output/lr_on_binary_mnist_test_grads")
e.save_test_ihvps("./output/lr_on_binary_mnist_test_ihvps")
e.save_test_feats("./output/lr_on_binary_mnist_test_feats")

# Calculate and save final scores.

influence = np.zeros((num_train_points, num_test_points))
theta_relatif = np.zeros((num_train_points, num_test_points))
l_relatif = np.zeros((num_train_points, num_test_points))
representer_values = np.zeros((num_train_points, num_test_points))
grad_cos = np.zeros((num_train_points, num_test_points))

for i in range(num_train_points):
    for j in range(num_test_points):
        influence[i, j] = e.get_influence(i, j)
        theta_relatif[i, j] = e.get_theta_relatif(i, j)
        l_relatif[i, j] = e.get_l_relatif(i, j)
        representer_values[i, j] = e.get_representer_value(i, j)
        grad_cos[i, j] = e.get_grad_cos(i, j)

np.savez_compressed(
    "./output/lr_on_binary_mnist_scores",
    influence=influence,
    theta_relatif=theta_relatif,
    l_relatif=l_relatif,
    representer_values=representer_values,
    grad_cos=grad_cos,
)

# Save trained model predictions.
np.savez_compressed(
    "./output/lr_on_binary_mnist_preds",
    train_preds=model(train_x),
    test_pred=model(test_x),
)
