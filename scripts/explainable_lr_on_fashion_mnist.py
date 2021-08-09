import numpy as np
import tensorflow as tf

from explainable_model import ExplainableModel

# Using Fashion MNIST dataset.
dataset = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

# Using a subset of training and test points.
num_train_points = 6000
num_test_points = 1000

# Normalise image values.
train_x = train_images[:num_train_points] / 255.0
test_x = test_images[:num_test_points] / 255.0

# Outputs are in categorical form.
train_y = tf.keras.utils.to_categorical(train_labels[:num_train_points], dtype="float64")
test_y = tf.keras.utils.to_categorical(test_labels[:num_test_points], dtype="float64")

# float64 gives better CG speed.
tf.keras.backend.set_floatx("float64")

# Simple LR model.
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

model.fit(train_x, train_y, epochs=100, shuffle=False)

model.save_weights("./output/lr_on_fashion_mnist")

# Instantiate ExplainableModel
xmodel = ExplainableModel(
    model=model,
    feature_model=(lambda x: x),
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    loss_fn=model.loss,
    l2=0.01,
    ihvp_method="cg",
    ihvp_damping=0.01,
)

# Calculate and saves all intermediate values.
for i in range(num_train_points):
    print(
        "Calculating values for training point {} out of {}.".format(
            i, num_train_points
        )
    )

    xmodel.get_train_grad(i)
    xmodel.get_train_ihvp(i)
    xmodel.get_train_feat(i)
    xmodel.get_alpha_val(i)

for j in range(num_test_points):
    print("Calculating values for test point {} out of {}.".format(j, num_test_points))

    xmodel.get_test_grad(j)
    xmodel.get_test_ihvp(j)
    xmodel.get_test_feat(j)

xmodel.save_train_grads("./output/lr_on_fashion_mnist_train_grads")
xmodel.save_train_ihvps("./output/lr_on_fashion_mnist_train_ihvps")
xmodel.save_train_feats("./output/lr_on_fashion_mnist_train_feats")
xmodel.save_alpha_vals("./output/lr_on_fashion_mnist_alpha_vals")

xmodel.save_test_grads("./output/lr_on_fashion_mnist_test_grads")
xmodel.save_test_ihvps("./output/lr_on_fashion_mnist_test_ihvps")
xmodel.save_test_feats("./output/lr_on_fashion_mnist_test_feats")

# Calculate and save final scores.
influence = np.zeros((num_train_points, num_test_points))
theta_relatif = np.zeros((num_train_points, num_test_points))
l_relatif = np.zeros((num_train_points, num_test_points))
representer_values = np.zeros((num_train_points, num_test_points, 10))
grad_cos = np.zeros((num_train_points, num_test_points))

for i in range(num_train_points):
    for j in range(num_test_points):
        influence[i, j] = xmodel.get_influence(i, j)
        theta_relatif[i, j] = xmodel.get_theta_relatif(i, j)
        l_relatif[i, j] = xmodel.get_l_relatif(i, j)
        representer_values[i, j] = xmodel.get_representer_value(i, j)
        grad_cos[i, j] = xmodel.get_grad_cos(i, j)

np.savez_compressed(
    "./output/lr_on_fashion_mnist_scores",
    influence=influence,
    theta_relatif=theta_relatif,
    l_relatif=l_relatif,
    representer_values=representer_values,
    grad_cos=grad_cos,
)

# Save trained model predictions.
np.savez_compressed(
    "./output/lr_on_fashion_mnist_preds",
    train_preds=model(train_x),
    test_preds=model(test_x),
)
