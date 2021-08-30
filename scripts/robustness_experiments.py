import numpy as np
import tensorflow as tf

from explainable_model import ExplainableModel

seed = 42
num_train_points = 1000
num_test_points = 200

# Generate synthetic data.
mu0, sigma0 = [3, 3], [[2, 1], [1, 2]] # Class 0
mu1, sigma1 = [-2,-2], [[1, 0], [0, 1]] # Class 1

small_noise = 0.25
large_noise = 1.0

rng = np.random.default_rng(seed)

train_x0 = rng.multivariate_normal(mu0, sigma0, size=num_train_points//2)
train_x1 = rng.multivariate_normal(mu1, sigma1, size=num_train_points//2)
train_perm = rng.permutation(num_train_points)

train_x = np.append(train_x0, train_x1, axis=0)[train_perm]
train_y = np.append(np.zeros(num_train_points//2), np.ones(num_train_points//2))[train_perm, np.newaxis]

test_x0 = rng.multivariate_normal(mu0, sigma0, size=num_test_points//2)
test_x1 = rng.multivariate_normal(mu1, sigma1, size=num_test_points//2)
test_perm = rng.permutation(num_test_points)

test_x = np.append(test_x0, test_x1, axis=0)[test_perm]
test_y = np.append(np.zeros(num_test_points//2), np.ones(num_test_points//2))[test_perm, np.newaxis]

small_noisy_x = train_x + rng.multivariate_normal([0, 0], small_noise*np.eye(2), size=num_train_points)
large_noisy_x = train_x + rng.multivariate_normal([0, 0], large_noise*np.eye(2), size=num_train_points)

np.savez(
    "./output/synthetic_data",
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    small_noisy_x=small_noisy_x,
    large_noisy_x=large_noisy_x,
)

# Train models and extract explanations.

# float64 gives better CG speed.
tf.keras.backend.set_floatx("float64")

train_xs = [train_x, small_noisy_x, large_noisy_x]
filenames = [
    "synthetic_baseline",
    "synthetic_small_noisy",
    "synthetic_large_noisy"
]

for k in range(3):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(2,)),
            tf.keras.layers.Dense(1, kernel_regularizer="l2", use_bias=False, kernel_initializer="zeros"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(train_xs[k], train_y, epochs=100, shuffle=False)

    xmodel = ExplainableModel(
        model=model,
        feature_model=(lambda x: x),
        train_x=train_xs[k],
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        loss_fn=model.loss,
        l2=0.01,
        ihvp_method="cg",
        ihvp_damping=0.01,
    )

    influence = np.zeros((num_train_points, num_test_points))
    theta_relatif = np.zeros((num_train_points, num_test_points))
    l_relatif = np.zeros((num_train_points, num_test_points))
    representer_values = np.zeros((num_train_points, num_test_points))
    grad_cos = np.zeros((num_train_points, num_test_points))

    for i in range(num_train_points):
        for j in range(num_test_points):
            influence[i, j] = xmodel.get_influence(i, j)
            theta_relatif[i, j] = xmodel.get_theta_relatif(i, j)
            l_relatif[i, j] = xmodel.get_l_relatif(i, j)
            representer_values[i, j] = xmodel.get_representer_value(i, j)
            grad_cos[i, j] = xmodel.get_grad_cos(i, j)

    np.savez_compressed(
        "./output/{}_scores".format(filenames[k]),
        influence=influence,
        theta_relatif=theta_relatif,
        l_relatif=l_relatif,
        representer_values=representer_values,
        grad_cos=grad_cos,
    )

    np.savez_compressed(
        "./output/{}_preds".format(filenames[k]),
        train_preds=model(train_xs[k]),
        test_preds=model(test_x),
    )
