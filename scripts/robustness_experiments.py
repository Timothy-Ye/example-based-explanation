import numpy as np
import tensorflow as tf

from explainable_model import ExplainableModel

seed = 42
num_train_points = 500
num_test_points = 100
num_runs = 10

# Generate synthetic data.
mu0, sigma0 = [3, 3], [[2, 1], [1, 2]] # Class 0
mu1, sigma1 = [-2,-2], [[1, 0], [0, 1]] # Class 1

low_noise = 0.25
high_noise = 1.0

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

low_noise_xs = np.array([train_x + rng.multivariate_normal([0, 0], low_noise*np.eye(2), size=num_train_points) for _ in range(num_runs)])
high_noise_xs = np.array([train_x + rng.multivariate_normal([0, 0], high_noise*np.eye(2), size=num_train_points) for _ in range(num_runs)])

np.savez(
    "./output/synthetic_data",
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
    low_noise_xs=low_noise_xs,
    high_noise_xs=high_noise_xs,
)

# Train models and extract explanations.

# float64 gives better CG speed.
tf.keras.backend.set_floatx("float64")

def run_experiment(run_train_x):

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

    model.fit(run_train_x, train_y, epochs=100, shuffle=False)

    xmodel = ExplainableModel(
        model=model,
        feature_model=(lambda x: x),
        train_x=run_train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        loss_fn=model.loss,
        l2=0.01,
        ihvp_method="cg",
        ihvp_damping=0.01,
    )

    influence = np.zeros((num_train_points, num_test_points))
    relatif = np.zeros((num_train_points, num_test_points))
    representer_values = np.zeros((num_train_points, num_test_points))
    grad_cos = np.zeros((num_train_points, num_test_points))

    for i in range(num_train_points):
        for j in range(num_test_points):
            influence[i, j] = xmodel.get_influence(i, j)
            relatif[i, j] = xmodel.get_theta_relatif(i, j)
            representer_values[i, j] = xmodel.get_representer_value(i, j)
            grad_cos[i, j] = xmodel.get_grad_cos(i, j)

    train_preds = model(run_train_x)
    test_preds = model(test_x)

    return influence, relatif, representer_values, grad_cos, train_preds, test_preds

# Baseline model and explanations.

print("Running baseline experiments.")

influence, relatif, representer_values, grad_cos, train_preds, test_preds = run_experiment(train_x)

np.savez_compressed(
    "./output/synthetic_baseline_scores",
    influence=influence,
    relatif=relatif,
    representer_values=representer_values,
    grad_cos=grad_cos,
    train_preds=train_preds,
    test_preds=test_preds
)

# Noisy models and explanations.

low_noise_influence = np.zeros((num_runs, num_train_points, num_test_points))
low_noise_relatif = np.zeros((num_runs, num_train_points, num_test_points))
low_noise_representer_values = np.zeros((num_runs, num_train_points, num_test_points))
low_noise_grad_cos = np.zeros((num_runs, num_train_points, num_test_points))
low_noise_train_preds = np.zeros((num_runs, num_train_points, 1))
low_noise_test_preds = np.zeros((num_runs, num_test_points, 1))

high_noise_influence = np.zeros((num_runs, num_train_points, num_test_points))
high_noise_relatif = np.zeros((num_runs, num_train_points, num_test_points))
high_noise_representer_values = np.zeros((num_runs, num_train_points, num_test_points))
high_noise_grad_cos = np.zeros((num_runs, num_train_points, num_test_points))
high_noise_train_preds = np.zeros((num_runs, num_train_points, 1))
high_noise_test_preds = np.zeros((num_runs, num_test_points, 1))

for k in range(num_runs):

    print("Run {} of noisy experiments.".format(k))

    influence, relatif, representer_values, grad_cos, train_preds, test_preds = run_experiment(low_noise_xs[k])

    low_noise_influence[k] = influence
    low_noise_relatif[k] = relatif
    low_noise_representer_values[k] = representer_values
    low_noise_grad_cos[k] = grad_cos
    low_noise_train_preds[k] = train_preds
    low_noise_test_preds[k] = test_preds

    influence, relatif, representer_values, grad_cos, train_preds, test_preds = run_experiment(high_noise_xs[k])

    high_noise_influence[k] = influence
    high_noise_relatif[k] = relatif
    high_noise_representer_values[k] = representer_values
    high_noise_grad_cos[k] = grad_cos
    high_noise_train_preds[k] = train_preds
    high_noise_test_preds[k] = test_preds


np.savez_compressed(
    "./output/synthetic_low_noise_scores",
    influence=low_noise_influence,
    relatif=low_noise_relatif,
    representer_values=low_noise_representer_values,
    grad_cos=low_noise_grad_cos,
    train_preds=low_noise_train_preds,
    test_preds=low_noise_test_preds
)

np.savez_compressed(
    "./output/synthetic_high_noise_scores",
    influence=high_noise_influence,
    relatif=high_noise_relatif,
    representer_values=high_noise_representer_values,
    grad_cos=high_noise_grad_cos,
    train_preds=high_noise_train_preds,
    test_preds=high_noise_test_preds
)
