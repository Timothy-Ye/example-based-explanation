import numpy as np
import tensorflow as tf

def get_shapley_values(
    model,
    num_training_points,
    performance_metric,
    model_reset_fn,
    model_train_fn,
    max_iters=1000,
    truncate_threshold=0.0,
    convergence_threshold=0.05,
    mrae_gap=100,
    verbose=False,
):
    """
    Calculates the Data Shapley values for all training points.
    """

    converged = False

    rng = np.random.default_rng()
    shapley_values = np.zeros(num_training_points)
    cached_shapley = np.zeros((mrae_gap, num_training_points))
    num_iters = 0

    model_reset_fn(model)
    model_train_fn(model, np.arange(num_training_points))
    full_performance = performance_metric(model)

    if verbose:
        print("Fully trained performance: {:.4f}".format(full_performance))

    for t in range(max_iters):

        num_iters += 1
        perm = rng.permutation(num_training_points)
        points_used = 0
        mrae = np.nan

        model_reset_fn(model)
        old_performance = performance_metric(model)

        for i in range(num_training_points):

            if np.abs(full_performance - old_performance) < truncate_threshold:
                points_used = i
                break

            model_train_fn(model, perm[i:i+1])
            new_performance = performance_metric(model)

            shapley_values[perm[i]] += new_performance - old_performance
            old_performance = new_performance

        if num_iters > mrae_gap and np.all(shapley_values != 0):
            mrae = np.mean(
                np.abs(
                    (shapley_values / num_iters) - cached_shapley[num_iters % mrae_gap]
                ) / np.abs(shapley_values / num_iters)
            )

            if mrae < convergence_threshold:
                converged = True
                break

        cached_shapley[num_iters % mrae_gap] = shapley_values / num_iters

        if verbose:
            print(
                "Iteration {}: Training Points Sampled={}, Performance={:.4f}, MRAE={:.4f}".format(
                    num_iters,
                    points_used,
                    old_performance,
                    mrae
                )
            )

    shapley_values = shapley_values / num_iters

    if verbose:
        print(
            "Calculation Complete, Convergence Achieved={}, Number of Iterations={}".format(
                converged,
                num_iters,
            )
        )

    return shapley_values
