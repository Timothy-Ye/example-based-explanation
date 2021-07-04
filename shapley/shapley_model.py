import time

import numpy as np
import tensorflow as tf

class ShapleyModel(object):
    """Class representing a TensorFlow model on which data Shapley can be applied."""

    def __init__(
        self,
        model,
        num_training_points,
        performance_metric,
        model_reset_fn,
        model_train_fn,
        max_iters,
        truncate_threshold=0.0,
        convergence_threshold=0.05,
        mrae_gap=100,
        verbose=False,
    ):
        self.model = model
        self.num_training_points = num_training_points
        self.performance_metric = performance_metric
        self.model_reset_fn = model_reset_fn
        self.model_train_fn = model_train_fn
        self.max_iters = max_iters
        self.truncate_threshold = truncate_threshold
        self.convergence_threshold = convergence_threshold
        self.mrae_gap = mrae_gap
        self.verbose = verbose

    def get_shapley_values(self):
        """Calculates the Data Shapley values for all training points."""

        start_time = time.time()
        converged = False

        rng = np.random.default_rng()
        shapley_values = np.zeros(self.num_training_points)
        cached_shapley = np.zeros((self.mrae_gap, self.num_training_points))
        num_iters = 0

        self.model_reset_fn(self.model)
        self.model_train_fn(self.model, np.arange(self.num_training_points))
        full_performance = self.performance_metric(self.model)

        if self.verbose:
            print("Fully trained performance: {:.4f}".format(full_performance))

        for t in range(self.max_iters):

            num_iters += 1
            perm = rng.permutation(self.num_training_points)
            points_used = 0
            mrae = np.nan

            self.model_reset_fn(self.model)
            old_performance = self.performance_metric(self.model)

            for i in range(self.num_training_points):

                if np.abs(full_performance - old_performance) < self.truncate_threshold:
                    points_used = i
                    break

                self.model_train_fn(self.model, perm[i:i+1])
                new_performance = self.performance_metric(self.model)

                shapley_values[perm[i]] += new_performance - old_performance
                old_performance = new_performance

            if num_iters > self.mrae_gap and np.all(shapley_values != 0):
                mrae = np.mean(
                    np.abs(
                        (shapley_values / num_iters) - cached_shapley[num_iters % self.mrae_gap]
                    ) / np.abs(shapley_values / num_iters)
                )

                if mrae < self.convergence_threshold:
                    converged = True
                    break

            cached_shapley[num_iters % self.mrae_gap] = shapley_values / num_iters

            if self.verbose:
                print(
                    "Iteration {}: Training Points Sampled={}, Performance={:.4f}, MRAE={:.4f}".format(
                        num_iters,
                        points_used,
                        old_performance,
                        mrae
                    )
                )

        shapley_values = shapley_values / num_iters

        if self.verbose:
            end_time = time.time()
            print(
                "Calculation Complete, Convergence Achieved={}, Number of Iterations={}, Time Taken={:.3f}s".format(
                    converged,
                    num_iters,
                    end_time - start_time
                )
            )

        return shapley_values
