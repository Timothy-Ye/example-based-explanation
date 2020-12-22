import math

import numpy as np
import scipy.optimize
import tensorflow as tf


class InfluenceModel(object):
    """Class representing a TensorFlow model."""

    # Maybe extend to up-weighted sets of points?

    def __init__(
        self,
        model,
        training_inputs,
        training_labels,
        test_inputs,
        test_labels,
        loss_fn,
        parameters=None,
        scaling=1.0,
        damping=0.0,
        verbose=False,
        dtype=np.float32,
        method="cg",
        cg_tol=1e-05,
        lissa_samples=1,
        lissa_depth=1000,
    ):
        self.model = model
        self.training_inputs = training_inputs
        self.training_labels = training_labels
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.loss_fn = loss_fn

        if parameters is None:
            self.parameters = model.trainable_variables
        else:
            self.parameters = parameters

        self.scaling = scaling
        self.damping = damping
        self.verbose = verbose
        self.dtype = dtype
        self.method = method
        self.cg_tol = cg_tol
        self.lissa_samples = lissa_samples
        self.lissa_depth = lissa_depth

        if len(training_inputs) != len(training_labels):
            raise ValueError(
                "Training inputs and labels have different lengths: Inputs = "
                + len(training_inputs)
                + ", Labels = "
                + len(training_labels)
            )
        if len(test_inputs) != len(test_labels):
            raise ValueError(
                "Test inputs and labels have different lengths: Inputs = "
                + len(test_inputs)
                + ", Labels = "
                + len(test_labels)
            )

        self.training_gradients = {}
        self.inverse_hvps = {}
        self.test_gradients = {}
        self.influences_on_loss = {}

    def reshape_flat_vector(self, flat_vector):
        """Takes a flat vector and reshapes it to a tensor with the same shape as the model's trainable variables."""

        # Check the overall lengths match.
        length = np.sum([len(tf.reshape(t, [-1])) for t in self.parameters])
        if len(flat_vector) != length:
            raise ValueError(
                "Flat vector and parameters have different lengths: Flat vector = "
                + len(flat_vector)
                + ", Parameters = "
                + length
            )

        # Reshape flat_vector.
        reshaped_flat_vector = []
        i = 0
        for t in self.parameters:
            var_length = len(tf.reshape(t, [-1]))
            reshaped_flat_vector.append(
                tf.reshape(flat_vector[i : i + var_length], tf.shape(t))
            )
            i += var_length

        return reshaped_flat_vector

    def get_hvp(self, vector):
        """Calculates the product of the Hessian of the loss over all training points w.r.t. trainable variables, with an input vector."""

        # Calculate HVP using back-over-back auto-diff.
        with tf.GradientTape() as outer_tape:
            with tf.GradientTape() as inner_tape:
                predicted_labels = self.model(self.training_inputs)
                loss = (
                    self.loss_fn(self.training_labels, predicted_labels) * self.scaling
                )

            grads = inner_tape.gradient(
                loss,
                self.parameters,
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

        hvp = outer_tape.gradient(
            grads,
            self.parameters,
            output_gradients=vector,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        return hvp

    def get_training_gradient(self, training_idx):
        """Calculates the gradient of loss at an up-weighted training point w.r.t. trainable variables."""

        if training_idx in self.training_gradients:
            return self.training_gradients[training_idx]

        with tf.GradientTape() as tape:
            predicted_label = self.model(np.array([self.training_inputs[training_idx]]))
            loss = (
                self.loss_fn(
                    np.array([self.training_labels[training_idx]]), predicted_label
                )
                * self.scaling
            )

        training_gradient = tape.gradient(
            loss,
            self.parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        self.training_gradients[training_idx] = training_gradient

        return training_gradient

    def get_inverse_hvp_cg(self, training_idx):
        """Calculates the inverse HVP using Conjugate Gradient method."""

        # Flattened training gradient used both for iteration, and as initial guess.
        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in self.get_training_gradient(training_idx)]
        )

        def cg_loss_fn(x):

            # Need to reshape vector before passing into get_hvp().
            reshaped_vector = self.reshape_flat_vector(x.astype(self.dtype))

            hvp = self.get_hvp(reshaped_vector)

            # Easier to flatten tensors and just use np.dot().
            flat_hvp = (
                np.concatenate([tf.reshape(t, [-1]) for t in hvp]) + self.damping * x
            )

            return 0.5 * np.dot(flat_hvp, x) - np.dot(flat_training_gradient, x)

        def cg_jac_fn(x):

            # Need to reshape vector before passing into get_hvp().
            reshaped_vector = self.reshape_flat_vector(x.astype(self.dtype))

            hvp = self.get_hvp(reshaped_vector)
            flat_hvp = (
                np.concatenate([tf.reshape(t, [-1]) for t in hvp]) + self.damping * x
            )

            return flat_hvp - flat_training_gradient

        cg_callback = None
        if self.verbose:

            def verbose_cg_callback(xk):
                distance = np.linalg.norm(cg_jac_fn(xk))
                print(
                    "Current error:",
                    distance,
                    ", Relative error:",
                    distance / np.linalg.norm(flat_training_gradient),
                )
                return

            cg_callback = verbose_cg_callback
            print("Calculating inverse HVP using Conjugate Gradient method:")

        result = scipy.optimize.minimize(
            cg_loss_fn,
            flat_training_gradient,
            method="CG",
            jac=cg_jac_fn,
            callback=cg_callback,
            options={"gtol": self.cg_tol, "maxiter": 100, "disp": self.verbose},
        )

        return result.x

    def get_inverse_hvp_lissa(self, training_idx):
        """Approximates the inverse HVP using LiSSA method."""

        if self.verbose:
            print("Calculating inverse HVP using LiSSA method:")

        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in self.get_training_gradient(training_idx)]
        )

        estimates = []

        for i in range(self.lissa_samples):
            current_estimate = self.get_training_gradient(training_idx)

            for j in range(self.lissa_depth):
                sample_idx = np.random.choice(range(len(self.training_inputs)))

                # Calculate HVP using back-over-back auto-diff.
                with tf.GradientTape() as outer_tape:
                    with tf.GradientTape() as inner_tape:
                        predicted_label = self.model(
                            np.array([self.training_inputs[sample_idx]])
                        )
                        loss = (
                            self.loss_fn(
                                np.array([self.training_labels[sample_idx]]),
                                predicted_label,
                            )
                            * self.scaling
                        )

                    grads = inner_tape.gradient(
                        loss,
                        self.parameters,
                        unconnected_gradients=tf.UnconnectedGradients.ZERO,
                    )

                hvp = outer_tape.gradient(
                    grads,
                    self.parameters,
                    output_gradients=current_estimate,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )

                # Form new estimate recursively.
                current_estimate = [
                    tf.subtract(
                        tf.add(
                            self.get_training_gradient(training_idx)[k],
                            tf.scalar_mul(1 - self.damping, current_estimate[k]),
                        ),
                        hvp[k],
                    )
                    for k in range(len(self.parameters))
                ]

            if self.verbose:
                current_hvp = self.get_hvp(current_estimate)
                flat_hvp = np.concatenate([tf.reshape(t, [-1]) for t in current_hvp])
                distance = np.linalg.norm(flat_hvp - flat_training_gradient)
                print(
                    "Sample",
                    i + 1,
                    "with depth",
                    self.lissa_depth,
                    "- Current error:",
                    distance,
                    ", Relative error:",
                    distance / np.linalg.norm(flat_training_gradient),
                )

            estimates.append(
                np.concatenate([tf.reshape(t, [-1]) for t in current_estimate])
            )

        inverse_hvp = np.mean(estimates, axis=0)

        if self.verbose:
            current_hvp = self.get_hvp(self.reshape_flat_vector(inverse_hvp))
            flat_hvp = np.concatenate([tf.reshape(t, [-1]) for t in current_hvp])
            distance = np.linalg.norm(flat_hvp - flat_training_gradient)
            print(
                "Overall error:",
                distance,
                ", Overall relative error:",
                distance / np.linalg.norm(flat_training_gradient),
            )

        return inverse_hvp

    def get_inverse_hvp(self, training_idx):
        """Calculates the inverse HVP using the specified method."""

        if training_idx in self.inverse_hvps:
            return self.inverse_hvps[training_idx]

        if self.method == "cg":
            inverse_hvp = self.get_inverse_hvp_cg(training_idx)
        elif self.method == "lissa":
            inverse_hvp = self.get_inverse_hvp_lissa(training_idx)
        else:
            raise ValueError(
                "'"
                + self.method
                + "' is not a supported method of calcuating the inverse HVP."
            )

        self.inverse_hvps[training_idx] = inverse_hvp

        return inverse_hvp

    def get_test_gradient(self, test_idx):
        """Calculates the gradient of loss at a test point w.r.t. trainable variables."""

        if test_idx in self.test_gradients:
            return self.test_gradients[test_idx]

        with tf.GradientTape() as tape:
            predicted_label = self.model(np.array([self.test_inputs[test_idx]]))
            loss = self.loss_fn(np.array([self.test_labels[test_idx]]), predicted_label)

        test_gradient = tape.gradient(
            loss,
            self.parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        self.test_gradients[test_idx] = test_gradient

        return test_gradient

    def get_influence_on_loss(self, training_idx, test_idx):
        """Calculates the influence of the given training point at the given test point."""

        if (training_idx, test_idx) in self.influences_on_loss:
            return self.influences_on_loss[(training_idx, test_idx)]

        test_gradient = self.get_test_gradient(test_idx)
        flat_test_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in test_gradient]
        )
        influence_on_loss = -np.dot(
            self.get_inverse_hvp(training_idx), flat_test_gradient
        )

        self.influences_on_loss[(training_idx, test_idx)] = influence_on_loss

        return influence_on_loss

    def get_theta_relatif(self, training_idx, test_idx):
        """Calculates the theta-relative influence of the given training point at the given test point."""

        influence_on_loss = self.get_influence_on_loss(training_idx, test_idx)
        theta_relatif = influence_on_loss / np.linalg.norm(
            self.get_inverse_hvp(training_idx)
        )

        return theta_relatif

    def get_l_relatif(self, training_idx, test_idx):
        """Calculates the l-relative influence of the given training point at the given test point."""

        influence_on_loss = self.get_influence_on_loss(training_idx, test_idx)
        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in self.get_training_gradient(training_idx)]
        )
        l_relatif = influence_on_loss / math.sqrt(
            np.dot(self.get_inverse_hvp(training_idx), flat_training_gradient)
        )

        return l_relatif

    def get_new_parameters(self, training_idx, epsilon=None):
        """Calculates the approximated new parameters with given training point up-weighted by epsilon."""

        # By default, we use epsilon = -1/n, which is equivalent to leave-one-out retraining.
        if epsilon is None:
            epsilon = -1.0 / len(self.training_inputs)

        flat_change_in_parameters = self.get_inverse_hvp(training_idx) * epsilon
        flat_parameters = np.concatenate([tf.reshape(t, [-1]) for t in self.parameters])

        flat_new_parameters = flat_change_in_parameters + flat_parameters
        new_parameters = self.reshape_flat_vector(flat_new_parameters)

        return new_parameters
