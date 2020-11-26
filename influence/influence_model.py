import math

import numpy as np
import scipy.optimize
import tensorflow as tf


class InfluenceModel(object):
    """Class representing a TensorFlow model with some up-weighted training point."""

    # Maybe extend to up-weighted sets of points?
    # Maybe allow specification of parameters?

    def __init__(
        self,
        model,
        training_inputs,
        training_labels,
        loss_fn,
        upweighted_training_idx,
        parameters=None,
        scaling=1.0,
        damping=0.0,
        verbose=False,
        dtype=np.float32,
        gtol=1e-05,
    ):
        self.model = model
        self.training_inputs = training_inputs
        self.training_labels = training_labels
        self.loss_fn = loss_fn

        self.upweighted_training_input = training_inputs[upweighted_training_idx]
        self.upweighted_training_label = training_labels[upweighted_training_idx]

        if parameters is None:
            self.parameters = model.trainable_variables
        else:
            self.parameters = parameters

        self.scaling = scaling
        self.damping = damping
        self.verbose = verbose
        self.dtype = dtype
        self.gtol = gtol

        self.training_gradient = None
        self.inverse_hvp = None

    def reshape_flat_vector(self, flat_vector):
        """Takes a flat vector and reshapes it to a tensor with the same shape as the model's trainable variables."""

        # Check the overall lengths match.
        length = np.sum([len(tf.reshape(t, [-1])) for t in self.parameters])
        if len(flat_vector) != length:
            raise ValueError(
                "Length of flat vector is "
                + len(flat_vector)
                + ", while model has "
                + length
                + " trainable variables."
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

    def get_training_gradient(self):
        """Calculates the gradient of loss at up-weighted training point w.r.t. trainable variables."""

        if self.training_gradient is not None:
            return self.training_gradient

        with tf.GradientTape() as tape:
            predicted_label = self.model(np.array([self.upweighted_training_input]))
            loss = (
                self.loss_fn(
                    np.array([self.upweighted_training_label]), predicted_label
                )
                * self.scaling
            )

        training_gradient = tape.gradient(
            loss,
            self.parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        self.training_gradient = training_gradient

        return training_gradient

    def get_inverse_hvp(self):
        """Calculates the inverse HVP using Conjugate Gradient method."""
        # Can use other methods in future?

        if self.inverse_hvp is not None:
            return self.inverse_hvp

        # Flattened training gradient used both for iteration, and as initial guess.
        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in self.get_training_gradient()]
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
                print(
                    "CG Loss: ",
                    cg_loss_fn(xk),
                    "; CG Jac Norm:",
                    np.linalg.norm(cg_jac_fn(xk)),
                )
                return

            cg_callback = verbose_cg_callback
            print("Calculating Inverse HVP:")

        result = scipy.optimize.minimize(
            cg_loss_fn,
            flat_training_gradient,
            method="CG",
            jac=cg_jac_fn,
            callback=cg_callback,
            options={"gtol": self.gtol, "maxiter": 100, "disp": self.verbose},
        )

        self.inverse_hvp = result.x

        return result.x

    def get_test_gradient(self, test_input, test_label):
        """Calculates the gradient of loss at a test point w.r.t. trainable variables."""

        with tf.GradientTape() as tape:
            predicted_label = self.model(np.array([test_input]))
            loss = self.loss_fn(np.array([test_label]), predicted_label)

        test_gradient = tape.gradient(
            loss,
            self.parameters,
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

        return test_gradient

    def get_influence_on_loss(self, test_input, test_label):
        """Calculates the influence of the up-weighted training point on the loss at the given test point."""

        test_gradient = self.get_test_gradient(test_input, test_label)
        flat_test_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in test_gradient]
        )

        influence_on_loss = np.dot(self.get_inverse_hvp(), flat_test_gradient)

        return influence_on_loss

    def get_theta_relatif(self, test_input, test_label):
        """Calculates the theta-relative influence of the up-weighted training point on the loss at the given test point."""

        test_gradient = self.get_test_gradient(test_input, test_label)
        flat_test_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in test_gradient]
        )

        influence_on_loss = np.dot(self.get_inverse_hvp(), flat_test_gradient)

        theta_relatif = influence_on_loss / np.linalg.norm(self.get_inverse_hvp())

        return theta_relatif

    def get_l_relatif(self, test_input, test_label):
        """Calculates the l-relative influence of the up-weighted training point on the loss at the given test point."""

        test_gradient = self.get_test_gradient(test_input, test_label)
        flat_test_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in test_gradient]
        )

        influence_on_loss = np.dot(self.get_inverse_hvp(), flat_test_gradient)

        flat_training_gradient = np.concatenate(
            [tf.reshape(t, [-1]) for t in self.get_training_gradient()]
        )
        l_relatif = influence_on_loss / math.sqrt(
            np.dot(self.get_inverse_hvp(), flat_training_gradient)
        )

        return l_relatif

    def get_new_parameters(self, epsilon=None):
        """Calculates the approximated new parameters with training point up-weighted by epsilon."""

        # By default, we use epsilon = -1/n, which is equivalent to leave-one-out retraining.
        if epsilon is None:
            epsilon = -1.0 / len(self.training_inputs)

        flat_change_in_parameters = self.get_inverse_hvp() * epsilon
        flat_parameters = np.concatenate([tf.reshape(t, [-1]) for t in self.parameters])

        flat_new_parameters = flat_change_in_parameters + flat_parameters
        new_parameters = self.reshape_flat_vector(flat_new_parameters)

        return new_parameters
