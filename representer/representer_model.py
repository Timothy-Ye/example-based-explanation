import numpy as np
import tensorflow as tf


class RepresenterModel(object):
    """Class representing a TensorFlow model which uses representer point theorem."""

    def __init__(
        self,
        feature_model,
        prediction_network,
        training_inputs,
        training_labels,
        test_inputs,
        loss_fn,
        l2=0.01,
        num_training_points=None,
    ):
        self.feature_model = feature_model
        self.prediction_network = prediction_network
        self.training_inputs = training_inputs
        self.training_labels = training_labels
        self.test_inputs = test_inputs
        self.loss_fn = loss_fn
        self.l2 = l2

        if num_training_points is None:
            self.num_training_points = len(training_inputs)
        else:
            self.num_training_points = num_training_points

        if len(training_inputs) != len(training_labels):
            raise ValueError(
                "Training inputs and labels have different lengths: Inputs = "
                + len(training_inputs)
                + ", Labels = "
                + len(training_labels)
            )

        self.alpha_values = {}

    def get_alpha_value(self, training_idx):
        """Calcuates the alpha value for a given training point."""

        if training_idx in self.alpha_values:
            return self.alpha_values[training_idx]

        with tf.GradientTape() as tape:
            predicted_label = self.prediction_network(
                self.feature_model(np.array([self.training_inputs[training_idx]]))
            )
            loss = self.loss = self.loss_fn(
                np.array([self.training_labels[training_idx]]), predicted_label
            )

        gradient = tape.gradient(
            loss, predicted_label, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        alpha_value = -gradient / (2 * self.l2 * self.num_training_points)

        self.alpha_values[training_idx] = alpha_value

        return alpha_value

    def get_representer_value(self, training_idx, test_idx):
        """Calculates the representer value for a training point given a test point."""

        training_features = self.feature_model(
            np.array([self.training_inputs[training_idx]])
        )
        test_features = self.feature_model(np.array([self.test_inputs[test_idx]]))

        representer_value = tf.math.reduce_sum(
            tf.math.multiply(training_features, test_features)
        ) * self.get_alpha_value(training_idx)

        return representer_value
