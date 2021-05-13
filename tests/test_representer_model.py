import unittest
import math

import numpy as np
import tensorflow as tf

from representer.representer_model import RepresenterModel


class ConstantModelTestCase(unittest.TestCase):
    def setUp(self):
        class ConstantModel(tf.Module):
            def __init__(self, name=None):
                super(ConstantModel, self)

            def __call__(self, x):
                return tf.constant(3.0)

        def loss_fn(y_true, y_pred):
            return y_pred

        self.representer_model = RepresenterModel(
            ConstantModel(),
            lambda x : x,
            tf.constant([7.0]),
            tf.constant([11.0]),
            tf.constant([13.0]),
            loss_fn
        )

    def test_get_alpha_value(self):
        self.assertAlmostEqual(self.representer_model.get_alpha_value(0).numpy(), -50.0)
        pass

    def test_get_representer_value(self):
        self.assertAlmostEqual(self.representer_model.get_representer_value(0, 0).numpy(), -450.0)
        pass

class IdentityModelTestCase(unittest.TestCase):
    def setUp(self):
        class IdentityModel(tf.Module):
            def __init__(self, name=None):
                super(IdentityModel, self)

            def __call__(self, x):
                return tf.constant(x)

        def loss_fn(y_true, y_pred):
            return y_pred

        self.representer_model = RepresenterModel(
            IdentityModel(),
            lambda x : x,
            tf.constant([7.0]),
            tf.constant([11.0]),
            tf.constant([13.0]),
            loss_fn
        )

    def test_get_alpha_value(self):
        self.assertAlmostEqual(self.representer_model.get_alpha_value(0).numpy(), -50.0)
        pass

    def test_get_representer_value(self):
        self.assertAlmostEqual(self.representer_model.get_representer_value(0, 0).numpy(), -4550.0)
        pass

if __name__ == "__main__":
    unittest.main()
