import unittest
import math

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel


class ConstantModelTestCase(unittest.TestCase):
    def setUp(self):
        class ConstantModel(tf.Module):
            def __init__(self, name=None):
                super(ConstantModel, self)
                self.v1 = tf.Variable(2.0)
                self.v2 = tf.Variable(3.0)

            def __call__(self, x):
                return self.v1 * self.v1 + self.v2 * self.v2

        def loss_fn(y_true, y_pred):
            return y_pred

        self.influence_model = InfluenceModel(
            ConstantModel(),
            tf.constant([0.0]),
            tf.constant([0.0]),
            tf.constant([0.0]),
            tf.constant([0.0]),
            loss_fn,
        )

    def test_get_hvp(self):
        self.assertAlmostEqual(self.influence_model.get_hvp([1.0, 1.0])[0], 2.0)
        self.assertAlmostEqual(self.influence_model.get_hvp([1.0, 1.0])[1], 2.0)
        pass

    def test_get_training_gradient(self):
        self.assertAlmostEqual(self.influence_model.get_training_gradient(0)[0], 4.0)
        self.assertAlmostEqual(self.influence_model.get_training_gradient(0)[1], 6.0)
        pass

    def test_get_inverse_hvp(self):
        # For float32, there is an error in order 1e-08.
        # This does not occur for float64, so assume it is just some precision limitation.
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[0], 2.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[1], 3.0, places=5
        )
        pass

    def test_get_inverse_hvp_with_lissa(self):
        # Similar precision error as with CG.
        self.influence_model.method = "lissa"
        self.influence_model.scaling = 0.1
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[0], 2.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[1], 3.0, places=5
        )
        pass

    def test_get_test_gradient(self):
        self.assertAlmostEqual(
            self.influence_model.get_test_gradient(0)[0],
            4.0,
        )
        self.assertAlmostEqual(
            self.influence_model.get_test_gradient(0)[1],
            6.0,
        )
        pass

    def test_get_influence_on_loss(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_influence_on_loss(0, 0),
            -26.0,
            places=5,
        )
        pass

    def test_get_theta_relatif(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_theta_relatif(0, 0),
            -26.0 / math.sqrt(13.0),
            places=5,
        )
        pass

    def test_get_l_relatif(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_l_relatif(0, 0),
            -26.0 / math.sqrt(26.0),
            places=5,
        )
        pass

    def test_get_new_parameters(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_new_parameters(0)[0].numpy(), 0.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_new_parameters(0)[1].numpy(), 0.0, places=5
        )
        pass


class HighPrecisionTestCase(unittest.TestCase):
    def setUp(self):
        class ConstantModel(tf.Module):
            def __init__(self, name=None):
                super(ConstantModel, self)
                self.v1 = tf.Variable(2.0, dtype=tf.float64)
                self.v2 = tf.Variable(3.0, dtype=tf.float64)

            def __call__(self, x):
                return self.v1 * self.v1 + self.v2 * self.v2

        def loss_fn(y_true, y_pred):
            return y_pred

        self.influence_model = InfluenceModel(
            ConstantModel(),
            tf.constant([0.0], dtype=tf.float64),
            tf.constant([0.0], dtype=tf.float64),
            tf.constant([0.0]),
            tf.constant([0.0]),
            loss_fn,
            dtype=np.float64,
        )

    def test_get_inverse_hvp(self):
        self.assertAlmostEqual(self.influence_model.get_inverse_hvp(0)[0], 2.0)
        self.assertAlmostEqual(self.influence_model.get_inverse_hvp(0)[1], 3.0)
        pass

    def test_get_inverse_hvp_with_lissa(self):
        self.influence_model.method = "lissa"
        self.influence_model.scaling = 0.1
        self.assertAlmostEqual(self.influence_model.get_inverse_hvp(0)[0], 2.0)
        self.assertAlmostEqual(self.influence_model.get_inverse_hvp(0)[1], 3.0)
        pass

class LinearModelTestCase(unittest.TestCase):
    def setUp(self):
        class LinearModel(tf.Module):
            def __init__(self, name=None):
                super(LinearModel, self)
                self.v1 = tf.Variable(2.0)
                self.v2 = tf.Variable(3.0)

            def __call__(self, x):
                return (self.v1 * self.v1 + self.v2 * self.v2) * x

        def loss_fn(y_true, y_pred):
            return y_pred

        self.influence_model = InfluenceModel(
            LinearModel(),
            tf.constant([5.0]),
            tf.constant([0.0]),
            tf.constant([7.0]),
            tf.constant([0.0]),
            loss_fn,
        )

    def test_get_hvp(self):
        self.assertAlmostEqual(self.influence_model.get_hvp([1.0, 1.0])[0], 10.0)
        self.assertAlmostEqual(self.influence_model.get_hvp([1.0, 1.0])[1], 10.0)
        pass

    def test_get_training_gradient(self):
        self.assertAlmostEqual(self.influence_model.get_training_gradient(0)[0], 20.0)
        self.assertAlmostEqual(self.influence_model.get_training_gradient(0)[1], 30.0)
        pass

    def test_get_inverse_hvp(self):
        # For float32, there is an error in order 1e-08.
        # This does not occur for float64, so assume it is just some precision limitation.
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[0], 2.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[1], 3.0, places=5
        )
        pass

    def test_get_inverse_hvp_with_lissa(self):
        # Similar precision error as with CG.
        self.influence_model.method = "lissa"
        self.influence_model.scaling = 0.1
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[0], 2.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_inverse_hvp(0)[1], 3.0, places=5
        )
        pass

    def test_get_test_gradient(self):
        self.assertAlmostEqual(
            self.influence_model.get_test_gradient(0)[0],
            28.0,
        )
        self.assertAlmostEqual(
            self.influence_model.get_test_gradient(0)[1],
            42.0,
        )
        pass

    def test_get_influence_on_loss(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_influence_on_loss(0, 0),
            -182.0,
            places=4,
        )
        pass

    def test_get_theta_relatif(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_theta_relatif(0, 0),
            -182.0 / math.sqrt(13.0),
            places=5,
        )
        pass

    def test_get_l_relatif(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_l_relatif(0, 0),
            -182.0 / math.sqrt(130.0),
            places=5,
        )
        pass

    def test_get_new_parameters(self):
        # Carry-over precision loss from get_inverse_hvp().
        self.assertAlmostEqual(
            self.influence_model.get_new_parameters(0)[0].numpy(), 0.0, places=5
        )
        self.assertAlmostEqual(
            self.influence_model.get_new_parameters(0)[1].numpy(), 0.0, places=5
        )
        pass

if __name__ == "__main__":
    unittest.main()
