import unittest

import numpy as np
import tensorflow as tf

from influence.influence_model import InfluenceModel

class ConstantModelTestCase(unittest.TestCase):

    def setUp(self):

        class ConstantModel(tf.Module):
            def __init__(self, name=None):
                super(ConstantModel, self)
                self.v1 = tf.Variable(2.)
                self.v2 = tf.Variable(3.)

            def __call__(self, x):
                return self.v1 * self.v1 + self.v2 * self.v2

        def loss_fn(y_true, y_pred):
            return y_pred

        self.influence_model = InfluenceModel(
            ConstantModel(),
            tf.constant([0.]),
            tf.constant([0.]),
            tf.constant([0.]),
            tf.constant([0.]),
            loss_fn
        )

    def test_get_hvp(self):
        self.assertAlmostEqual(self.influence_model.get_hvp([1., 1.])[0], 2.)
        self.assertAlmostEqual(self.influence_model.get_hvp([1., 1.])[1], 2.)
        pass

    def test_training_gradient(self):
        self.assertAlmostEqual(self.influence_model.get_training_gradient()[0], 4.)
        self.assertAlmostEqual(self.influence_model.get_training_gradient()[1], 6.)
        pass

if __name__ == '__main__':
    unittest.main()