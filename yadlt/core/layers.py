"""Restricted Boltzmann Machine TensorFlow implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class Layer(object):
    """Base Layer class."""

    def __init__(self):
        """Create a new layer instance."""
        raise NotImplementedError("Should have implemented this.")

    def forward(self, X):
        """Propagate X through the layer in forward direction."""
        raise NotImplementedError("Should have implemented this.")

    def backward(self, H):
        """Propagate H though the layer in backward direction."""
        raise NotImplementedError("Should have implemented this.")

    def get_train_parameters(self):
        """Return the trainable parameters of this layer."""
        raise NotImplementedError("Should have implemented this.")

    def get_parameters(self):
        """Return all the parameters of this layer."""
        raise NotImplementedError("Should have implemented this.")


class Linear(Layer):
    """Fully-Connected layer."""

    def __init__(self, shape=None, names=["W", "b"]):
        """Create a new layer instance."""
        self.names = names
        if shape:
            self.W = tf.Variable(
                tf.truncated_normal(shape=shape, stddev=0.1), name=names[0])
            self.b = tf.Variable(
                tf.constant(0.1, shape=shape[1]), name=names[1])

    def forward(self, X):
        """Forward propagate X through the fc layer."""
        return tf.add(tf.matmul(X, self.W), self.b)

    def backward(self, H):
        """Backward propagate H through the fc layer."""
        pass

    def get_train_parameters(self):
        """Return the trainable parameters of this layer."""
        with tf.Session() as sess:
            return {
                self.names[0]: sess.run(self.W),
                self.names[1]: sess.run(self.b)
            }

    def get_parameters(self):
        """Return all the parameters of this layer."""
        with tf.Session() as sess:
            return {
                self.names[0]: sess.run(self.W),
                self.names[1]: sess.run(self.b)
            }
